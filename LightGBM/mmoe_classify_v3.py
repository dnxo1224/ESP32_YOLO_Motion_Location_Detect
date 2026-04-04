"""
MMOE — Wi-Fi CSI Multi-task Classification (Zone + Action)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
v1 — Shared CNN + AvgPool 인코더
  - Per-RX 1D CNN + AdaptiveAvgPool (공유)
  - MMOE: 6 Experts + 2 Gates
  - [문제] AvgPool이 시간 정보 소멸 → Action 57%

v2 — Shared CNN + GRU 인코더
  - AvgPool → Bi-GRU 교체 (시간 순서 보존)
  - ENCODER_MODE = 'gru_shared'

v3 — Dual 인코더
  - Zone: CNN + AvgPool / Action: CNN + Bi-GRU
  - 공유 CNN backbone + 태스크별 temporal aggregator
  - ENCODER_MODE = 'dual'

v4 — Multi-scale CNN + Dual 인코더  ← 현재
  - 4-branch Multi-scale CNN (k=3/7/15/31) → 주파수별 분해
    k=3(고주파) k=7(중주파) k=15(저주파) k=31(Doppler)
  - Zone path  : MultiScaleCNN → AvgPool  (공간 지문)
  - Action path: MultiScaleCNN → Bi-GRU  (시간 다이나믹)
  - 서브샘플링 제거 (800 타임스텝 그대로 사용)
  - ENCODER_MODE = 'multiscale'
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

모드 전환 (상단 상수 하나만 변경):
    ENCODER_MODE = 'multiscale'   # v4 (현재)
    ENCODER_MODE = 'gru_shared'   # v2
    ENCODER_MODE = 'dual'         # v3

실험 구성:
  - test  : kjh, kms (2명 고정)
  - train : 나머지 11명

사전 실행:
    python preprocess.py   # 최초 1회

실행:
    python mmoe_classify.py
"""

import os
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
)

warnings.filterwarnings('ignore')

# ── 상수 ──────────────────────────────────────────────────────────────────────

NUM_FEATURES  = 166
NUM_RX        = 4

TEST_SUBJECTS = ['kjh', 'kms']
ACTION_NAMES  = ['handsup', 'sit', 'stand', 'walk']
ZONE_NAMES    = ['Zone0', 'Zone1', 'Zone2', 'Zone3']

PROCESSED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'processed')
RESULTS_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'mmoe')
WEIGHTS_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights')

# ── 모드 전환 ─────────────────────────────────────────────────────────────────
#   'multiscale' : v4 — Multi-scale CNN + Dual 인코더  (현재)
#   'gru_shared' : v2 — 공유 GRU 인코더
#   'dual'       : v3 — Dual 인코더 (단순 CNN backbone)
ENCODER_MODE  = 'multiscale'

# ── 하이퍼파라미터 ────────────────────────────────────────────────────────────

# Multi-scale CNN
MS_KERNELS    = [3, 7, 15, 31]   # 브랜치별 커널 크기 (고주파→Doppler)
MS_BRANCH_CH  = 64               # 브랜치별 채널 수
MS_OUT_CH     = 128              # Reduction 출력 채널

# 공통 인코더
CNN_CHANNELS  = [64, 128, 128]   # gru_shared / dual 용 CNN 채널
GRU_HIDDEN    = 128              # Bi-GRU 단방향 hidden
FUSION_DIM    = 256              # RX fusion 출력
NUM_EXPERTS   = 6
EXPERT_DIM    = 128
TOWER_HIDDEN  = 64
NUM_CLASSES   = 4

# 학습
BATCH_SIZE    = 32
LR            = 1e-3
WEIGHT_DECAY  = 1e-3
MAX_EPOCHS    = 100
PATIENCE      = 20
GRAD_CLIP     = 1.0
MIXUP_ALPHA   = 0.4
SEED          = 42


# ── 데이터 로딩 ──────────────────────────────────────────────────────────────

def load_processed(processed_dir):
    samples = []
    npz_files = sorted([f for f in os.listdir(processed_dir) if f.endswith('.npz')])
    if not npz_files:
        raise FileNotFoundError(
            f'No NPZ files found in {processed_dir}. Run preprocess.py first.'
        )
    for fname in tqdm(npz_files, desc='Loading processed data'):
        data = np.load(os.path.join(processed_dir, fname), allow_pickle=True)
        samples.append({
            'grids':   data['grids'],       # (4, 800, 166)
            'action':  int(data['action']),
            'zone':    int(data['zone']),
            'subject': str(data['subject']),
        })
    return samples


def split_by_subjects(samples, test_subjects):
    test_set = set(test_subjects)
    train = [s for s in samples if s['subject'] not in test_set]
    test  = [s for s in samples if s['subject'] in test_set]
    return train, test


class CSIDataset(Dataset):
    """
    Wi-Fi CSI 데이터셋.

    서브샘플링 없이 (4, 800, 166) 그대로 사용.
    각 모델이 내부적으로 시간 해상도를 처리함.
    """
    def __init__(self, grids, zones, actions, scaler=None, fit_scaler=False):
        # grids: (N, 4, 800, 166)
        N    = grids.shape[0]
        flat = grids.reshape(-1, NUM_FEATURES)   # (N*4*800, 166)
        if fit_scaler:
            assert scaler is not None
            scaler.fit(flat)
        if scaler is not None:
            flat = scaler.transform(flat).astype(np.float32)
        grids = flat.reshape(N, NUM_RX, 800, NUM_FEATURES)

        self.grids   = torch.from_numpy(grids)
        self.zones   = torch.from_numpy(np.array(zones,   dtype=np.int64))
        self.actions = torch.from_numpy(np.array(actions, dtype=np.int64))

    def __len__(self):
        return len(self.zones)

    def __getitem__(self, idx):
        return self.grids[idx], self.zones[idx], self.actions[idx]


def build_dataloaders(train_samples, test_samples):
    def collect(samples):
        grids   = np.stack([s['grids'] for s in samples])
        zones   = np.array([s['zone']   for s in samples])
        actions = np.array([s['action'] for s in samples])
        return grids, zones, actions

    tr_g, tr_z, tr_a = collect(train_samples)
    te_g, te_z, te_a = collect(test_samples)

    scaler   = StandardScaler()
    train_ds = CSIDataset(tr_g, tr_z, tr_a, scaler=scaler, fit_scaler=True)
    test_ds  = CSIDataset(te_g, te_z, te_a, scaler=scaler, fit_scaler=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True, drop_last=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=True)
    return train_loader, test_loader, scaler


# ── 공통 헬퍼 ─────────────────────────────────────────────────────────────────

def _rx_input(x):
    """(B, 4, T, 166) → (B*4, 166, T)"""
    B, R, T, C = x.shape
    return x.reshape(B * R, T, C).permute(0, 2, 1)   # (B*4, 166, T)


def _gru_aggregate(cnn_out, gru, proj):
    """
    (B*4, C, T) → Bi-GRU → (B*4, C)

    cnn_out : (B*4, C, T)
    gru     : nn.GRU(batch_first=True, bidirectional=True)
    proj    : nn.Linear(C*2, C)
    """
    seq = cnn_out.permute(0, 2, 1)             # (B*4, T, C)
    _, h_n = gru(seq)                          # h_n: (2, B*4, GRU_HIDDEN)
    h = torch.cat([h_n[0], h_n[1]], dim=-1)   # (B*4, GRU_HIDDEN*2)
    return proj(h)                             # (B*4, C)


def _build_fusion(in_dim=None):
    if in_dim is None:
        in_dim = NUM_RX * MS_OUT_CH
    return nn.Sequential(
        nn.Linear(in_dim, FUSION_DIM),
        nn.GELU(),
        nn.Dropout(0.3),
    )


def _build_experts():
    return nn.ModuleList([
        nn.Sequential(
            nn.Linear(FUSION_DIM, EXPERT_DIM), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(EXPERT_DIM, EXPERT_DIM), nn.GELU(),
        )
        for _ in range(NUM_EXPERTS)
    ])


def _build_tower():
    return nn.Sequential(
        nn.Linear(EXPERT_DIM, TOWER_HIDDEN), nn.GELU(), nn.Dropout(0.3),
        nn.Linear(TOWER_HIDDEN, NUM_CLASSES),
    )


def _expert_mixture(experts, gate_logits, fused):
    """
    experts     : ModuleList
    gate_logits : (B, NUM_EXPERTS)
    fused       : (B, FUSION_DIM)
    Returns     : (B, EXPERT_DIM)
    """
    weights = F.softmax(gate_logits, dim=-1)                      # (B, E)
    stacked = torch.stack([e(fused) for e in experts], dim=1)     # (B, E, D)
    return torch.einsum('be,bed->bd', weights, stacked)            # (B, D)


# ── Loss 헬퍼 ─────────────────────────────────────────────────────────────────

def _uncertainty_loss(zl, al, yz, ya, lvz, lva):
    lz = F.cross_entropy(zl, yz) / (2 * torch.exp(lvz)) + lvz / 2
    la = F.cross_entropy(al, ya) / (2 * torch.exp(lva)) + lva / 2
    return (lz + la).squeeze()


def _uncertainty_mixup_loss(zl, al, yza, yaa, yzb, yab, lam, lvz, lva):
    ce_z = lam * F.cross_entropy(zl, yza) + (1 - lam) * F.cross_entropy(zl, yzb)
    ce_a = lam * F.cross_entropy(al, yaa) + (1 - lam) * F.cross_entropy(al, yab)
    lz   = ce_z / (2 * torch.exp(lvz)) + lvz / 2
    la   = ce_a / (2 * torch.exp(lva)) + lva / 2
    return (lz + la).squeeze()


# ── v4: Multi-scale CNN Backbone ──────────────────────────────────────────────

class MultiScaleCNNBackbone(nn.Module):
    """
    4-branch Multi-scale 1D CNN.

    각 브랜치가 서로 다른 주파수 대역을 담당:
      k=3  → 고주파 패턴 (순간 변화)
      k=7  → 중주파 패턴 (bins 32-44)
      k=15 → 저주파 패턴 (bins 5-7)
      k=31 → 매우 저주파 (Doppler, 걷기/앉기 추세)

    입력: (B*4, 166, 800)
    출력: (B*4, 128, 25)
    """
    def __init__(self):
        super().__init__()

        # 4개 병렬 브랜치 (stride=4: 800 → 200)
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(NUM_FEATURES, MS_BRANCH_CH,
                          kernel_size=k, stride=4, padding=(k - 1) // 2),
                nn.BatchNorm1d(MS_BRANCH_CH),
                nn.GELU(),
            )
            for k in MS_KERNELS
        ])   # 각 브랜치: (B*4, 64, 200)

        total_ch = MS_BRANCH_CH * len(MS_KERNELS)   # 256

        # Reduction: 256ch, 200 → 128ch, 25
        self.reduction = nn.Sequential(
            nn.Conv1d(total_ch,  MS_OUT_CH, kernel_size=3, stride=4, padding=1),
            nn.BatchNorm1d(MS_OUT_CH), nn.GELU(),   # (B*4, 128, 50)
            nn.Conv1d(MS_OUT_CH, MS_OUT_CH, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(MS_OUT_CH), nn.GELU(),   # (B*4, 128, 25)
        )

    def forward(self, x):
        """x: (B*4, 166, 800) → (B*4, 128, 25)"""
        branch_outs = [b(x) for b in self.branches]   # 4 × (B*4, 64, 200)
        x = torch.cat(branch_outs, dim=1)              # (B*4, 256, 200)
        return self.reduction(x)                        # (B*4, 128, 25)


# ── v4: Multi-scale MMOE Model ────────────────────────────────────────────────

class MultiScaleMMOEModel(nn.Module):
    """
    v4 — Multi-scale CNN + Dual Aggregation + MMOE.

    Zone   path: MultiScaleCNN → AvgPool  (공간 지문)
    Action path: MultiScaleCNN → Bi-GRU  (시간 다이나믹)

    공유: Multi-scale CNN backbone
    분리: temporal aggregator, RX fusion, MMOE experts/gates
    """
    def __init__(self):
        super().__init__()

        # ── Multi-scale CNN (공유) ──
        self.cnn = MultiScaleCNNBackbone()

        # ── Zone: AvgPool ──
        self.zone_pool = nn.AdaptiveAvgPool1d(1)

        # ── Action: Bi-GRU ──
        self.action_gru  = nn.GRU(MS_OUT_CH, GRU_HIDDEN, num_layers=1,
                                   batch_first=True, bidirectional=True)
        self.action_proj = nn.Linear(GRU_HIDDEN * 2, MS_OUT_CH)

        # ── RX Fusion (분리) ──
        self.zone_fusion   = _build_fusion()   # 4*128 → 256
        self.action_fusion = _build_fusion()

        # ── MMOE: 분리 experts + gates ──
        self.zone_experts   = _build_experts()
        self.action_experts = _build_experts()
        self.gate_zone      = nn.Linear(FUSION_DIM, NUM_EXPERTS)
        self.gate_action    = nn.Linear(FUSION_DIM, NUM_EXPERTS)

        # ── Towers ──
        self.tower_zone   = _build_tower()
        self.tower_action = _build_tower()

        # ── Uncertainty Weighting ──
        self.log_var_zone   = nn.Parameter(torch.zeros(1))
        self.log_var_action = nn.Parameter(torch.zeros(1))

    def _encode(self, x):
        """(B, 4, 800, 166) → zone_fused(B, 256), action_fused(B, 256)"""
        B       = x.size(0)
        cnn_out = self.cnn(_rx_input(x))   # (B*4, 128, 25)

        # Zone: AvgPool
        z_feat  = self.zone_pool(cnn_out).squeeze(-1)   # (B*4, 128)
        z_fused = self.zone_fusion(
            z_feat.reshape(B, NUM_RX * MS_OUT_CH)
        )   # (B, 256)

        # Action: Bi-GRU
        a_feat  = _gru_aggregate(cnn_out, self.action_gru, self.action_proj)  # (B*4, 128)
        a_fused = self.action_fusion(
            a_feat.reshape(B, NUM_RX * MS_OUT_CH)
        )   # (B, 256)

        return z_fused, a_fused

    def forward(self, x):
        z_fused, a_fused = self._encode(x)
        z_mix = _expert_mixture(self.zone_experts,   self.gate_zone(z_fused),   z_fused)
        a_mix = _expert_mixture(self.action_experts, self.gate_action(a_fused), a_fused)
        return self.tower_zone(z_mix), self.tower_action(a_mix)

    def get_gate_weights(self, x):
        z_fused, a_fused = self._encode(x)
        return (
            F.softmax(self.gate_zone(z_fused),   dim=-1).mean(0).detach().cpu().numpy(),
            F.softmax(self.gate_action(a_fused), dim=-1).mean(0).detach().cpu().numpy(),
        )

    def compute_loss(self, zl, al, yz, ya):
        return _uncertainty_loss(zl, al, yz, ya, self.log_var_zone, self.log_var_action)

    def compute_mixup_loss(self, zl, al, yza, yaa, yzb, yab, lam):
        return _uncertainty_mixup_loss(zl, al, yza, yaa, yzb, yab, lam,
                                       self.log_var_zone, self.log_var_action)


# ── v2: Shared GRU MMOE Model ────────────────────────────────────────────────

class CNNBackbone(nn.Module):
    """단순 3-layer 1D CNN. gru_shared / dual 모드용."""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(NUM_FEATURES,    CNN_CHANNELS[0], kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(CNN_CHANNELS[0]), nn.GELU(),
            nn.Conv1d(CNN_CHANNELS[0], CNN_CHANNELS[1], kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(CNN_CHANNELS[1]), nn.GELU(),
            nn.Conv1d(CNN_CHANNELS[1], CNN_CHANNELS[2], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(CNN_CHANNELS[2]), nn.GELU(),
        )

    def forward(self, x):
        return self.layers(x)


class MMOEModel(nn.Module):
    """v2 — Zone/Action 공유 Bi-GRU 인코더."""
    def __init__(self):
        super().__init__()
        self.cnn      = CNNBackbone()
        self.gru      = nn.GRU(CNN_CHANNELS[-1], GRU_HIDDEN, num_layers=1,
                               batch_first=True, bidirectional=True)
        self.gru_proj = nn.Linear(GRU_HIDDEN * 2, CNN_CHANNELS[-1])
        self.rx_fusion = _build_fusion(NUM_RX * CNN_CHANNELS[-1])

        self.experts      = _build_experts()
        self.gate_zone    = nn.Linear(FUSION_DIM, NUM_EXPERTS)
        self.gate_action  = nn.Linear(FUSION_DIM, NUM_EXPERTS)
        self.tower_zone   = _build_tower()
        self.tower_action = _build_tower()
        self.log_var_zone   = nn.Parameter(torch.zeros(1))
        self.log_var_action = nn.Parameter(torch.zeros(1))

    def _encode(self, x):
        B    = x.size(0)
        feat = _gru_aggregate(self.cnn(_rx_input(x)), self.gru, self.gru_proj)
        return self.rx_fusion(feat.reshape(B, NUM_RX * CNN_CHANNELS[-1]))

    def forward(self, x):
        fused = self._encode(x)
        z_mix = _expert_mixture(self.experts, self.gate_zone(fused),   fused)
        a_mix = _expert_mixture(self.experts, self.gate_action(fused), fused)
        return self.tower_zone(z_mix), self.tower_action(a_mix)

    def get_gate_weights(self, x):
        fused = self._encode(x)
        return (F.softmax(self.gate_zone(fused),   dim=-1).mean(0).detach().cpu().numpy(),
                F.softmax(self.gate_action(fused), dim=-1).mean(0).detach().cpu().numpy())

    def compute_loss(self, zl, al, yz, ya):
        return _uncertainty_loss(zl, al, yz, ya, self.log_var_zone, self.log_var_action)

    def compute_mixup_loss(self, zl, al, yza, yaa, yzb, yab, lam):
        return _uncertainty_mixup_loss(zl, al, yza, yaa, yzb, yab, lam,
                                       self.log_var_zone, self.log_var_action)


# ── v3: Dual Encoder MMOE Model ───────────────────────────────────────────────

class DualMMOEModel(nn.Module):
    """v3 — Zone(AvgPool) / Action(Bi-GRU) 분리 인코더, 단순 CNN backbone."""
    def __init__(self):
        super().__init__()
        self.cnn         = CNNBackbone()
        self.zone_pool   = nn.AdaptiveAvgPool1d(1)
        self.action_gru  = nn.GRU(CNN_CHANNELS[-1], GRU_HIDDEN, num_layers=1,
                                   batch_first=True, bidirectional=True)
        self.action_proj = nn.Linear(GRU_HIDDEN * 2, CNN_CHANNELS[-1])

        self.zone_fusion   = _build_fusion(NUM_RX * CNN_CHANNELS[-1])
        self.action_fusion = _build_fusion(NUM_RX * CNN_CHANNELS[-1])

        self.zone_experts   = _build_experts()
        self.action_experts = _build_experts()
        self.gate_zone      = nn.Linear(FUSION_DIM, NUM_EXPERTS)
        self.gate_action    = nn.Linear(FUSION_DIM, NUM_EXPERTS)
        self.tower_zone     = _build_tower()
        self.tower_action   = _build_tower()
        self.log_var_zone   = nn.Parameter(torch.zeros(1))
        self.log_var_action = nn.Parameter(torch.zeros(1))

    def _encode(self, x):
        B       = x.size(0)
        cnn_out = self.cnn(_rx_input(x))
        z_feat  = self.zone_pool(cnn_out).squeeze(-1)
        a_feat  = _gru_aggregate(cnn_out, self.action_gru, self.action_proj)
        z_fused = self.zone_fusion(z_feat.reshape(B, NUM_RX * CNN_CHANNELS[-1]))
        a_fused = self.action_fusion(a_feat.reshape(B, NUM_RX * CNN_CHANNELS[-1]))
        return z_fused, a_fused

    def forward(self, x):
        z_fused, a_fused = self._encode(x)
        z_mix = _expert_mixture(self.zone_experts,   self.gate_zone(z_fused),   z_fused)
        a_mix = _expert_mixture(self.action_experts, self.gate_action(a_fused), a_fused)
        return self.tower_zone(z_mix), self.tower_action(a_mix)

    def get_gate_weights(self, x):
        z_fused, a_fused = self._encode(x)
        return (F.softmax(self.gate_zone(z_fused),   dim=-1).mean(0).detach().cpu().numpy(),
                F.softmax(self.gate_action(a_fused), dim=-1).mean(0).detach().cpu().numpy())

    def compute_loss(self, zl, al, yz, ya):
        return _uncertainty_loss(zl, al, yz, ya, self.log_var_zone, self.log_var_action)

    def compute_mixup_loss(self, zl, al, yza, yaa, yzb, yab, lam):
        return _uncertainty_mixup_loss(zl, al, yza, yaa, yzb, yab, lam,
                                       self.log_var_zone, self.log_var_action)


# ── 모델 팩토리 ───────────────────────────────────────────────────────────────

def build_model(mode=ENCODER_MODE):
    """ENCODER_MODE 에 따라 모델 생성."""
    if mode == 'multiscale':
        return MultiScaleMMOEModel()
    elif mode == 'gru_shared':
        return MMOEModel()
    elif mode == 'dual':
        return DualMMOEModel()
    else:
        raise ValueError(f"Unknown ENCODER_MODE: '{mode}'. "
                         f"Use 'multiscale' / 'gru_shared' / 'dual'.")


# ── 학습 유틸리티 ────────────────────────────────────────────────────────────

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def mixup_data(x, y_zone, y_action, alpha=MIXUP_ALPHA):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y_zone, y_action, y_zone[idx], y_action[idx], lam


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, n = 0.0, 0
    for x, y_zone, y_action in loader:
        x, y_zone, y_action = x.to(device), y_zone.to(device), y_action.to(device)
        x_mix, yza, yaa, yzb, yab, lam = mixup_data(x, y_zone, y_action)

        zl, al = model(x_mix)
        loss   = model.compute_mixup_loss(zl, al, yza, yaa, yzb, yab, lam)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        total_loss += loss.item(); n += 1

    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0; n = 0
    z_preds, z_labels, a_preds, a_labels = [], [], [], []
    gate_z_all, gate_a_all = [], []

    for x, y_zone, y_action in loader:
        x, y_zone, y_action = x.to(device), y_zone.to(device), y_action.to(device)
        zl, al = model(x)
        loss   = model.compute_loss(zl, al, y_zone, y_action)
        total_loss += loss.item(); n += 1

        z_preds.append(zl.argmax(1).cpu().numpy())
        z_labels.append(y_zone.cpu().numpy())
        a_preds.append(al.argmax(1).cpu().numpy())
        a_labels.append(y_action.cpu().numpy())

        gz, ga = model.get_gate_weights(x)
        gate_z_all.append(gz); gate_a_all.append(ga)

    z_preds   = np.concatenate(z_preds);  z_labels  = np.concatenate(z_labels)
    a_preds   = np.concatenate(a_preds);  a_labels  = np.concatenate(a_labels)
    zone_acc   = accuracy_score(z_labels, z_preds)
    action_acc = accuracy_score(a_labels, a_preds)
    gate_z_avg = np.stack(gate_z_all).mean(0)
    gate_a_avg = np.stack(gate_a_all).mean(0)

    return (total_loss / max(n, 1),
            zone_acc, action_acc,
            z_preds, z_labels, a_preds, a_labels,
            gate_z_avg, gate_a_avg)


# ── 시각화 ───────────────────────────────────────────────────────────────────

def save_confusion_matrix(y_true, y_pred, label_names, task_name, acc):
    cm   = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(f'{task_name}  (acc={acc * 100:.1f}%)')
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f'{task_name.lower()}_cm.png')
    fig.savefig(path, dpi=120); plt.close(fig)
    print(f'  Confusion matrix saved → {path}')


def plot_training_curves(history):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(epochs, history['train_loss'], label='Train', linewidth=1.5)
    axes[0].plot(epochs, history['test_loss'],  label='Test',  linewidth=1.5)
    axes[0].set(xlabel='Epoch', ylabel='Loss', title='Multi-task Loss')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, [a * 100 for a in history['zone_acc']],   'g-', linewidth=1.5)
    axes[1].set(xlabel='Epoch', ylabel='Accuracy (%)', title='Zone Accuracy')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, [a * 100 for a in history['action_acc']], 'r-', linewidth=1.5)
    axes[2].set(xlabel='Epoch', ylabel='Accuracy (%)', title='Action Accuracy')
    axes[2].grid(True, alpha=0.3)

    plt.suptitle(f'MMOE Training Curves  [{ENCODER_MODE}]', fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'training_curves.png')
    fig.savefig(path, dpi=120, bbox_inches='tight'); plt.close(fig)
    print(f'  Training curves saved → {path}')


def save_gate_analysis(gate_zone_avg, gate_action_avg):
    experts = [f'E{i}' for i in range(NUM_EXPERTS)]
    x = np.arange(NUM_EXPERTS); w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    bz = ax.bar(x - w / 2, gate_zone_avg,   w, label='Zone',   color='#4C72B0')
    ba = ax.bar(x + w / 2, gate_action_avg, w, label='Action', color='#DD8452')
    for bar in list(bz) + list(ba):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
    ax.set(xlabel='Expert', ylabel='Avg Gate Weight',
           title=f'MMOE Gate Analysis  [{ENCODER_MODE}]')
    ax.set_xticks(x); ax.set_xticklabels(experts); ax.legend()
    ax.set_ylim(0, max(gate_zone_avg.max(), gate_action_avg.max()) * 1.3)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'gate_analysis.png')
    fig.savefig(path, dpi=120); plt.close(fig)
    print(f'  Gate analysis saved → {path}')


def save_multiscale_branch_analysis(model, loader, device):
    """Multi-scale 브랜치별 활성화 평균 저장 (multiscale 모드 전용)."""
    if not isinstance(model, MultiScaleMMOEModel):
        return

    model.eval()
    branch_acts = [[] for _ in MS_KERNELS]

    with torch.no_grad():
        for x, _, _ in loader:
            x = x.to(device)
            inp = _rx_input(x)                         # (B*4, 166, 800)
            for i, branch in enumerate(model.cnn.branches):
                act = branch(inp).abs().mean().item()  # 평균 활성화
                branch_acts[i].append(act)

    avgs = [np.mean(a) for a in branch_acts]
    labels = [f'k={k}' for k in MS_KERNELS]
    desc   = ['고주파', '중주파', '저주파', 'Doppler']

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, avgs, color=['#4C72B0', '#DD8452', '#55A868', '#C44E52'])
    for bar, val, d in zip(bars, avgs, desc):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f'{d}\n{val:.4f}', ha='center', va='bottom', fontsize=9)
    ax.set(xlabel='Branch (kernel size)', ylabel='Mean Activation',
           title='Multi-scale Branch Activation')
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'multiscale_branch_activation.png')
    fig.savefig(path, dpi=120); plt.close(fig)
    print(f'  Branch activation saved → {path}')


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main():
    set_seed(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    device = get_device()
    print(f'Device       : {device}')
    print(f'Encoder mode : {ENCODER_MODE}')

    # 데이터
    print('\nLoading preprocessed data ...')
    all_samples = load_processed(PROCESSED_DIR)
    print(f'Total recordings: {len(all_samples)}')

    train_samples, test_samples = split_by_subjects(all_samples, TEST_SUBJECTS)
    print(f'Train: {len(train_samples)}  |  Test: {len(test_samples)}  '
          f'(test subjects: {TEST_SUBJECTS})')

    train_loader, test_loader, _ = build_dataloaders(train_samples, test_samples)

    # 모델
    model   = build_model(ENCODER_MODE).to(device)
    n_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Parameters   : {n_param:,}')

    if ENCODER_MODE == 'multiscale':
        print(f'MS kernels   : {MS_KERNELS}  (고주파→Doppler)')

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

    # 학습
    print(f'\n{"="*60}')
    print(f'Training MMOE [{ENCODER_MODE}]  (max {MAX_EPOCHS} epochs, patience={PATIENCE})')
    print(f'{"="*60}')

    history      = {'train_loss': [], 'test_loss': [], 'zone_acc': [], 'action_acc': []}
    best_metric  = 0.0
    best_epoch   = 0
    patience_cnt = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        scheduler.step()

        (test_loss, zone_acc, action_acc,
         z_pred, z_true, a_pred, a_true,
         g_zone, g_action) = evaluate(model, test_loader, device)

        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['zone_acc'].append(zone_acc)
        history['action_acc'].append(action_acc)

        mean_acc = (zone_acc + action_acc) / 2
        print(f'  Epoch {epoch:3d}/{MAX_EPOCHS}  '
              f'loss={train_loss:.4f}/{test_loss:.4f}  '
              f'Zone={zone_acc*100:.1f}%  Action={action_acc*100:.1f}%  '
              f'Mean={mean_acc*100:.1f}%  '
              f'lr={optimizer.param_groups[0]["lr"]:.2e}  '
              f'σ²_z={torch.exp(model.log_var_zone).item():.3f}  '
              f'σ²_a={torch.exp(model.log_var_action).item():.3f}')

        if mean_acc > best_metric:
            best_metric  = mean_acc
            best_epoch   = epoch
            patience_cnt = 0
            torch.save(model.state_dict(),
                       os.path.join(WEIGHTS_DIR, f'mmoe_{ENCODER_MODE}_best.pt'))
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f'\n  Early stopping at epoch {epoch} '
                      f'(best={best_epoch}, mean={best_metric*100:.1f}%)')
                break

    # 최종 평가
    print(f'\n{"="*60}')
    print(f'Final Evaluation (best epoch={best_epoch})')
    print(f'{"="*60}')

    model.load_state_dict(
        torch.load(os.path.join(WEIGHTS_DIR, f'mmoe_{ENCODER_MODE}_best.pt'),
                   map_location=device, weights_only=True)
    )
    (_, zone_acc, action_acc,
     z_pred, z_true, a_pred, a_true,
     g_zone, g_action) = evaluate(model, test_loader, device)

    print(f'\n  Zone Accuracy  : {zone_acc * 100:.2f}%')
    print(classification_report(z_true, z_pred, target_names=ZONE_NAMES, zero_division=0))
    print(f'  Action Accuracy: {action_acc * 100:.2f}%')
    print(classification_report(a_true, a_pred, target_names=ACTION_NAMES, zero_division=0))

    # 시각화
    save_confusion_matrix(z_true, z_pred, ZONE_NAMES,   'Zone',   zone_acc)
    save_confusion_matrix(a_true, a_pred, ACTION_NAMES, 'Action', action_acc)
    plot_training_curves(history)
    save_gate_analysis(g_zone, g_action)
    save_multiscale_branch_analysis(model, test_loader, device)

    # 요약 CSV
    summary_path = os.path.join(RESULTS_DIR, 'accuracy_summary.csv')
    with open(summary_path, 'w') as f:
        f.write('mode,task,test_subjects,accuracy,best_epoch\n')
        f.write(f'{ENCODER_MODE},zone,"{"+".join(TEST_SUBJECTS)}",'
                f'{zone_acc*100:.4f},{best_epoch}\n')
        f.write(f'{ENCODER_MODE},action,"{"+".join(TEST_SUBJECTS)}",'
                f'{action_acc*100:.4f},{best_epoch}\n')
    print(f'\n  Summary saved → {summary_path}')

    print(f'\n{"="*60}')
    print('RESULTS')
    print(f'{"="*60}')
    print(f'  Mode           : {ENCODER_MODE}')
    print(f'  Zone   Accuracy: {zone_acc * 100:.2f}%')
    print(f'  Action Accuracy: {action_acc * 100:.2f}%')
    print(f'  Mean   Accuracy: {(zone_acc + action_acc) / 2 * 100:.2f}%')
    print(f'  Best Epoch     : {best_epoch}')


if __name__ == '__main__':
    main()
