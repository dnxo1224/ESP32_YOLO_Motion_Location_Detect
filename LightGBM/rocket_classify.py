"""
rocket_classify.py — ROCKET 기반 Wi-Fi CSI 분류 실험

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[A] 단독 ROCKET + Ridge Regression  (EXPERIMENT = 'standalone')
    - Raw CSI (N, 664, 800) → 랜덤 Conv 커널 → max + PPV 피쳐
    - DL 학습 없음, Ridge Regression으로 Zone / Action 각각 분류
    - 빠른 베이스라인, 해석 가능

[B] DUAL + ROCKET MMOE              (EXPERIMENT = 'mmoe_rocket')
    - 공유 CNN backbone
    - Zone   path : CNN → AvgPool  (공간 지문)
    - Action path : CNN → ROCKET   (주파수 에너지 분포, Bi-GRU 대체)
    - 하류(Fusion, Experts, Gates, Towers)만 학습
    - ROCKET 커널 가중치는 고정 (requires_grad=False)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ROCKET 원리:
    각 랜덤 커널마다 Conv1d 적용 후 두 가지 피쳐 추출:
      - max   : 전체 시간에서 최댓값     (패턴 존재 여부)
      - PPV   : 양수 비율               (패턴 지속 시간)
    커널 파라미터: 크기 ∈ {7,9,11,13,15,17,19,21,23}, dilation 랜덤, bias 균등분포

참고: Dempster et al., 2020. ROCKET: Exceptionally fast and accurate time series classification.

실행:
    EXPERIMENT = 'standalone'   → python rocket_classify.py
    EXPERIMENT = 'mmoe_rocket'  → python rocket_classify.py
"""

import os
import math
import random
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
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
)

warnings.filterwarnings('ignore')

# ── 실험 선택 ──────────────────────────────────────────────────────────────────
#   'standalone'  : ROCKET + Ridge Regression (DL 없음)
#   'mmoe_rocket' : DUAL + ROCKET MMOE (DL)
EXPERIMENT = 'standalone'

# ── 상수 ──────────────────────────────────────────────────────────────────────
NUM_FEATURES  = 166
NUM_RX        = 4
T_RAW         = 800   # 전처리 후 타임스텝
CNN_OUT_CH    = 128   # CNN backbone 출력 채널
CNN_OUT_T     = 100   # CNN 출력 시간 길이 (800 // stride=2^3)

TEST_SUBJECTS = ['kjh', 'kms']
ACTION_NAMES  = ['handsup', 'sit', 'stand', 'walk']
ZONE_NAMES    = ['Zone0', 'Zone1', 'Zone2', 'Zone3']

PROCESSED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'processed')
RESULTS_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'rocket')
WEIGHTS_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights')

# ── ROCKET 하이퍼파라미터 ──────────────────────────────────────────────────────
ROCKET_KERNEL_SIZES     = [7, 9, 11, 13, 15, 17, 19, 21, 23]
ROCKET_N_KERNELS_MMOE   = 500    # MMOE 내부 ROCKET (CNN 출력에 적용)
ROCKET_N_KERNELS_ALONE  = 1000   # 단독 ROCKET (raw CSI에 직접 적용)

# ── MMOE 하이퍼파라미터 ───────────────────────────────────────────────────────
CNN_CHANNELS  = [64, 128, 128]
GRU_HIDDEN    = 128
FUSION_DIM    = 256
NUM_EXPERTS   = 6
EXPERT_DIM    = 128
TOWER_HIDDEN  = 64
NUM_CLASSES   = 4

BATCH_SIZE    = 32
LR            = 1e-3
WEIGHT_DECAY  = 1e-3
MAX_EPOCHS    = 100
PATIENCE      = 20
GRAD_CLIP     = 1.0
MIXUP_ALPHA   = 0.4
SEED          = 42


# ══════════════════════════════════════════════════════════════════════════════
# 데이터 로딩 (mmoe_classify.py 와 동일)
# ══════════════════════════════════════════════════════════════════════════════

def load_processed(processed_dir):
    samples = []
    npz_files = sorted([f for f in os.listdir(processed_dir) if f.endswith('.npz')])
    if not npz_files:
        raise FileNotFoundError(f'No NPZ files in {processed_dir}. Run preprocess.py first.')
    for fname in tqdm(npz_files, desc='Loading data'):
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
    return ([s for s in samples if s['subject'] not in test_set],
            [s for s in samples if s['subject'] in test_set])


def collect(samples):
    grids   = np.stack([s['grids'] for s in samples]).astype(np.float32)
    zones   = np.array([s['zone']   for s in samples], dtype=np.int64)
    actions = np.array([s['action'] for s in samples], dtype=np.int64)
    return grids, zones, actions


class CSIDataset(Dataset):
    def __init__(self, grids, zones, actions, scaler=None, fit_scaler=False):
        N    = grids.shape[0]
        flat = grids.reshape(-1, NUM_FEATURES)
        if fit_scaler:
            scaler.fit(flat)
        if scaler is not None:
            flat = scaler.transform(flat).astype(np.float32)
        grids = flat.reshape(N, NUM_RX, T_RAW, NUM_FEATURES)
        self.grids   = torch.from_numpy(grids)
        self.zones   = torch.from_numpy(zones)
        self.actions = torch.from_numpy(actions)

    def __len__(self):  return len(self.zones)
    def __getitem__(self, idx):
        return self.grids[idx], self.zones[idx], self.actions[idx]


def build_dataloaders(train_samples, test_samples):
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


# ══════════════════════════════════════════════════════════════════════════════
# ROCKET Transform
# ══════════════════════════════════════════════════════════════════════════════

class RocketTransform(nn.Module):
    """
    ROCKET: RandOm Convolutional KErnel Transform (Dempster et al., 2020)

    랜덤 1D Conv 커널을 고정 특징 추출기로 사용.
    커널 가중치는 학습 안 함 (requires_grad=False).

    각 커널 → Conv1d 적용 후:
      max  : 전체 시간 슬롯 중 최댓값   → 패턴이 얼마나 강하게 나타나는가
      PPV  : 출력 중 양수 비율          → 패턴이 얼마나 자주 나타나는가
    → 커널당 2개 피쳐, 총 n_features = 2 × n_kernels

    커널 설정:
      - 크기: ROCKET_KERNEL_SIZES에서 균등 분배
      - dilation: 2^U  (U ~ Uniform[0, floor(log2(seq_len/k))])
      - 가중치: N(0,1) + 평균 제거 (mean-subtracted per filter)
      - 편향: Uniform(-1, 1)
      - 패딩: half-padding (출력 길이 ≈ 입력 길이 유지)
    """

    def __init__(self, in_channels: int, seq_len: int, n_kernels: int = 500):
        super().__init__()
        self.in_channels = in_channels
        self.seq_len     = seq_len
        self.n_kernels   = n_kernels

        n_sizes  = len(ROCKET_KERNEL_SIZES)
        n_base   = n_kernels // n_sizes   # 크기별 기본 커널 수

        self.convs              = nn.ModuleList()
        self._kernels_per_conv  = []   # 각 Conv1d의 출력 채널 수

        for i, k_size in enumerate(ROCKET_KERNEL_SIZES):
            # 마지막 그룹에 나머지 할당
            n = n_base if i < n_sizes - 1 else (n_kernels - n_base * (n_sizes - 1))
            if n <= 0:
                continue

            # 랜덤 dilation (시퀀스 길이에 맞게 상한 제한)
            max_exp  = max(0, int(math.floor(math.log2(max(seq_len / k_size, 1)))))
            dilation = 2 ** random.randint(0, min(max_exp, 3))
            padding  = ((k_size - 1) * dilation) // 2   # half-padding

            conv = nn.Conv1d(in_channels, n, k_size,
                             dilation=dilation, padding=padding, bias=True)

            # 가중치: 정규분포 → 평균 제거 (ROCKET 논문 방식)
            nn.init.normal_(conv.weight)
            with torch.no_grad():
                # 커널 시간 축(dim=-1) 기준 평균 제거
                conv.weight -= conv.weight.mean(dim=-1, keepdim=True)

            # 편향: 균등분포
            nn.init.uniform_(conv.bias, -1.0, 1.0)

            # ★ 가중치 동결 — ROCKET 핵심: 커널은 학습하지 않음
            for param in conv.parameters():
                param.requires_grad = False

            self.convs.append(conv)
            self._kernels_per_conv.append(n)

        # 전체 피쳐 수: 각 커널마다 max + PPV = 2개
        self.n_features = sum(self._kernels_per_conv) * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T)
        반환: (B, n_features)   — max와 PPV를 이어 붙인 벡터
        """
        parts = []
        for conv in self.convs:
            out = conv(x)                               # (B, n, T')
            max_val = out.max(dim=-1).values            # (B, n)  — 최댓값
            ppv     = (out > 0).float().mean(dim=-1)   # (B, n)  — 양수 비율
            parts.append(torch.cat([max_val, ppv], dim=-1))   # (B, 2n)
        return torch.cat(parts, dim=-1)                 # (B, n_features)

    def extra_repr(self):
        return (f'in_channels={self.in_channels}, seq_len={self.seq_len}, '
                f'n_kernels={self.n_kernels}, n_features={self.n_features}')


# ══════════════════════════════════════════════════════════════════════════════
# 공통 헬퍼 (mmoe_classify.py 와 동일)
# ══════════════════════════════════════════════════════════════════════════════

def _rx_input(x):
    """(B, 4, T, 166) → (B*4, 166, T)"""
    B, R, T, C = x.shape
    return x.reshape(B * R, T, C).permute(0, 2, 1)


def _build_fusion(in_dim):
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
    """gate softmax 가중합으로 Expert 혼합."""
    weights = F.softmax(gate_logits, dim=-1)                   # (B, E)
    stacked = torch.stack([e(fused) for e in experts], dim=1)  # (B, E, D)
    return torch.einsum('be,bed->bd', weights, stacked)         # (B, D)


def _uncertainty_loss(zl, al, yz, ya, lvz, lva):
    lz = F.cross_entropy(zl, yz) / (2 * torch.exp(lvz)) + lvz / 2
    la = F.cross_entropy(al, ya) / (2 * torch.exp(lva)) + lva / 2
    return (lz + la).squeeze()


def _uncertainty_mixup_loss(zl, al, yza, yaa, yzb, yab, lam, lvz, lva):
    ce_z = lam * F.cross_entropy(zl, yza) + (1 - lam) * F.cross_entropy(zl, yzb)
    ce_a = lam * F.cross_entropy(al, yaa) + (1 - lam) * F.cross_entropy(al, yab)
    return (ce_z / (2 * torch.exp(lvz)) + lvz / 2
          + ce_a / (2 * torch.exp(lva)) + lva / 2).squeeze()


# ══════════════════════════════════════════════════════════════════════════════
# CNN Backbone (mmoe_classify.py 의 CNNBackbone 과 동일)
# ══════════════════════════════════════════════════════════════════════════════

class CNNBackbone(nn.Module):
    """3-layer 1D CNN.  (B*4, 166, 800) → (B*4, 128, 100)"""
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
    def forward(self, x): return self.layers(x)


# ══════════════════════════════════════════════════════════════════════════════
# [B] DUAL + ROCKET MMOE 모델
# ══════════════════════════════════════════════════════════════════════════════

class RocketDualMMOEModel(nn.Module):
    """
    DUAL + ROCKET MMOE.

    Zone   path: 공유 CNN → AdaptiveAvgPool  (공간 지문)
    Action path: 공유 CNN → ROCKET           (주파수 에너지 분포)
                           ↑ Bi-GRU 대체

    ROCKET 커널은 학습하지 않음.
    학습 파라미터: CNN backbone + 두 RX Fusion + 두 Expert 집합 + Gates + Towers
    """

    def __init__(self):
        super().__init__()

        # ── 공유 CNN Backbone ──
        self.cnn = CNNBackbone()   # (B*4, 166, 800) → (B*4, 128, 100)

        # ── Zone path: AvgPool ──
        self.zone_pool = nn.AdaptiveAvgPool1d(1)   # (B*4, 128, 100) → (B*4, 128)

        # ── Action path: ROCKET ──
        # CNN 출력 (B*4, 128, 100) 에 적용
        self.rocket = RocketTransform(
            in_channels=CNN_OUT_CH,
            seq_len=CNN_OUT_T,
            n_kernels=ROCKET_N_KERNELS_MMOE,
        )
        # self.rocket.n_features = ROCKET_N_KERNELS_MMOE * 2

        # ── RX Fusion (분리) ──
        zone_in   = NUM_RX * CNN_OUT_CH            # 4 × 128 = 512
        action_in = NUM_RX * self.rocket.n_features  # 4 × (n_kernels×2)
        self.zone_fusion   = _build_fusion(zone_in)
        self.action_fusion = _build_fusion(action_in)

        # ── MMOE: 분리 Experts + Gates ──
        self.zone_experts   = _build_experts()
        self.action_experts = _build_experts()
        self.gate_zone      = nn.Linear(FUSION_DIM, NUM_EXPERTS)
        self.gate_action    = nn.Linear(FUSION_DIM, NUM_EXPERTS)

        # ── Task Towers ──
        self.tower_zone   = _build_tower()
        self.tower_action = _build_tower()

        # ── Uncertainty Weighting (Kendall 2018) ──
        self.log_var_zone   = nn.Parameter(torch.zeros(1))
        self.log_var_action = nn.Parameter(torch.zeros(1))

    def _encode(self, x):
        """(B, 4, 800, 166) → zone_fused(B, 256), action_fused(B, 256)"""
        B       = x.size(0)
        cnn_out = self.cnn(_rx_input(x))   # (B*4, 128, 100)

        # Zone: AvgPool → 공간 지문
        z_feat  = self.zone_pool(cnn_out).squeeze(-1)          # (B*4, 128)
        z_fused = self.zone_fusion(
            z_feat.reshape(B, NUM_RX * CNN_OUT_CH)             # (B, 512)
        )                                                       # (B, 256)

        # Action: ROCKET → 주파수 에너지 분포
        a_feat  = self.rocket(cnn_out)                         # (B*4, n_features)
        a_fused = self.action_fusion(
            a_feat.reshape(B, NUM_RX * self.rocket.n_features) # (B, 4*n_feat)
        )                                                       # (B, 256)

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


# ══════════════════════════════════════════════════════════════════════════════
# 학습 유틸리티
# ══════════════════════════════════════════════════════════════════════════════

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():    return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def mixup_data(x, y_zone, y_action, alpha=MIXUP_ALPHA):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y_zone, y_action, y_zone[idx], y_action[idx], lam


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total, n = 0.0, 0
    for x, yz, ya in loader:
        x, yz, ya = x.to(device), yz.to(device), ya.to(device)
        x_mix, yza, yaa, yzb, yab, lam = mixup_data(x, yz, ya)
        zl, al = model(x_mix)
        loss   = model.compute_mixup_loss(zl, al, yza, yaa, yzb, yab, lam)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        total += loss.item(); n += 1
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total, n = 0.0, 0
    z_preds, z_labels, a_preds, a_labels = [], [], [], []
    gate_z_all, gate_a_all = [], []
    for x, yz, ya in loader:
        x, yz, ya = x.to(device), yz.to(device), ya.to(device)
        zl, al = model(x)
        total += model.compute_loss(zl, al, yz, ya).item(); n += 1
        z_preds.append(zl.argmax(1).cpu().numpy())
        z_labels.append(yz.cpu().numpy())
        a_preds.append(al.argmax(1).cpu().numpy())
        a_labels.append(ya.cpu().numpy())
        gz, ga = model.get_gate_weights(x)
        gate_z_all.append(gz); gate_a_all.append(ga)

    z_preds  = np.concatenate(z_preds);  z_labels = np.concatenate(z_labels)
    a_preds  = np.concatenate(a_preds);  a_labels = np.concatenate(a_labels)
    return (total / max(n, 1),
            accuracy_score(z_labels, z_preds),
            accuracy_score(a_labels, a_preds),
            z_preds, z_labels, a_preds, a_labels,
            np.stack(gate_z_all).mean(0),
            np.stack(gate_a_all).mean(0))


# ══════════════════════════════════════════════════════════════════════════════
# 시각화
# ══════════════════════════════════════════════════════════════════════════════

def save_confusion_matrix(y_true, y_pred, label_names, task_name, acc):
    cm   = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=label_names)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(f'{task_name}  (acc={acc*100:.1f}%)')
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f'{task_name.lower()}_cm.png')
    fig.savefig(path, dpi=120); plt.close(fig)
    print(f'  Confusion matrix → {path}')


def plot_training_curves(history):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ep = range(1, len(history['train_loss']) + 1)
    axes[0].plot(ep, history['train_loss'], label='Train')
    axes[0].plot(ep, history['test_loss'],  label='Test')
    axes[0].set(title='Loss', xlabel='Epoch'); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(ep, [a*100 for a in history['zone_acc']],   'g-')
    axes[1].set(title='Zone Accuracy (%)', xlabel='Epoch'); axes[1].grid(alpha=0.3)
    axes[2].plot(ep, [a*100 for a in history['action_acc']], 'r-')
    axes[2].set(title='Action Accuracy (%)', xlabel='Epoch'); axes[2].grid(alpha=0.3)
    plt.suptitle('DUAL + ROCKET MMOE  Training Curves', fontsize=13, y=1.02)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'training_curves.png')
    fig.savefig(path, dpi=120, bbox_inches='tight'); plt.close(fig)
    print(f'  Training curves → {path}')


def save_gate_analysis(gate_zone_avg, gate_action_avg):
    labels = [f'E{i+1}' for i in range(NUM_EXPERTS)]
    x = np.arange(NUM_EXPERTS); w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    bz = ax.bar(x - w/2, gate_zone_avg,   w, label='Zone',   color='#4C72B0')
    ba = ax.bar(x + w/2, gate_action_avg, w, label='Action', color='#DD8452')
    for bar in list(bz) + list(ba):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
    ax.set(xlabel='Expert', ylabel='Avg Gate Weight',
           title='DUAL+ROCKET MMOE  Gate Analysis')
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.legend()
    ax.set_ylim(0, max(gate_zone_avg.max(), gate_action_avg.max()) * 1.3)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'gate_analysis.png')
    fig.savefig(path, dpi=120); plt.close(fig)
    print(f'  Gate analysis → {path}')


# ══════════════════════════════════════════════════════════════════════════════
# [A] 단독 ROCKET 실험
# ══════════════════════════════════════════════════════════════════════════════

def run_standalone_rocket(train_samples, test_samples):
    """
    ROCKET + Ridge Regression (DL 없음).

    파이프라인:
      (N, 4, 800, 166)
        → StandardScaler
        → reshape: (N, 4×166=664, 800)   ← 664채널 다변량 시계열
        → RocketTransform(664, 800, n_kernels=1000)
        → (N, 2000)   ← 커널당 max + PPV
        → StandardScaler (피쳐 스케일링)
        → RidgeClassifierCV  (Zone)
        → RidgeClassifierCV  (Action)

    각 태스크를 독립적으로 분류함 (MMOE처럼 공유 표현 없음).
    """
    print('\n' + '='*60)
    print('[A] 단독 ROCKET + Ridge Regression')
    print('='*60)

    tr_g, tr_z, tr_a = collect(train_samples)
    te_g, te_z, te_a = collect(test_samples)
    N_tr, N_te = len(tr_g), len(te_g)

    # ── StandardScaler (채널 기준, mmoe 와 동일) ──
    scaler = StandardScaler()
    flat_tr = tr_g.reshape(-1, NUM_FEATURES)
    scaler.fit(flat_tr)
    tr_g = scaler.transform(tr_g.reshape(-1, NUM_FEATURES)).reshape(N_tr, NUM_RX, T_RAW, NUM_FEATURES)
    te_g = scaler.transform(te_g.reshape(-1, NUM_FEATURES)).reshape(N_te, NUM_RX, T_RAW, NUM_FEATURES)

    # ── (N, 4, 800, 166) → (N, 664, 800): 4 RX × 166채널을 시계열 축으로 합침 ──
    # transpose: (N, 4, 800, 166) → (N, 4, 166, 800) → reshape → (N, 664, 800)
    tr_mv = tr_g.transpose(0, 1, 3, 2).reshape(N_tr, NUM_RX * NUM_FEATURES, T_RAW)
    te_mv = te_g.transpose(0, 1, 3, 2).reshape(N_te, NUM_RX * NUM_FEATURES, T_RAW)
    print(f'  다변량 시계열 shape: {tr_mv.shape}  (N, 664채널, 800타임스텝)')

    # ── ROCKET 특징 추출 ──
    set_seed(SEED)
    rocket = RocketTransform(
        in_channels=NUM_RX * NUM_FEATURES,   # 664
        seq_len=T_RAW,                        # 800
        n_kernels=ROCKET_N_KERNELS_ALONE,     # 1000
    )
    rocket.eval()
    print(f'  ROCKET 커널: {ROCKET_N_KERNELS_ALONE}개  →  피쳐 수: {rocket.n_features}')

    def extract(data_np, batch=32):
        t = torch.from_numpy(data_np.astype(np.float32))
        feats = []
        with torch.no_grad():
            for i in range(0, len(t), batch):
                feats.append(rocket(t[i:i+batch]).numpy())
        return np.concatenate(feats, axis=0)

    print('  특징 추출 중 (train)...')
    tr_feat = extract(tr_mv)
    print('  특징 추출 중 (test)...')
    te_feat = extract(te_mv)
    print(f'  피쳐 shape: {tr_feat.shape}')

    # ── 피쳐 스케일링 (Ridge 입력) ──
    feat_scaler = StandardScaler()
    tr_feat = feat_scaler.fit_transform(tr_feat)
    te_feat = feat_scaler.transform(te_feat)

    # ── Ridge Regression (Zone) ──
    print('\n  Zone 분류 학습 (RidgeClassifierCV)...')
    ridge_z = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), cv=5)
    ridge_z.fit(tr_feat, tr_z)
    z_preds    = ridge_z.predict(te_feat)
    zone_acc   = accuracy_score(te_z, z_preds)

    # ── Ridge Regression (Action) ──
    print('  Action 분류 학습 (RidgeClassifierCV)...')
    ridge_a = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), cv=5)
    ridge_a.fit(tr_feat, tr_a)
    a_preds    = ridge_a.predict(te_feat)
    action_acc = accuracy_score(te_a, a_preds)

    # ── 결과 출력 ──
    print(f'\n  ┌─ 결과 ─────────────────────────────┐')
    print(f'  │  Zone   정확도: {zone_acc*100:6.2f}%              │')
    print(f'  │  Action 정확도: {action_acc*100:6.2f}%              │')
    print(f'  └────────────────────────────────────┘')

    print('\n  [Zone] 분류 리포트')
    print(classification_report(te_z, z_preds, target_names=ZONE_NAMES))
    print('\n  [Action] 분류 리포트')
    print(classification_report(te_a, a_preds, target_names=ACTION_NAMES))

    # ── Confusion Matrix 저장 ──
    save_confusion_matrix(te_z, z_preds,   ZONE_NAMES,   'standalone_rocket_zone',   zone_acc)
    save_confusion_matrix(te_a, a_preds,   ACTION_NAMES, 'standalone_rocket_action', action_acc)

    return zone_acc, action_acc


# ══════════════════════════════════════════════════════════════════════════════
# [B] DUAL + ROCKET MMOE 학습
# ══════════════════════════════════════════════════════════════════════════════

def run_mmoe_rocket(train_samples, test_samples):
    print('\n' + '='*60)
    print('[B] DUAL + ROCKET MMOE')
    print('='*60)

    device = get_device()
    print(f'  Device: {device}')

    train_loader, test_loader, _ = build_dataloaders(train_samples, test_samples)

    # ── 모델 생성 ──
    set_seed(SEED)   # ROCKET 커널 초기화 재현성
    model = RocketDualMMOEModel().to(device)

    n_total  = sum(p.numel() for p in model.parameters())
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    n_train  = n_total - n_frozen
    print(f'  파라미터: 전체={n_total:,}  │  ROCKET(고정)={n_frozen:,}  │  학습={n_train:,}')
    print(f'  ROCKET 피쳐: {model.rocket.n_features}개 / RX  '
          f'→  Action Fusion 입력: {NUM_RX * model.rocket.n_features}')

    # ── 옵티마이저 (학습 파라미터만) ──
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

    best_acc   = 0.0
    best_epoch = 0
    patience_cnt = 0
    history = {'train_loss': [], 'test_loss': [], 'zone_acc': [], 'action_acc': []}
    weight_path = os.path.join(WEIGHTS_DIR, 'mmoe_rocket_best.pt')

    print(f'\n  학습 시작 (최대 {MAX_EPOCHS} epoch, patience={PATIENCE})')
    print(f'  {"Epoch":>6}  {"TrainLoss":>10}  {"TestLoss":>10}  '
          f'{"Zone%":>7}  {"Action%":>8}')
    print('  ' + '-'*52)

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        (test_loss, z_acc, a_acc,
         z_preds, z_labels, a_preds, a_labels,
         gate_z, gate_a) = evaluate(model, test_loader, device)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['zone_acc'].append(z_acc)
        history['action_acc'].append(a_acc)

        combined = (z_acc + a_acc) / 2
        if combined > best_acc:
            best_acc   = combined
            best_epoch = epoch
            patience_cnt = 0
            torch.save(model.state_dict(), weight_path)
        else:
            patience_cnt += 1

        marker = ' ★' if patience_cnt == 0 else ''
        print(f'  {epoch:>6}  {train_loss:>10.4f}  {test_loss:>10.4f}  '
              f'{z_acc*100:>6.2f}%  {a_acc*100:>7.2f}%{marker}')

        if patience_cnt >= PATIENCE:
            print(f'\n  Early stopping at epoch {epoch}  (best: {best_epoch})')
            break

    # ── 최적 모델 로드 후 최종 평가 ──
    model.load_state_dict(torch.load(weight_path, map_location=device))
    (_, z_acc, a_acc,
     z_preds, z_labels, a_preds, a_labels,
     gate_z, gate_a) = evaluate(model, test_loader, device)

    print(f'\n  ┌─ 최종 결과 (best epoch={best_epoch}) ────────────┐')
    print(f'  │  Zone   정확도: {z_acc*100:6.2f}%                      │')
    print(f'  │  Action 정확도: {a_acc*100:6.2f}%                      │')
    print(f'  └────────────────────────────────────────────────┘')

    print('\n  [Zone] 분류 리포트')
    print(classification_report(z_labels, z_preds, target_names=ZONE_NAMES))
    print('\n  [Action] 분류 리포트')
    print(classification_report(a_labels, a_preds, target_names=ACTION_NAMES))

    save_confusion_matrix(z_labels, z_preds, ZONE_NAMES,   'mmoe_rocket_zone',   z_acc)
    save_confusion_matrix(a_labels, a_preds, ACTION_NAMES, 'mmoe_rocket_action', a_acc)
    plot_training_curves(history)
    save_gate_analysis(gate_z, gate_a)

    return z_acc, a_acc


# ══════════════════════════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════════════════════════

def main():
    set_seed(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    print('Loading preprocessed data...')
    all_samples = load_processed(PROCESSED_DIR)
    print(f'총 샘플: {len(all_samples)}')

    train_samples, test_samples = split_by_subjects(all_samples, TEST_SUBJECTS)
    print(f'Train: {len(train_samples)}  |  Test: {len(test_samples)}  '
          f'(test: {TEST_SUBJECTS})')

    if EXPERIMENT == 'standalone':
        run_standalone_rocket(train_samples, test_samples)

    elif EXPERIMENT == 'mmoe_rocket':
        run_mmoe_rocket(train_samples, test_samples)

    else:
        raise ValueError(f"Unknown EXPERIMENT: '{EXPERIMENT}'. "
                         f"Use 'standalone' or 'mmoe_rocket'.")


if __name__ == '__main__':
    main()
