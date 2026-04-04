"""
MMOE — Wi-Fi CSI Multi-task Classification (Zone + Action)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
v1 — Multi-gate Mixture of Experts
  - Per-RX 1D CNN 인코더 (가중치 공유)
  - RX Fusion: Concat → Linear(512→256)
  - MMOE: 6 Experts + 2 Gates (Zone/Action)
  - Task Towers: MLP(128→64→4)
  - Uncertainty Weighting (Kendall et al., 2018)
  - Mixup 증강 (alpha=0.4)
  - StandardScaler 정규화 (train fit)
  - 입력 서브샘플링: 800 → 200 (stride 4)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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

NUM_FEATURES   = 166
NUM_RX         = 4
SUBSAMPLE      = 4          # 800 → 200
L_INPUT        = 200        # 서브샘플 후 길이

TEST_SUBJECTS  = ['kjh', 'kms']
ACTION_NAMES   = ['handsup', 'sit', 'stand', 'walk']
ZONE_NAMES     = ['Zone0', 'Zone1', 'Zone2', 'Zone3']

PROCESSED_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'processed')
RESULTS_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'mmoe')
WEIGHTS_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights')

# ── 하이퍼파라미터 ────────────────────────────────────────────────────────────

# 모델
CNN_CHANNELS   = [64, 128, 128]     # Per-RX CNN 채널
FUSION_DIM     = 256                # RX fusion 출력
NUM_EXPERTS    = 6
EXPERT_DIM     = 128
TOWER_HIDDEN   = 64
NUM_CLASSES    = 4

# 학습
BATCH_SIZE     = 32
LR             = 1e-3
WEIGHT_DECAY   = 1e-3
MAX_EPOCHS     = 100
PATIENCE       = 20
GRAD_CLIP      = 1.0
MIXUP_ALPHA    = 0.4

SEED           = 42


# ── 데이터 로딩 ──────────────────────────────────────────────────────────────

def load_processed(processed_dir):
    """processed/ 폴더의 NPZ 파일 전체 로드."""
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
    """Wi-Fi CSI 데이터셋. 서브샘플링 + 정규화 포함."""

    def __init__(self, grids, zones, actions, scaler=None, fit_scaler=False):
        """
        Parameters
        ----------
        grids   : (N, 4, 800, 166) float32
        zones   : (N,) int
        actions : (N,) int
        scaler  : StandardScaler or None
        fit_scaler : True면 이 데이터로 scaler fit
        """
        # 서브샘플링: 800 → 200
        grids = grids[:, :, ::SUBSAMPLE, :]   # (N, 4, 200, 166)

        N = grids.shape[0]

        # StandardScaler: (N*4*200, 166) 평탄화 후 fit/transform
        flat = grids.reshape(-1, NUM_FEATURES)  # (N*4*200, 166)
        if fit_scaler:
            assert scaler is not None
            scaler.fit(flat)
        if scaler is not None:
            flat = scaler.transform(flat).astype(np.float32)
        grids = flat.reshape(N, NUM_RX, L_INPUT, NUM_FEATURES)

        self.grids   = torch.from_numpy(grids)                    # (N, 4, 200, 166)
        self.zones   = torch.from_numpy(np.array(zones, dtype=np.int64))
        self.actions = torch.from_numpy(np.array(actions, dtype=np.int64))

    def __len__(self):
        return len(self.zones)

    def __getitem__(self, idx):
        return self.grids[idx], self.zones[idx], self.actions[idx]


def build_dataloaders(train_samples, test_samples):
    """
    샘플 리스트 → DataLoader 쌍 + scaler.

    Returns
    -------
    train_loader, test_loader, scaler
    """
    def collect(samples):
        grids   = np.stack([s['grids'] for s in samples])   # (N, 4, 800, 166)
        zones   = np.array([s['zone'] for s in samples])
        actions = np.array([s['action'] for s in samples])
        return grids, zones, actions

    tr_g, tr_z, tr_a = collect(train_samples)
    te_g, te_z, te_a = collect(test_samples)

    scaler = StandardScaler()
    train_ds = CSIDataset(tr_g, tr_z, tr_a, scaler=scaler, fit_scaler=True)
    test_ds  = CSIDataset(te_g, te_z, te_a, scaler=scaler, fit_scaler=False)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=True,
    )
    return train_loader, test_loader, scaler


# ── 모델 ─────────────────────────────────────────────────────────────────────

class PerRXEncoder(nn.Module):
    """1D CNN encoder shared across 4 RX antennas."""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            # (B*4, 166, 200) → (B*4, 64, 100)
            nn.Conv1d(NUM_FEATURES, CNN_CHANNELS[0], kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(CNN_CHANNELS[0]),
            nn.GELU(),
            # (B*4, 64, 100) → (B*4, 128, 50)
            nn.Conv1d(CNN_CHANNELS[0], CNN_CHANNELS[1], kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(CNN_CHANNELS[1]),
            nn.GELU(),
            # (B*4, 128, 50) → (B*4, 128, 25)
            nn.Conv1d(CNN_CHANNELS[1], CNN_CHANNELS[2], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(CNN_CHANNELS[2]),
            nn.GELU(),
            # (B*4, 128, 25) → (B*4, 128, 1)
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x):
        """
        x: (B, 4, 200, 166)
        Returns: (B, 4, 128)
        """
        B = x.size(0)
        # (B, 4, 200, 166) → (B*4, 200, 166) → permute → (B*4, 166, 200)
        x = x.reshape(B * NUM_RX, L_INPUT, NUM_FEATURES).permute(0, 2, 1)
        x = self.layers(x)           # (B*4, 128, 1)
        x = x.squeeze(-1)            # (B*4, 128)
        return x.reshape(B, NUM_RX, CNN_CHANNELS[-1])   # (B, 4, 128)


class MMOEModel(nn.Module):
    """
    Multi-gate Mixture of Experts for dual-task classification.

    Architecture:
        Per-RX CNN → RX Fusion → MMOE(Experts + Gates) → Task Towers
    """

    def __init__(self):
        super().__init__()

        # ── Per-RX CNN Encoder ──
        self.rx_encoder = PerRXEncoder()

        # ── RX Fusion: concat(4×128=512) → 256 ──
        self.rx_fusion = nn.Sequential(
            nn.Linear(NUM_RX * CNN_CHANNELS[-1], FUSION_DIM),
            nn.GELU(),
            nn.Dropout(0.3),
        )

        # ── MMOE Experts ──
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(FUSION_DIM, EXPERT_DIM),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(EXPERT_DIM, EXPERT_DIM),
                nn.GELU(),
            )
            for _ in range(NUM_EXPERTS)
        ])

        # ── Task-specific Gates ──
        self.gate_zone   = nn.Linear(FUSION_DIM, NUM_EXPERTS)
        self.gate_action = nn.Linear(FUSION_DIM, NUM_EXPERTS)

        # ── Task Towers ──
        self.tower_zone = nn.Sequential(
            nn.Linear(EXPERT_DIM, TOWER_HIDDEN),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(TOWER_HIDDEN, NUM_CLASSES),
        )
        self.tower_action = nn.Sequential(
            nn.Linear(EXPERT_DIM, TOWER_HIDDEN),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(TOWER_HIDDEN, NUM_CLASSES),
        )

        # ── Uncertainty Weighting (Kendall et al., 2018) ──
        self.log_var_zone   = nn.Parameter(torch.zeros(1))
        self.log_var_action = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
        x: (B, 4, 200, 166)
        Returns: zone_logits (B, 4), action_logits (B, 4)
        """
        # Per-RX encoding
        rx_feats = self.rx_encoder(x)                     # (B, 4, 128)

        # RX fusion
        fused = self.rx_fusion(rx_feats.reshape(x.size(0), -1))  # (B, 256)

        # Expert outputs
        expert_outs = torch.stack(
            [expert(fused) for expert in self.experts], dim=1
        )   # (B, 6, 128)

        # Task-specific gating
        g_zone   = F.softmax(self.gate_zone(fused), dim=-1)    # (B, 6)
        g_action = F.softmax(self.gate_action(fused), dim=-1)  # (B, 6)

        # Weighted expert mixture
        zone_mix   = torch.einsum('be,bed->bd', g_zone,   expert_outs)  # (B, 128)
        action_mix = torch.einsum('be,bed->bd', g_action, expert_outs)  # (B, 128)

        # Task towers
        zone_logits   = self.tower_zone(zone_mix)      # (B, 4)
        action_logits = self.tower_action(action_mix)   # (B, 4)

        return zone_logits, action_logits

    def compute_loss(self, zone_logits, action_logits, y_zone, y_action):
        """Uncertainty-weighted multi-task loss."""
        ce_zone   = F.cross_entropy(zone_logits, y_zone)
        ce_action = F.cross_entropy(action_logits, y_action)

        loss_zone   = ce_zone   / (2 * torch.exp(self.log_var_zone))   + self.log_var_zone / 2
        loss_action = ce_action / (2 * torch.exp(self.log_var_action)) + self.log_var_action / 2

        return (loss_zone + loss_action).squeeze()

    def compute_mixup_loss(self, zone_logits, action_logits,
                           y_zone_a, y_action_a, y_zone_b, y_action_b, lam):
        """Mixup 적용 시 loss 계산."""
        ce_zone   = lam * F.cross_entropy(zone_logits, y_zone_a) + \
                    (1 - lam) * F.cross_entropy(zone_logits, y_zone_b)
        ce_action = lam * F.cross_entropy(action_logits, y_action_a) + \
                    (1 - lam) * F.cross_entropy(action_logits, y_action_b)

        loss_zone   = ce_zone   / (2 * torch.exp(self.log_var_zone))   + self.log_var_zone / 2
        loss_action = ce_action / (2 * torch.exp(self.log_var_action)) + self.log_var_action / 2

        return (loss_zone + loss_action).squeeze()


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
    """Mixup 증강. 입력과 라벨에 적용."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    B = x.size(0)
    idx = torch.randperm(B, device=x.device)

    x_mixed = lam * x + (1 - lam) * x[idx]
    return x_mixed, y_zone, y_action, y_zone[idx], y_action[idx], lam


def train_one_epoch(model, loader, optimizer, device):
    """1 에포크 학습. Returns (avg_loss,)."""
    model.train()
    total_loss = 0.0
    n_batches  = 0

    for x, y_zone, y_action in loader:
        x        = x.to(device)
        y_zone   = y_zone.to(device)
        y_action = y_action.to(device)

        # Mixup
        x_mix, yz_a, ya_a, yz_b, ya_b, lam = mixup_data(x, y_zone, y_action)

        # Forward
        zone_logits, action_logits = model(x_mix)
        loss = model.compute_mixup_loss(
            zone_logits, action_logits, yz_a, ya_a, yz_b, ya_b, lam
        )

        # Backward
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    """
    평가.

    Returns
    -------
    avg_loss, zone_acc, action_acc, all_preds, all_labels
    """
    model.eval()
    total_loss = 0.0
    n_batches  = 0

    zone_preds,   zone_labels   = [], []
    action_preds, action_labels = [], []

    # Gate 가중치 수집 (분석용)
    gate_zone_all   = []
    gate_action_all = []

    for x, y_zone, y_action in loader:
        x        = x.to(device)
        y_zone   = y_zone.to(device)
        y_action = y_action.to(device)

        zone_logits, action_logits = model(x)
        loss = model.compute_loss(zone_logits, action_logits, y_zone, y_action)

        total_loss += loss.item()
        n_batches  += 1

        zone_preds.append(zone_logits.argmax(dim=1).cpu().numpy())
        zone_labels.append(y_zone.cpu().numpy())
        action_preds.append(action_logits.argmax(dim=1).cpu().numpy())
        action_labels.append(y_action.cpu().numpy())

        # Gate 가중치 캡처
        fused = model.rx_fusion(
            model.rx_encoder(x).reshape(x.size(0), -1)
        )
        g_z = F.softmax(model.gate_zone(fused), dim=-1).cpu().numpy()
        g_a = F.softmax(model.gate_action(fused), dim=-1).cpu().numpy()
        gate_zone_all.append(g_z)
        gate_action_all.append(g_a)

    zone_preds   = np.concatenate(zone_preds)
    zone_labels  = np.concatenate(zone_labels)
    action_preds = np.concatenate(action_preds)
    action_labels = np.concatenate(action_labels)

    zone_acc   = accuracy_score(zone_labels, zone_preds)
    action_acc = accuracy_score(action_labels, action_preds)

    gate_zone_avg   = np.concatenate(gate_zone_all).mean(axis=0)    # (6,)
    gate_action_avg = np.concatenate(gate_action_all).mean(axis=0)  # (6,)

    return (
        total_loss / max(n_batches, 1),
        zone_acc, action_acc,
        zone_preds, zone_labels,
        action_preds, action_labels,
        gate_zone_avg, gate_action_avg,
    )


# ── 시각화 ───────────────────────────────────────────────────────────────────

def save_confusion_matrix(y_true, y_pred, label_names, task_name, acc):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(f'{task_name}  (acc={acc * 100:.1f}%)')
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f'{task_name.lower()}_cm.png')
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f'  Confusion matrix saved → {path}')


def plot_training_curves(history):
    """학습 곡선 (loss + accuracy) 저장."""
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    axes[0].plot(epochs, history['train_loss'], label='Train loss', linewidth=1.5)
    axes[0].plot(epochs, history['test_loss'],  label='Test loss',  linewidth=1.5)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Multi-task Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Zone accuracy
    axes[1].plot(epochs, [a * 100 for a in history['zone_acc']], 'g-',
                 label='Zone Acc', linewidth=1.5)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Zone Classification')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Action accuracy
    axes[2].plot(epochs, [a * 100 for a in history['action_acc']], 'r-',
                 label='Action Acc', linewidth=1.5)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Accuracy (%)')
    axes[2].set_title('Action Classification')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('MMOE Training Curves', fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'training_curves.png')
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'  Training curves saved → {path}')


def save_gate_analysis(gate_zone_avg, gate_action_avg):
    """Gate 가중치 분포 시각화 (Expert 활용도)."""
    experts = [f'E{i}' for i in range(NUM_EXPERTS)]
    x = np.arange(NUM_EXPERTS)
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars_z = ax.bar(x - width / 2, gate_zone_avg,   width, label='Zone',   color='#4C72B0')
    bars_a = ax.bar(x + width / 2, gate_action_avg, width, label='Action', color='#DD8452')

    ax.set_xlabel('Expert')
    ax.set_ylabel('Average Gate Weight')
    ax.set_title('MMOE Gate Analysis — Expert Utilization')
    ax.set_xticks(x)
    ax.set_xticklabels(experts)
    ax.legend()
    ax.set_ylim(0, max(gate_zone_avg.max(), gate_action_avg.max()) * 1.25)

    for bar in bars_z:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)
    for bar in bars_a:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'gate_analysis.png')
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f'  Gate analysis saved → {path}')


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main():
    set_seed(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    device = get_device()
    print(f'Device: {device}')
    print(f'Input: (B, {NUM_RX}, {L_INPUT}, {NUM_FEATURES})')
    print(f'MMOE: {NUM_EXPERTS} experts, Fusion dim={FUSION_DIM}')

    # ── 데이터 로드 ──
    print('\nLoading preprocessed data ...')
    all_samples = load_processed(PROCESSED_DIR)
    print(f'Total recordings: {len(all_samples)}')

    train_samples, test_samples = split_by_subjects(all_samples, TEST_SUBJECTS)
    print(f'Train: {len(train_samples)}  |  Test: {len(test_samples)}  '
          f'(test subjects: {TEST_SUBJECTS})')

    train_loader, test_loader, scaler = build_dataloaders(train_samples, test_samples)
    print(f'Train batches: {len(train_loader)}  |  Test batches: {len(test_loader)}')

    # ── 모델 생성 ──
    model = MMOEModel().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'\nModel parameters: {n_params:,}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

    # ── 학습 루프 ──
    print(f'\n{"="*60}')
    print(f'Training MMOE  (max {MAX_EPOCHS} epochs, patience={PATIENCE})')
    print(f'{"="*60}')

    history = {
        'train_loss': [], 'test_loss': [],
        'zone_acc': [], 'action_acc': [],
    }

    best_metric = 0.0
    best_epoch  = 0
    patience_cnt = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        scheduler.step()

        # Evaluate
        (test_loss, zone_acc, action_acc,
         z_pred, z_true, a_pred, a_true,
         g_zone, g_action) = evaluate(model, test_loader, device)

        # 기록
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['zone_acc'].append(zone_acc)
        history['action_acc'].append(action_acc)

        mean_acc = (zone_acc + action_acc) / 2
        lr_now   = optimizer.param_groups[0]['lr']

        print(f'  Epoch {epoch:3d}/{MAX_EPOCHS}  '
              f'loss={train_loss:.4f}/{test_loss:.4f}  '
              f'Zone={zone_acc*100:.1f}%  Action={action_acc*100:.1f}%  '
              f'Mean={mean_acc*100:.1f}%  lr={lr_now:.2e}  '
              f'σ²_z={torch.exp(model.log_var_zone).item():.3f}  '
              f'σ²_a={torch.exp(model.log_var_action).item():.3f}')

        # Best model checkpoint
        if mean_acc > best_metric:
            best_metric  = mean_acc
            best_epoch   = epoch
            patience_cnt = 0
            torch.save(model.state_dict(),
                       os.path.join(WEIGHTS_DIR, 'mmoe_best.pt'))
        else:
            patience_cnt += 1

        if patience_cnt >= PATIENCE:
            print(f'\n  Early stopping at epoch {epoch} '
                  f'(best={best_epoch}, mean_acc={best_metric*100:.1f}%)')
            break

    # ── 최종 평가 (best model) ──
    print(f'\n{"="*60}')
    print(f'Final Evaluation (best epoch={best_epoch})')
    print(f'{"="*60}')

    model.load_state_dict(
        torch.load(os.path.join(WEIGHTS_DIR, 'mmoe_best.pt'),
                   map_location=device, weights_only=True)
    )
    (test_loss, zone_acc, action_acc,
     z_pred, z_true, a_pred, a_true,
     g_zone, g_action) = evaluate(model, test_loader, device)

    # Classification reports
    print(f'\n  Zone Accuracy  : {zone_acc * 100:.2f}%')
    print(classification_report(z_true, z_pred,
                                target_names=ZONE_NAMES, zero_division=0))
    print(f'  Action Accuracy: {action_acc * 100:.2f}%')
    print(classification_report(a_true, a_pred,
                                target_names=ACTION_NAMES, zero_division=0))

    # ── 시각화 저장 ──
    save_confusion_matrix(z_true, z_pred, ZONE_NAMES, 'Zone', zone_acc)
    save_confusion_matrix(a_true, a_pred, ACTION_NAMES, 'Action', action_acc)
    plot_training_curves(history)
    save_gate_analysis(g_zone, g_action)

    # 요약 CSV
    summary_path = os.path.join(RESULTS_DIR, 'accuracy_summary.csv')
    with open(summary_path, 'w') as f:
        f.write('task,test_subjects,accuracy,best_epoch\n')
        f.write(f'zone,"{"+".join(TEST_SUBJECTS)}",{zone_acc*100:.4f},{best_epoch}\n')
        f.write(f'action,"{"+".join(TEST_SUBJECTS)}",{action_acc*100:.4f},{best_epoch}\n')
    print(f'\n  Summary saved → {summary_path}')

    print(f'\n{"="*60}')
    print('RESULTS')
    print(f'{"="*60}')
    print(f'  Zone   Accuracy : {zone_acc * 100:.2f}%')
    print(f'  Action Accuracy : {action_acc * 100:.2f}%')
    print(f'  Mean   Accuracy : {(zone_acc + action_acc) / 2 * 100:.2f}%')
    print(f'  Best Epoch      : {best_epoch}')


if __name__ == '__main__':
    main()
