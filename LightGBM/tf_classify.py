"""
Transformer — Wi-Fi CSI Multi-task Classification (Zone + Action)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
비교군 모델 (vs MMoE)
  - Per-RX Linear Projection + Positional Encoding
  - Transformer Encoder (d_model=128, nhead=4, layers=2)
  - [CLS] 토큰 기반 시퀀스 요약
  - RX Fusion: Concat → Linear(4*128 → 256)
  - Task Towers: MLP(256→64→4)
  - Uncertainty Weighting (Kendall et al., 2018)
  - Mixup 증강 (alpha=0.4)
  - StandardScaler 정규화 (train fit)
  - 입력 서브샘플링: 800 → 100 (stride 8)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

실행:
    python tf_classify.py
"""

import os
import math
import time
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
SUBSAMPLE     = 8        # 800 → 100
L_INPUT       = 100

TEST_SUBJECTS = ['kms']
ACTION_NAMES  = ['handsup', 'sit', 'stand', 'walk']
ZONE_NAMES    = ['Zone0', 'Zone1', 'Zone2', 'Zone3']

PROCESSED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'processed')
RESULTS_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'transformer')
WEIGHTS_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights')

# ── 하이퍼파라미터 ────────────────────────────────────────────────────────────

D_MODEL       = 128
N_HEADS       = 4
N_LAYERS      = 2
DIM_FF        = 256
DROPOUT       = 0.1
FUSION_DIM    = 256
TOWER_HIDDEN  = 64
NUM_CLASSES   = 4

BATCH_SIZE    = 16
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
        raise FileNotFoundError(f'No NPZ files in {processed_dir}. Run preprocess.py first.')
    for fname in tqdm(npz_files, desc='Loading processed data'):
        data = np.load(os.path.join(processed_dir, fname), allow_pickle=True)
        samples.append({
            'grids':   data['grids'],
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
    def __init__(self, grids, zones, actions, scaler=None, fit_scaler=False):
        grids = np.nan_to_num(grids, nan=0.0, posinf=0.0, neginf=0.0)
        grids = np.clip(grids, 0.0, 10000.0)
        grids = grids[:, :, ::SUBSAMPLE, :]   # (N, 4, 100, 166)
        N    = grids.shape[0]
        flat = grids.reshape(-1, NUM_FEATURES)
        if fit_scaler:
            assert scaler is not None
            scaler.fit(flat)
        if scaler is not None:
            flat = scaler.transform(flat).astype(np.float32)
        grids = flat.reshape(N, NUM_RX, L_INPUT, NUM_FEATURES)

        self.grids   = torch.from_numpy(grids)
        self.zones   = torch.from_numpy(np.array(zones,   dtype=np.int64))
        self.actions = torch.from_numpy(np.array(actions, dtype=np.int64))

    def __len__(self):
        return len(self.zones)

    def __getitem__(self, idx):
        return self.grids[idx], self.zones[idx], self.actions[idx]


def build_dataloaders(train_samples, test_samples):
    def collect(samples):
        grids   = np.stack([s['grids']  for s in samples])
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


# ── 모델 ─────────────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


def _build_tower():
    return nn.Sequential(
        nn.Linear(FUSION_DIM, TOWER_HIDDEN), nn.GELU(), nn.Dropout(0.3),
        nn.Linear(TOWER_HIDDEN, NUM_CLASSES),
    )


class TransformerModel(nn.Module):
    """
    Per-RX Transformer Encoder ([CLS] token) → RX Concat Fusion → Dual Task Towers.
    입력: (B, 4, 100, 166)
    """

    def __init__(self):
        super().__init__()

        # Per-RX input projection: 166 → d_model
        self.input_proj = nn.Linear(NUM_FEATURES, D_MODEL)
        self.pos_enc    = PositionalEncoding(D_MODEL, max_len=L_INPUT + 1, dropout=DROPOUT)

        # [CLS] token (per-RX 공유)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, D_MODEL))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=N_HEADS,
            dim_feedforward=DIM_FF,
            dropout=DROPOUT,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=N_LAYERS)

        # RX fusion: 4 × D_MODEL → FUSION_DIM
        self.fusion = nn.Sequential(
            nn.Linear(NUM_RX * D_MODEL, FUSION_DIM),
            nn.GELU(),
            nn.Dropout(0.3),
        )

        self.tower_zone   = _build_tower()
        self.tower_action = _build_tower()

        self.log_var_zone   = nn.Parameter(torch.zeros(1))
        self.log_var_action = nn.Parameter(torch.zeros(1))

    def _encode(self, x):
        """
        x: (B, 4, 100, 166)
        Returns: (B, 4, D_MODEL)  — CLS token output per RX
        """
        B = x.size(0)
        # (B*4, 100, 166) → projection → (B*4, 100, D_MODEL)
        seq  = self.input_proj(x.reshape(B * NUM_RX, L_INPUT, NUM_FEATURES))

        # CLS 토큰 prepend: (B*4, 101, D_MODEL)
        cls  = self.cls_token.expand(B * NUM_RX, -1, -1)
        seq  = torch.cat([cls, seq], dim=1)
        seq  = self.pos_enc(seq)

        out  = self.transformer(seq)          # (B*4, 101, D_MODEL)
        cls_out = out[:, 0]                   # (B*4, D_MODEL) — CLS only
        return cls_out.reshape(B, NUM_RX, D_MODEL)

    def forward(self, x):
        B   = x.size(0)
        enc = self._encode(x)                       # (B, 4, D_MODEL)
        fused = self.fusion(enc.reshape(B, -1))     # (B, FUSION_DIM)
        return self.tower_zone(fused), self.tower_action(fused)

    def compute_loss(self, zl, al, yz, ya):
        return _uncertainty_loss(zl, al, yz, ya, self.log_var_zone, self.log_var_action)

    def compute_mixup_loss(self, zl, al, yza, yaa, yzb, yab, lam):
        return _uncertainty_mixup_loss(zl, al, yza, yaa, yzb, yab, lam,
                                       self.log_var_zone, self.log_var_action)


# ── Loss 헬퍼 ─────────────────────────────────────────────────────────────────

def _uncertainty_loss(zl, al, yz, ya, log_var_z, log_var_a):
    lz = F.cross_entropy(zl, yz) / (2 * torch.exp(log_var_z)) + log_var_z / 2
    la = F.cross_entropy(al, ya) / (2 * torch.exp(log_var_a)) + log_var_a / 2
    return (lz + la).squeeze()


def _uncertainty_mixup_loss(zl, al, yza, yaa, yzb, yab, lam, log_var_z, log_var_a):
    ce_z = lam * F.cross_entropy(zl, yza) + (1 - lam) * F.cross_entropy(zl, yzb)
    ce_a = lam * F.cross_entropy(al, yaa) + (1 - lam) * F.cross_entropy(al, yab)
    lz   = ce_z / (2 * torch.exp(log_var_z)) + log_var_z / 2
    la   = ce_a / (2 * torch.exp(log_var_a)) + log_var_a / 2
    return (lz + la).squeeze()


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
        total_loss += loss.item()
        n += 1

    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    n = 0
    z_preds, z_labels, a_preds, a_labels = [], [], [], []

    for x, y_zone, y_action in loader:
        x, y_zone, y_action = x.to(device), y_zone.to(device), y_action.to(device)
        zl, al = model(x)
        loss   = model.compute_loss(zl, al, y_zone, y_action)
        total_loss += loss.item()
        n += 1

        z_preds.append(zl.argmax(1).cpu().numpy())
        z_labels.append(y_zone.cpu().numpy())
        a_preds.append(al.argmax(1).cpu().numpy())
        a_labels.append(y_action.cpu().numpy())

    z_preds  = np.concatenate(z_preds)
    z_labels = np.concatenate(z_labels)
    a_preds  = np.concatenate(a_preds)
    a_labels = np.concatenate(a_labels)

    return (total_loss / max(n, 1),
            accuracy_score(z_labels, z_preds),
            accuracy_score(a_labels, a_preds),
            z_preds, z_labels, a_preds, a_labels)


# ── 추론 벤치마크 ────────────────────────────────────────────────────────────

@torch.no_grad()
def benchmark_inference(model, loader, device, warmup_batches=3):
    model.eval()

    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize()

    all_batches = list(loader)
    n_samples   = sum(x.size(0) for x, _, _ in all_batches)

    for x, _, _ in all_batches[:warmup_batches]:
        model(x.to(device))
    sync()

    batch_times   = []
    t_total_start = time.perf_counter()

    for x, _, _ in all_batches:
        x = x.to(device)
        sync()
        t0 = time.perf_counter()
        model(x)
        sync()
        batch_times.append((time.perf_counter() - t0) * 1000)

    total_ms   = (time.perf_counter() - t_total_start) * 1000
    per_sample = total_ms / n_samples
    per_batch  = np.mean(batch_times)
    std_batch  = np.std(batch_times)

    print(f'\n{"="*60}')
    print('Inference Benchmark  [Transformer]')
    print(f'{"="*60}')
    print(f'  Device          : {device}')
    print(f'  Total samples   : {n_samples}')
    print(f'  Batch size      : {loader.batch_size}')
    print(f'  Warmup batches  : {warmup_batches}')
    print(f'  Total time      : {total_ms:.1f} ms  ({total_ms/1000:.3f} s)')
    print(f'  Per sample      : {per_sample:.3f} ms')
    print(f'  Per batch (avg) : {per_batch:.2f} ± {std_batch:.2f} ms')
    print(f'  Throughput      : {n_samples / (total_ms / 1000):.1f} samples/s')

    return {
        'total_ms':   total_ms,
        'per_sample': per_sample,
        'per_batch':  per_batch,
        'std_batch':  std_batch,
        'n_samples':  n_samples,
    }


# ── 시각화 ───────────────────────────────────────────────────────────────────

def save_confusion_matrix(y_true, y_pred, label_names, task_name, acc):
    cm   = confusion_matrix(y_true, y_pred)
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
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(epochs, history['train_loss'], label='Train', linewidth=1.5)
    axes[0].plot(epochs, history['test_loss'],  label='Test',  linewidth=1.5)
    axes[0].set(xlabel='Epoch', ylabel='Loss', title='Multi-task Loss')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, [a * 100 for a in history['zone_acc']], 'g-', linewidth=1.5)
    axes[1].set(xlabel='Epoch', ylabel='Accuracy (%)', title='Zone Accuracy')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, [a * 100 for a in history['action_acc']], 'r-', linewidth=1.5)
    axes[2].set(xlabel='Epoch', ylabel='Accuracy (%)', title='Action Accuracy')
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('Transformer Training Curves', fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'training_curves.png')
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'  Training curves saved → {path}')


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main():
    set_seed(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    device = get_device()
    print(f'Device       : {device}')
    print(f'Model        : Transformer (d={D_MODEL}, heads={N_HEADS}, layers={N_LAYERS})')
    print(f'Input shape  : (B, {NUM_RX}, {L_INPUT}, {NUM_FEATURES})')

    print('\nLoading preprocessed data ...')
    all_samples = load_processed(PROCESSED_DIR)
    print(f'Total recordings: {len(all_samples)}')

    train_samples, test_samples = split_by_subjects(all_samples, TEST_SUBJECTS)
    print(f'Train: {len(train_samples)}  |  Test: {len(test_samples)}  '
          f'(test subjects: {TEST_SUBJECTS})')

    train_loader, test_loader, _ = build_dataloaders(train_samples, test_samples)
    print(f'Train batches: {len(train_loader)}  |  Test batches: {len(test_loader)}')

    model   = TransformerModel().to(device)
    n_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Parameters   : {n_param:,}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

    print(f'\n{"="*60}')
    print(f'Training Transformer  (max {MAX_EPOCHS} epochs, patience={PATIENCE})')
    print(f'{"="*60}')

    history      = {'train_loss': [], 'test_loss': [], 'zone_acc': [], 'action_acc': []}
    best_metric  = 0.0
    best_epoch   = 0
    patience_cnt = 0
    weight_path  = os.path.join(WEIGHTS_DIR, 'transformer_best.pt')

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        scheduler.step()

        (test_loss, zone_acc, action_acc,
         z_pred, z_true, a_pred, a_true) = evaluate(model, test_loader, device)

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
            torch.save(model.state_dict(), weight_path)
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f'\n  Early stopping at epoch {epoch} '
                      f'(best={best_epoch}, mean={best_metric*100:.1f}%)')
                break

    print(f'\n{"="*60}')
    print(f'Final Evaluation (best epoch={best_epoch})')
    print(f'{"="*60}')

    model.load_state_dict(
        torch.load(weight_path, map_location=device, weights_only=True)
    )
    (_, zone_acc, action_acc,
     z_pred, z_true, a_pred, a_true) = evaluate(model, test_loader, device)

    print(f'\n  Zone Accuracy  : {zone_acc * 100:.2f}%')
    print(classification_report(z_true, z_pred, target_names=ZONE_NAMES, zero_division=0))
    print(f'  Action Accuracy: {action_acc * 100:.2f}%')
    print(classification_report(a_true, a_pred, target_names=ACTION_NAMES, zero_division=0))

    save_confusion_matrix(z_true, z_pred, ZONE_NAMES,   'Zone',   zone_acc)
    save_confusion_matrix(a_true, a_pred, ACTION_NAMES, 'Action', action_acc)
    plot_training_curves(history)

    bench = benchmark_inference(model, test_loader, device)

    summary_path = os.path.join(RESULTS_DIR, 'accuracy_summary.csv')
    with open(summary_path, 'w') as f:
        f.write('model,task,test_subjects,accuracy,best_epoch,'
                'infer_total_ms,infer_per_sample_ms,infer_throughput_sps\n')
        throughput = bench['n_samples'] / (bench['total_ms'] / 1000)
        for task, acc in [('zone', zone_acc), ('action', action_acc)]:
            f.write(f'Transformer,{task},"{"+".join(TEST_SUBJECTS)}",'
                    f'{acc*100:.4f},{best_epoch},'
                    f'{bench["total_ms"]:.1f},{bench["per_sample"]:.3f},{throughput:.1f}\n')
    print(f'\n  Summary saved → {summary_path}')

    print(f'\n{"="*60}')
    print('RESULTS')
    print(f'{"="*60}')
    print(f'  Model          : Transformer')
    print(f'  Zone   Accuracy: {zone_acc * 100:.2f}%')
    print(f'  Action Accuracy: {action_acc * 100:.2f}%')
    print(f'  Mean   Accuracy: {(zone_acc + action_acc) / 2 * 100:.2f}%')
    print(f'  Best Epoch     : {best_epoch}')
    print(f'  Infer (sample) : {bench["per_sample"]:.3f} ms/sample')
    print(f'  Throughput     : {bench["n_samples"] / (bench["total_ms"] / 1000):.1f} samples/s')


if __name__ == '__main__':
    main()
