"""Transformer — Action Classification only (단일 태스크 비교군)"""

import os, math, time, warnings
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

warnings.filterwarnings('ignore')

# ── 상수 ──────────────────────────────────────────────────────────────────────

NUM_FEATURES  = 166
NUM_RX        = 4
SUBSAMPLE     = 8        # 800 → 100
L_INPUT       = 100
NUM_CLASSES   = 4

TEST_SUBJECTS = ['kms']
LABEL_NAMES   = ['handsup', 'sit', 'stand', 'walk']
TASK          = 'action'

PROCESSED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'processed')
RESULTS_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'tf_action')
WEIGHTS_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights')

# ── 하이퍼파라미터 ────────────────────────────────────────────────────────────

D_MODEL      = 128
N_HEADS      = 4
N_LAYERS     = 2
DIM_FF       = 256
DROPOUT      = 0.1
FUSION_DIM   = 256
TOWER_HIDDEN = 64
BATCH_SIZE   = 16
LR           = 1e-3
WEIGHT_DECAY = 1e-3
MAX_EPOCHS   = 100
PATIENCE     = 20
GRAD_CLIP    = 1.0
MIXUP_ALPHA  = 0.4
SEED         = 42


# ── 데이터 ───────────────────────────────────────────────────────────────────

def load_processed(processed_dir):
    samples = []
    for fname in tqdm(sorted(f for f in os.listdir(processed_dir) if f.endswith('.npz')),
                      desc='Loading'):
        d = np.load(os.path.join(processed_dir, fname), allow_pickle=True)
        samples.append({'grids': d['grids'], 'label': int(d[TASK]),
                        'subject': str(d['subject'])})
    if not samples:
        raise FileNotFoundError(f'No NPZ in {processed_dir}. Run preprocess.py first.')
    return samples


def split_by_subjects(samples, test_subjects):
    s = set(test_subjects)
    return [x for x in samples if x['subject'] not in s], \
           [x for x in samples if x['subject'] in s]


class CSIDataset(Dataset):
    def __init__(self, grids, labels, scaler=None, fit_scaler=False):
        grids = np.nan_to_num(np.clip(grids, 0, 10000))
        grids = grids[:, :, ::SUBSAMPLE, :]
        N, flat = grids.shape[0], grids.reshape(-1, NUM_FEATURES)
        if fit_scaler: scaler.fit(flat)
        if scaler is not None: flat = scaler.transform(flat).astype(np.float32)
        self.x = torch.from_numpy(flat.reshape(N, NUM_RX, L_INPUT, NUM_FEATURES))
        self.y = torch.from_numpy(np.array(labels, dtype=np.int64))

    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.x[i], self.y[i]


def build_dataloaders(train_samples, test_samples):
    def collect(s):
        return np.stack([x['grids'] for x in s]), np.array([x['label'] for x in s])
    tr_g, tr_l = collect(train_samples)
    te_g, te_l = collect(test_samples)
    scaler = StandardScaler()
    tr_ds = CSIDataset(tr_g, tr_l, scaler, fit_scaler=True)
    te_ds = CSIDataset(te_g, te_l, scaler)
    return (DataLoader(tr_ds, BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True, drop_last=True),
            DataLoader(te_ds, BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True))


# ── 모델 ─────────────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class TransformerAction(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Linear(NUM_FEATURES, D_MODEL)
        self.pos_enc    = PositionalEncoding(D_MODEL, max_len=L_INPUT+1, dropout=DROPOUT)
        self.cls_token  = nn.Parameter(torch.zeros(1, 1, D_MODEL))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(D_MODEL, N_HEADS, DIM_FF, DROPOUT, batch_first=True),
            num_layers=N_LAYERS)
        self.fusion = nn.Sequential(
            nn.Linear(NUM_RX * D_MODEL, FUSION_DIM), nn.GELU(), nn.Dropout(0.3))
        self.head = nn.Sequential(
            nn.Linear(FUSION_DIM, TOWER_HIDDEN), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(TOWER_HIDDEN, NUM_CLASSES))

    def forward(self, x):
        B = x.size(0)
        seq = self.input_proj(x.reshape(B * NUM_RX, L_INPUT, NUM_FEATURES))
        cls = self.cls_token.expand(B * NUM_RX, -1, -1)
        seq = self.pos_enc(torch.cat([cls, seq], dim=1))
        cls_out = self.transformer(seq)[:, 0].reshape(B, NUM_RX * D_MODEL)
        return self.head(self.fusion(cls_out))


# ── 학습 유틸리티 ────────────────────────────────────────────────────────────

def set_seed(s): np.random.seed(s); torch.manual_seed(s)

def get_device():
    if torch.cuda.is_available(): return torch.device('cuda')
    if hasattr(torch.backends,'mps') and torch.backends.mps.is_available(): return torch.device('mps')
    return torch.device('cpu')

def mixup(x, y, alpha=MIXUP_ALPHA):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam*x + (1-lam)*x[idx], y, y[idx], lam

def train_one_epoch(model, loader, opt, device):
    model.train(); total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        xm, ya, yb, lam = mixup(x, y)
        loss = lam*F.cross_entropy(model(xm), ya) + (1-lam)*F.cross_entropy(model(xm), yb)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step(); total += loss.item(); n += 1
    return total / max(n, 1)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); total, n = 0.0, 0
    preds, labels = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total += F.cross_entropy(logits, y).item(); n += 1
        preds.append(logits.argmax(1).cpu().numpy())
        labels.append(y.cpu().numpy())
    preds, labels = np.concatenate(preds), np.concatenate(labels)
    return total/max(n,1), accuracy_score(labels, preds), preds, labels


# ── 추론 벤치마크 ────────────────────────────────────────────────────────────

@torch.no_grad()
def benchmark_inference(model, loader, device, warmup_batches=3):
    model.eval()
    def sync():
        if device.type == 'cuda': torch.cuda.synchronize()

    all_batches = list(loader)
    n_samples   = sum(x.size(0) for x, _ in all_batches)

    for x, _ in all_batches[:warmup_batches]: model(x.to(device))
    sync()

    batch_times = []; t0_total = time.perf_counter()
    for x, _ in all_batches:
        x = x.to(device); sync()
        t0 = time.perf_counter(); model(x); sync()
        batch_times.append((time.perf_counter() - t0) * 1000)

    total_ms   = (time.perf_counter() - t0_total) * 1000
    per_sample = total_ms / n_samples

    print(f'\n{"="*60}')
    print(f'Inference Benchmark  [Transformer / Action]')
    print(f'{"="*60}')
    print(f'  Device          : {device}')
    print(f'  Total samples   : {n_samples}')
    print(f'  Batch size      : {loader.batch_size}')
    print(f'  Total time      : {total_ms:.1f} ms  ({total_ms/1000:.3f} s)')
    print(f'  Per sample      : {per_sample:.3f} ms')
    print(f'  Per batch (avg) : {np.mean(batch_times):.2f} ± {np.std(batch_times):.2f} ms')
    print(f'  Throughput      : {n_samples/(total_ms/1000):.1f} samples/s')
    return dict(total_ms=total_ms, per_sample=per_sample,
                per_batch=np.mean(batch_times), n_samples=n_samples)


# ── 시각화 ───────────────────────────────────────────────────────────────────

def save_cm(y_true, y_pred, acc):
    disp = ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred), display_labels=LABEL_NAMES)
    fig, ax = plt.subplots(figsize=(6,5)); disp.plot(ax=ax, colorbar=False)
    ax.set_title(f'Action  (acc={acc*100:.1f}%)'); plt.tight_layout()
    p = os.path.join(RESULTS_DIR, 'action_cm.png'); fig.savefig(p, dpi=120); plt.close(fig)
    print(f'  CM saved → {p}')

def plot_curves(history):
    ep = range(1, len(history['train_loss'])+1)
    fig, axes = plt.subplots(1, 2, figsize=(12,5))
    axes[0].plot(ep, history['train_loss'], label='Train'); axes[0].plot(ep, history['test_loss'], label='Test')
    axes[0].set(title='Loss', xlabel='Epoch'); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(ep, [a*100 for a in history['acc']], 'r-')
    axes[1].set(title='Action Accuracy (%)', xlabel='Epoch'); axes[1].grid(alpha=0.3)
    plt.suptitle('Transformer / Action', fontsize=13, y=1.02); plt.tight_layout()
    p = os.path.join(RESULTS_DIR, 'training_curves.png'); fig.savefig(p, dpi=120, bbox_inches='tight'); plt.close(fig)


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main():
    set_seed(SEED); os.makedirs(RESULTS_DIR, exist_ok=True); os.makedirs(WEIGHTS_DIR, exist_ok=True)
    device = get_device()
    print(f'Model  : Transformer / Action only  |  Device: {device}')

    all_samples = load_processed(PROCESSED_DIR)
    train_s, test_s = split_by_subjects(all_samples, TEST_SUBJECTS)
    print(f'Train: {len(train_s)}  |  Test: {len(test_s)}')
    train_loader, test_loader = build_dataloaders(train_s, test_s)

    model = TransformerAction().to(device)
    print(f'Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)

    history = {'train_loss': [], 'test_loss': [], 'acc': []}
    best, best_ep, patience_cnt = 0.0, 0, 0
    weight_path = os.path.join(WEIGHTS_DIR, 'tf_action_best.pt')

    for ep in range(1, MAX_EPOCHS+1):
        tr_loss = train_one_epoch(model, train_loader, opt, device); sch.step()
        te_loss, acc, _, _ = evaluate(model, test_loader, device)
        history['train_loss'].append(tr_loss); history['test_loss'].append(te_loss); history['acc'].append(acc)
        print(f'  Epoch {ep:3d}/{MAX_EPOCHS}  loss={tr_loss:.4f}/{te_loss:.4f}  Acc={acc*100:.1f}%  lr={opt.param_groups[0]["lr"]:.2e}')
        if acc > best:
            best, best_ep, patience_cnt = acc, ep, 0; torch.save(model.state_dict(), weight_path)
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f'\n  Early stopping (best={best_ep}, acc={best*100:.1f}%)'); break

    model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
    _, acc, preds, labels = evaluate(model, test_loader, device)
    print(f'\n  Action Accuracy: {acc*100:.2f}%')
    print(classification_report(labels, preds, target_names=LABEL_NAMES, zero_division=0))
    save_cm(labels, preds, acc); plot_curves(history)

    bench = benchmark_inference(model, test_loader, device)

    with open(os.path.join(RESULTS_DIR, 'summary.csv'), 'w') as f:
        f.write('model,task,accuracy,best_epoch,infer_per_sample_ms,throughput_sps\n')
        f.write(f'Transformer,action,{acc*100:.4f},{best_ep},'
                f'{bench["per_sample"]:.3f},{bench["n_samples"]/(bench["total_ms"]/1000):.1f}\n')

    print(f'\n{"="*60}\nRESULTS\n{"="*60}')
    print(f'  Action Accuracy : {acc*100:.2f}%')
    print(f'  Infer/sample    : {bench["per_sample"]:.3f} ms')
    print(f'  Throughput      : {bench["n_samples"]/(bench["total_ms"]/1000):.1f} samples/s')


if __name__ == '__main__':
    main()
