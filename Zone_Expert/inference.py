"""
Zone Expert 추론 파이프라인

[흐름]
  test 데이터 (800, 664)
    → 정규화 (train-fit StandardScaler)
    → 슬라이딩 윈도우 (200, stride=10)
    → ZoneMLP (mean_only)       → zone 예측
    → 윈도우 평균 제거
    → ZoneExpert[zone]          → action 예측

[모델 선택]
  MODEL_TYPE = 'transformer_norx' | 'cnn' | 'rnn' | 'svm

[주의]
  ZoneExpertLSTM은 window_size=80으로 학습되었으나,
  LSTM 특성상 가변 길이 입력 가능 → 200패킷으로 추론 시 정확도 차이 있을 수 있음
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from dataset import (
    CSIRawDataset, normalize_datasets, get_device,
    TRAIN_SUBJECTS, TEST_SUBJECT, FEATURE_DIM, SEQ_LEN,
)

# ─── 모델 타입 선택 ────────────────────────────────────────────────────────────
# 'lstm' | 'cnn' | 'rnn' | 'svm' | 'transformer'
MODEL_TYPE = 'cnn_norx_noos'

# ─── 설정 ──────────────────────────────────────────────────────────────────────
WINDOW_SIZE  = 200
STRIDE       = 10
BATCH_SIZE   = 256
NUM_ZONES    = 4
ACTION_NAMES = ['handsup', 'sit', 'stand', 'walk']

MLP_WEIGHTS_PATH = os.path.join(
    BASE_DIR, '..', '13_Data_Processing', 'weights', 'zone_mlp_mean_only_best.pt'
)
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─── MODEL_TYPE별 가중치 경로 매핑 ───────────────────────────────────────────
EXPERT_WEIGHTS_DIRS = {
    'lstm'            : os.path.join(BASE_DIR, 'weights'),
    'cnn'             : os.path.join(BASE_DIR, 'weights_cnn'),
    'cnn_norx'        : os.path.join(BASE_DIR, 'weights_cnn_norx'),
    'cnn_noos'        : os.path.join(BASE_DIR, 'weights_cnn_noos'),
    'cnn_norx_noos'   : os.path.join(BASE_DIR, 'weights_cnn_norx_noos'),
    'rnn'             : os.path.join(BASE_DIR, 'weights_rnn'),
    'svm'             : os.path.join(BASE_DIR, 'weights_svm'),
    'transformer'          : os.path.join(BASE_DIR, 'weights_transformer'),
    'transformer_norx'     : os.path.join(BASE_DIR, 'weights_transformer_norx'),
    'transformer_noos'     : os.path.join(BASE_DIR, 'weights_transformer_noos'),
    'transformer_norx_noos': os.path.join(BASE_DIR, 'weights_transformer_norx_noos'),
    'lstm_norx'            : os.path.join(BASE_DIR, 'weights_lstm_norx'),
    'lstm_noos'            : os.path.join(BASE_DIR, 'weights_lstm_noos'),
    'lstm_norx_noos'       : os.path.join(BASE_DIR, 'weights_lstm_norx_noos'),
}


# ─── ZoneMLP 정의 (mean_only, use_stats=False) ────────────────────────────────
# train_zone_mlp_872.py의 ZoneMLP와 동일한 구조

class ZoneMLP(nn.Module):
    def __init__(self, input_dim: int = FEATURE_DIM, num_classes: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 664) → 시간축 평균 → (B, 664) → MLP
        return self.net(x.mean(dim=1))


# ─── 추론용 슬라이딩 윈도우 Dataset ──────────────────────────────────────────

class InferenceWindowDataset(Dataset):
    """
    정규화된 CSIRawDataset → 슬라이딩 윈도우.
    평균 제거는 적용하지 않음 (추론 루프에서 모델별로 처리).
    """
    def __init__(self, raw_ds: CSIRawDataset,
                 window_size: int, stride: int):
        self.windows       = []
        self.action_labels = []
        self.zone_labels   = []

        for idx in range(len(raw_ds)):
            seq   = raw_ds.data[idx]          # (800, 664)
            a_lbl = int(raw_ds.action_labels[idx])
            z_lbl = int(raw_ds.zone_labels[idx])

            start = 0
            while start + window_size <= seq.shape[0]:
                self.windows.append(seq[start:start + window_size])
                self.action_labels.append(a_lbl)
                self.zone_labels.append(z_lbl)
                start += stride

        self.windows = np.array(self.windows, dtype=np.float32)
        print(f"[InferenceWindowDataset] {len(raw_ds)} samples "
              f"→ {len(self.windows)} windows (size={window_size}, stride={stride})")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        x        = torch.tensor(self.windows[idx],       dtype=torch.float32)
        y_action = torch.tensor(self.action_labels[idx], dtype=torch.long)
        y_zone   = torch.tensor(self.zone_labels[idx],   dtype=torch.long)
        return x, y_action, y_zone


# ─── 혼동행렬 저장 ────────────────────────────────────────────────────────────

def save_cm(labels, preds, class_names, title, filename):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[CM saved] {path}")


# ─── 메인 추론 루프 ───────────────────────────────────────────────────────────

def run():
    t_total_start = time.perf_counter()
    device = get_device()
    print(f"Device: {device}")
    print(f"Window: {WINDOW_SIZE}  Stride: {STRIDE}")

    # ── 데이터 로드 & 정규화 ─────────────────────────────────────────────────
    print("\n[1] Loading & normalizing data...")
    t0 = time.perf_counter()
    train_raw = CSIRawDataset(TRAIN_SUBJECTS)
    test_raw  = CSIRawDataset([TEST_SUBJECT])
    normalize_datasets(train_raw, test_raw)
    t_load = time.perf_counter() - t0
    print(f"    └─ 소요 시간: {t_load:.2f}s")

    # ── 슬라이딩 윈도우 (평균 제거 없음) ────────────────────────────────────
    print(f"\n[2] Creating windows (size={WINDOW_SIZE}, stride={STRIDE})...")
    t0 = time.perf_counter()
    test_win    = InferenceWindowDataset(test_raw, WINDOW_SIZE, STRIDE)
    test_loader = DataLoader(test_win, batch_size=BATCH_SIZE, shuffle=False)
    t_window = time.perf_counter() - t0
    print(f"    └─ 소요 시간: {t_window:.2f}s")

    # ── 모델 로드 ────────────────────────────────────────────────────────────
    print(f"\n[3] Loading models (MODEL_TYPE = '{MODEL_TYPE}')...")
    t0 = time.perf_counter()

    zone_mlp = ZoneMLP().to(device)
    zone_mlp.load_state_dict(
        torch.load(MLP_WEIGHTS_PATH, map_location=device)
    )
    zone_mlp.eval()
    print(f"  ZoneMLP loaded ← {MLP_WEIGHTS_PATH}")

    weights_dir = EXPERT_WEIGHTS_DIRS[MODEL_TYPE]
    experts = {}

    if MODEL_TYPE in ('lstm', 'lstm_norx', 'lstm_noos', 'lstm_norx_noos'):
        from model import ZoneExpertLSTM
        use_rx = ('norx' not in MODEL_TYPE)
        for z in range(NUM_ZONES):
            m = ZoneExpertLSTM(zone_id=z, use_rx_weight=use_rx).to(device)
            ckpt = os.path.join(weights_dir, f'zone_expert_action_{z}_best.pt')
            m.load_state_dict(torch.load(ckpt, map_location=device))
            m.eval()
            experts[z] = m
            print(f"  ZoneExpertLSTM[{z}] (use_rx_weight={use_rx}) loaded ← {ckpt}")

    elif MODEL_TYPE in ('cnn', 'cnn_norx', 'cnn_noos', 'cnn_norx_noos'):
        from model_cnn import ZoneExpertCNN
        use_rx = ('norx' not in MODEL_TYPE)
        for z in range(NUM_ZONES):
            m = ZoneExpertCNN(zone_id=z, use_rx_weight=use_rx).to(device)
            ckpt = os.path.join(weights_dir, f'zone_expert_action_{z}_best.pt')
            m.load_state_dict(torch.load(ckpt, map_location=device))
            m.eval()
            experts[z] = m
            print(f"  ZoneExpertCNN[{z}] (use_rx_weight={use_rx}) loaded ← {ckpt}")

    elif MODEL_TYPE == 'rnn':
        from model_rnn import ZoneExpertRNN
        for z in range(NUM_ZONES):
            m = ZoneExpertRNN(zone_id=z).to(device)
            ckpt = os.path.join(weights_dir, f'zone_expert_action_{z}_best.pt')
            m.load_state_dict(torch.load(ckpt, map_location=device))
            m.eval()
            experts[z] = m
            print(f"  ZoneExpertRNN[{z}] loaded ← {ckpt}")

    elif MODEL_TYPE == 'svm':
        import joblib
        from train_svm import extract_features, apply_zone_rx_weight
        for z in range(NUM_ZONES):
            ckpt = os.path.join(weights_dir, f'zone_expert_action_{z}.pkl')
            experts[z] = joblib.load(ckpt)
            print(f"  SVM Expert[{z}] loaded ← {ckpt}")

    elif MODEL_TYPE.startswith('transformer'):
        from model_transformer import ZoneExpertTransformer
        use_rx = ('norx' not in MODEL_TYPE)
        for z in range(NUM_ZONES):
            m = ZoneExpertTransformer(zone_id=z, use_rx_weight=use_rx).to(device)
            ckpt = os.path.join(weights_dir, f'zone_expert_action_{z}_best.pt')
            m.load_state_dict(torch.load(ckpt, map_location=device))
            m.eval()
            experts[z] = m
            print(f"  ZoneExpertTransformer[{z}] (use_rx_weight={use_rx}) loaded ← {ckpt}")

    else:
        raise ValueError(
            f"Unknown MODEL_TYPE: '{MODEL_TYPE}'. "
            f"Choose from: 'lstm', 'cnn', 'cnn_norx', 'cnn_noos', 'cnn_norx_noos', "
            f"'rnn', 'svm', 'transformer'."
        )
    t_model = time.perf_counter() - t0
    print(f"    └─ 소요 시간: {t_model:.2f}s")

    # ── 추론 ─────────────────────────────────────────────────────────────────
    print("\n[4] Running inference...")
    t0 = time.perf_counter()
    all_zone_preds   = []
    all_zone_labels  = []
    all_action_preds  = []
    all_action_labels = []

    # zone별 전문가 결과 수집 (예측 zone 기준으로 라우팅된 샘플만)
    zone_expert_preds  = {z: [] for z in range(NUM_ZONES)}
    zone_expert_labels = {z: [] for z in range(NUM_ZONES)}

    with torch.no_grad():
        for x, y_action, y_zone in tqdm(test_loader, desc='Inference'):
            x = x.to(device)          # (B, 200, 664)

            # Step 1: ZoneMLP — 원본 윈도우 그대로 (평균 제거 없음)
            zone_logits = zone_mlp(x)
            zone_preds  = zone_logits.argmax(1)   # (B,)

            # Step 2: 윈도우별 평균 제거 → ZoneExpertLSTM
            x_mean_removed = x - x.mean(dim=1, keepdim=True)  # (B, 200, 664)

            action_preds = torch.zeros(x.size(0), dtype=torch.long, device=device)
            for z in range(NUM_ZONES):
                mask = (zone_preds == z)
                if mask.sum() == 0:
                    continue

                if MODEL_TYPE == 'svm':
                    # SVM: numpy 기반 추론
                    x_np   = x_mean_removed[mask].cpu().numpy()          # (N, 200, 664)
                    x_feat = extract_features(x_np)                       # (N, 1988)
                    x_feat = apply_zone_rx_weight(x_feat, z)
                    x_feat = experts[z]['feature_scaler'].transform(x_feat)
                    preds_np = experts[z]['svm'].predict(x_feat)
                    preds_z  = torch.tensor(preds_np, dtype=torch.long, device=device)
                else:
                    # PyTorch 모델 (LSTM / CNN / RNN)
                    logits  = experts[z](x_mean_removed[mask])
                    preds_z = logits.argmax(1)

                action_preds[mask] = preds_z

                # zone별 결과 저장
                zone_expert_preds[z].extend(preds_z.cpu().tolist())
                zone_expert_labels[z].extend(y_action[mask.cpu()].tolist())

            all_zone_preds.extend(zone_preds.cpu().tolist())
            all_zone_labels.extend(y_zone.tolist())
            all_action_preds.extend(action_preds.cpu().tolist())
            all_action_labels.extend(y_action.tolist())

    t_inference = time.perf_counter() - t0

    # ── 추론 벤치마크 (warmup 포함, ZoneMLP / ZoneExpert 분리 측정) ────────────
    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize()

    WARMUP = 3
    all_batches = list(test_loader)
    n_windows_bench = sum(x.size(0) for x, _, _ in all_batches)

    # Warmup
    with torch.no_grad():
        for x, _, _ in all_batches[:WARMUP]:
            x = x.to(device)
            zp = zone_mlp(x).argmax(1)
            xm = x - x.mean(dim=1, keepdim=True)
            for z in range(NUM_ZONES):
                mask = (zp == z)
                if mask.sum() > 0 and MODEL_TYPE != 'svm':
                    experts[z](xm[mask])
    sync()

    # ZoneMLP 단독 측정
    t_mlp_list = []
    with torch.no_grad():
        for x, _, _ in all_batches:
            x = x.to(device); sync()
            t0b = time.perf_counter()
            zone_mlp(x)
            sync()
            t_mlp_list.append((time.perf_counter() - t0b) * 1000)

    # ZoneExpert 단독 측정 (zone_pred는 미리 계산)
    t_exp_list = []
    with torch.no_grad():
        for x, _, _ in all_batches:
            x  = x.to(device)
            zp = zone_mlp(x).argmax(1)
            xm = x - x.mean(dim=1, keepdim=True)
            sync()
            t0b = time.perf_counter()
            for z in range(NUM_ZONES):
                mask = (zp == z)
                if mask.sum() > 0 and MODEL_TYPE != 'svm':
                    experts[z](xm[mask])
            sync()
            t_exp_list.append((time.perf_counter() - t0b) * 1000)

    # 전체 파이프라인 측정 (ZoneMLP + routing + ZoneExpert)
    t_pipe_list = []
    t_pipe_start = time.perf_counter()
    with torch.no_grad():
        for x, _, _ in all_batches:
            x = x.to(device); sync()
            t0b = time.perf_counter()
            zp = zone_mlp(x).argmax(1)
            xm = x - x.mean(dim=1, keepdim=True)
            for z in range(NUM_ZONES):
                mask = (zp == z)
                if mask.sum() > 0 and MODEL_TYPE != 'svm':
                    experts[z](xm[mask])
            sync()
            t_pipe_list.append((time.perf_counter() - t0b) * 1000)
    t_pipe_total = (time.perf_counter() - t_pipe_start) * 1000

    mlp_total  = sum(t_mlp_list)
    exp_total  = sum(t_exp_list)
    mlp_per_w  = mlp_total  / n_windows_bench
    exp_per_w  = exp_total  / n_windows_bench
    pipe_per_w = t_pipe_total / n_windows_bench

    print(f'\n{"="*60}')
    print(f'Inference Benchmark  [ZoneMoE / {MODEL_TYPE}]')
    print(f'{"="*60}')
    print(f'  Device             : {device}')
    print(f'  Total windows      : {n_windows_bench}')
    print(f'  Batch size         : {BATCH_SIZE}')
    print(f'  Warmup batches     : {WARMUP}')
    print(f'  ── Pass 1 ZoneMLP ──────────────────────')
    print(f'  Total time         : {mlp_total:.1f} ms')
    print(f'  Per window         : {mlp_per_w:.3f} ms')
    print(f'  ── Pass 2 ZoneExpert ───────────────────')
    print(f'  Total time         : {exp_total:.1f} ms')
    print(f'  Per window         : {exp_per_w:.3f} ms')
    print(f'  ── Pipeline (Pass1 + Pass2) ────────────')
    print(f'  Total time         : {t_pipe_total:.1f} ms  ({t_pipe_total/1000:.3f} s)')
    print(f'  Per window         : {pipe_per_w:.3f} ms')
    print(f'  Per batch (avg)    : {np.mean(t_pipe_list):.2f} ± {np.std(t_pipe_list):.2f} ms')
    print(f'  Throughput         : {n_windows_bench / (t_pipe_total/1000):.1f} windows/s')

    # ── 결과 출력 ─────────────────────────────────────────────────────────────
    zone_acc   = sum(p == l for p, l in zip(all_zone_preds,   all_zone_labels))   / len(all_zone_labels)
    action_acc = sum(p == l for p, l in zip(all_action_preds, all_action_labels)) / len(all_action_labels)

    t_total = time.perf_counter() - t_total_start
    n_windows = len(all_action_labels)

    print(f"\n{'='*45}")
    print(f"  Zone   Accuracy : {zone_acc:.4f}")
    print(f"  Action Accuracy : {action_acc:.4f}")
    print(f"{'='*45}")
    print(f"\n{'─'*45}")
    print(f"  [시간 요약]")
    print(f"{'─'*45}")
    print(f"  [1] 데이터 로드 & 정규화 : {t_load:7.2f}s")
    print(f"  [2] 슬라이딩 윈도우 생성 : {t_window:7.2f}s")
    print(f"  [3] 모델 로드            : {t_model:7.2f}s")
    print(f"  [4] 추론 루프            : {t_inference:7.2f}s  "
          f"({n_windows} windows, {n_windows/t_inference:.1f} win/s)")
    print(f"{'─'*45}")
    print(f"  전체 소요                : {t_total:7.2f}s")
    print(f"{'─'*45}")

    print(f"\n[Zone Classification Report]")
    print(classification_report(all_zone_labels, all_zone_preds,
                                target_names=[f'Zone {i}' for i in range(4)],
                                zero_division=0))

    print(f"[Action Classification Report]")
    print(classification_report(all_action_labels, all_action_preds,
                                target_names=ACTION_NAMES, zero_division=0))

    # ── 혼동행렬 저장 ─────────────────────────────────────────────────────────
    mt = MODEL_TYPE  # 파일명 접두사용
    save_cm(all_zone_labels, all_zone_preds,
            [f'Zone {i}' for i in range(4)],
            f'Zone CM  (Acc: {zone_acc:.4f})',
            f'{mt}_pipeline_zone_cm.png')

    save_cm(all_action_labels, all_action_preds,
            ACTION_NAMES,
            f'Action CM — Zone Expert Pipeline [{mt.upper()}]\n(Acc: {action_acc:.4f})',
            f'{mt}_pipeline_action_cm.png')

    # ── Zone별 전문가 혼동행렬 ─────────────────────────────────────────────────
    print("\n[Zone별 전문가 Action Classification Report]")
    for z in range(NUM_ZONES):
        preds  = zone_expert_preds[z]
        labels = zone_expert_labels[z]
        if len(preds) == 0:
            print(f"  Zone {z}: 라우팅된 샘플 없음")
            continue

        acc = sum(p == l for p, l in zip(preds, labels)) / len(labels)
        print(f"\n  Zone {z} Expert — {len(labels)} windows, Acc: {acc:.4f}")
        print(classification_report(labels, preds,
                                    target_names=ACTION_NAMES,
                                    zero_division=0))

        save_cm(labels, preds,
                ACTION_NAMES,
                f'Zone {z} Expert [{mt.upper()}] — Action CM\n(Acc: {acc:.4f}, {len(labels)} windows)',
                f'{mt}_pipeline_zone{z}_expert_cm.png')


if __name__ == '__main__':
    run()
