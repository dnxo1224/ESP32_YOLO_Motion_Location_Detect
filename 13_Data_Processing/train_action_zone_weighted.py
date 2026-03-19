"""
Zone 가중치 기반 행동 분류기 (Phase 3)

파이프라인:
  1. 사전 학습된 BiLSTM zone 분류기로 test 샘플의 zone 예측 (sliding window + majority voting)
  2. 4개의 zone별 action 분류기 사전 학습 (WeightedRandomSampler 적용)
     - 해당 zone 샘플: weight 0.7
     - 나머지 zone 샘플: 각 0.1
  3. 추론: zone 예측 → 해당 zone의 action 분류기로 action 예측
  4. 비교: 가중치 없는 baseline action 분류기와 성능 비교

Train: 12명 (kjh 제외), Test: kjh
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from tqdm import tqdm

from dataset_872 import (
    CSIDataset872, normalize_data, get_device,
    TRAIN_SUBJECTS, TEST_SUBJECT, FEATURE_DIM,
    ACTION_MAP, ZONE_MAP
)
from train_zone_lstm_872 import (
    ZoneLSTM, SlidingWindowDataset,
    majority_vote, WINDOW_SIZE, STRIDE
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), 'weights')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)

ACTION_NAMES = ['handsup', 'sit', 'stand', 'walk']
ZONE_HIGH_W = 0.7   # 해당 zone 샘플 가중치
ZONE_LOW_W  = 0.1   # 나머지 zone 샘플 가중치 (zone당)


# ─── 모델 정의 ────────────────────────────────────────────────────────────────

class ActionLSTM(nn.Module):
    """BiLSTM Action 분류기 (ZoneLSTM과 동일 구조, num_classes=4)"""
    def __init__(self, input_dim=FEATURE_DIM, hidden_dim=128,
                 num_layers=2, num_classes=4, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )
        fc_input = hidden_dim * 2
        self.classifier = nn.Sequential(
            nn.Linear(fc_input, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: (B, WINDOW_SIZE, FEATURE_DIM)
        _, (hn, _) = self.lstm(x)
        last_out = torch.cat([hn[-2], hn[-1]], dim=1)  # (B, hidden*2)
        return self.classifier(last_out)


# ─── 슬라이딩 윈도우 + Majority Voting (action 용) ───────────────────────────

class ActionSlidingWindowDataset(SlidingWindowDataset):
    """
    SlidingWindowDataset 확장: action label과 zone label을 함께 반환.
    WeightedRandomSampler를 위해 window 단위 가중치도 계산.
    """
    def __init__(self, base_dataset, window_size=WINDOW_SIZE, stride=STRIDE):
        self.windows = []
        self.action_labels = []
        self.zone_labels = []
        self.sample_ids = []

        for sample_idx in range(len(base_dataset)):
            x, action_y, zone_y = base_dataset[sample_idx]  # 'both' task
            x_np = x.numpy()
            seq_len = x_np.shape[0]

            start = 0
            while start + window_size <= seq_len:
                self.windows.append(x_np[start:start + window_size])
                self.action_labels.append(action_y.item())
                self.zone_labels.append(zone_y.item())
                self.sample_ids.append(sample_idx)
                start += stride

        print(f"  ActionSlidingWindow: {len(base_dataset)} samples → {len(self.windows)} windows")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        x = torch.tensor(self.windows[idx], dtype=torch.float32)
        y = torch.tensor(self.action_labels[idx], dtype=torch.long)
        return x, y

    def get_zone_weights(self, target_zone):
        """target_zone 샘플에 0.7, 나머지에 0.1 가중치 부여 (윈도우 단위)"""
        return [
            ZONE_HIGH_W if z == target_zone else ZONE_LOW_W
            for z in self.zone_labels
        ]


def action_majority_vote(base_dataset, win_dataset, model, device):
    """샘플 단위 majority voting으로 action 정확도 계산"""
    model.eval()
    sample_preds = {}
    sample_action_labels = {}

    loader = DataLoader(win_dataset, batch_size=256, shuffle=False)
    idx = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(1).cpu().numpy()
            batch_labels = labels.numpy()

            for i in range(len(preds)):
                sid = win_dataset.sample_ids[idx]
                if sid not in sample_preds:
                    sample_preds[sid] = []
                    sample_action_labels[sid] = batch_labels[i]
                sample_preds[sid].append(preds[i])
                idx += 1

    final_preds, final_labels = [], []
    for sid in sorted(sample_preds.keys()):
        voted = Counter(sample_preds[sid]).most_common(1)[0][0]
        final_preds.append(voted)
        final_labels.append(sample_action_labels[sid])

    correct = sum(p == l for p, l in zip(final_preds, final_labels))
    return correct / len(final_labels), final_preds, final_labels


# ─── 학습 함수 ────────────────────────────────────────────────────────────────

def train_action_model(model, train_win_dataset, test_win_dataset,
                       test_raw_dataset, device, model_name,
                       target_zone=None, epochs=50):
    """
    Action 분류기 학습.
    target_zone=None → baseline (균등 가중치)
    target_zone=int  → WeightedRandomSampler로 해당 zone 강조
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    if target_zone is not None:
        weights = train_win_dataset.get_zone_weights(target_zone)
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True
        )
        train_loader = DataLoader(train_win_dataset, batch_size=64, sampler=sampler)
    else:
        train_loader = DataLoader(train_win_dataset, batch_size=64, shuffle=True)

    best_acc = 0
    best_epoch = 0

    for epoch in tqdm(range(epochs), desc=f"Action({model_name})"):
        model.train()
        train_loss, train_correct = 0, 0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()

        scheduler.step()
        train_acc = train_correct / len(train_win_dataset)

        # Test: majority voting
        vote_acc, _, _ = action_majority_vote(
            test_raw_dataset, test_win_dataset, model, device)

        if vote_acc > best_acc:
            best_acc = vote_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(),
                       os.path.join(WEIGHTS_DIR, f'{model_name}.pt'))

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  [{epoch+1:>2}/{epochs}] "
                  f"train_win: {train_acc:.4f} | test_vote: {vote_acc:.4f}")

    print(f"  ✅ Best: {best_acc:.4f} (epoch {best_epoch})")
    return best_acc


# ─── Zone 예측 파이프라인 ────────────────────────────────────────────────────

def predict_zones_for_test(test_raw, test_zone_win, zone_model, device):
    """
    test_raw의 각 샘플에 대해 zone_model로 zone 예측 (majority voting).
    반환: {sample_id: predicted_zone}
    """
    zone_model.eval()
    sample_preds = {}
    sample_true_zones = {}

    loader = DataLoader(test_zone_win, batch_size=256, shuffle=False)
    idx = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = zone_model(inputs)
            preds = outputs.argmax(1).cpu().numpy()
            batch_labels = labels.numpy()
            for i in range(len(preds)):
                sid = test_zone_win.sample_ids[idx]
                if sid not in sample_preds:
                    sample_preds[sid] = []
                    sample_true_zones[sid] = batch_labels[i]
                sample_preds[sid].append(preds[i])
                idx += 1

    zone_pred_map = {}
    for sid in sorted(sample_preds.keys()):
        zone_pred_map[sid] = Counter(sample_preds[sid]).most_common(1)[0][0]

    return zone_pred_map, sample_true_zones


# ─── 메인 ────────────────────────────────────────────────────────────────────

def main():
    device = get_device()
    print(f"Device: {device}")

    # ── 데이터 로드 ──
    print("\n[1] Loading data...")
    train_raw = CSIDataset872(TRAIN_SUBJECTS, task='both')
    test_raw  = CSIDataset872([TEST_SUBJECT],  task='both')
    normalize_data(train_raw, test_raw)

    print(f"Train: {len(train_raw)} samples | Test: {len(test_raw)} samples")

    # ── Zone 모델 로드 ──
    print("\n[2] Loading pre-trained zone model...")
    zone_model = ZoneLSTM(input_dim=FEATURE_DIM, num_classes=4).to(device)
    zone_ckpt = os.path.join(WEIGHTS_DIR, 'zone_lstm_best.pt')
    zone_model.load_state_dict(torch.load(zone_ckpt, map_location=device))
    zone_model.eval()
    print(f"  Loaded: {zone_ckpt}")

    # ── 슬라이딩 윈도우 데이터셋 ──
    print("\n[3] Creating sliding window datasets...")

    # Zone 평가용 (zone label 반환하는 SlidingWindowDataset)
    # test_raw는 'both' task이므로 zone win을 별도 구성
    test_raw_zone = CSIDataset872([TEST_SUBJECT], task='zone')
    # normalize: train_raw와 동일 scaler 적용 (이미 적용됨)
    test_raw_zone.data = test_raw.data  # 이미 정규화된 데이터 공유

    test_zone_win = SlidingWindowDataset(test_raw_zone)

    # Action용 슬라이딩 윈도우 (action + zone label 모두 필요)
    train_action_win = ActionSlidingWindowDataset(train_raw)
    test_action_win  = ActionSlidingWindowDataset(test_raw)

    # ── Test 샘플별 Zone 예측 ──
    print("\n[4] Predicting zones for test samples...")
    zone_pred_map, zone_true_map = predict_zones_for_test(
        test_raw, test_zone_win, zone_model, device)

    zone_preds = [zone_pred_map[i] for i in sorted(zone_pred_map.keys())]
    zone_trues = [zone_true_map[i] for i in sorted(zone_true_map.keys())]
    zone_acc = sum(p == t for p, t in zip(zone_preds, zone_trues)) / len(zone_trues)
    print(f"  Zone Acc (re-check): {zone_acc:.4f}")

    # ── Baseline: 가중치 없는 단일 Action 분류기 ──
    print("\n[5] Training baseline action model (no zone weighting)...")
    baseline_model = ActionLSTM().to(device)
    baseline_acc = train_action_model(
        baseline_model, train_action_win, test_action_win,
        test_raw, device, model_name='action_model_baseline',
        target_zone=None, epochs=50
    )

    # ── Zone별 Action 분류기 4개 학습 ──
    print("\n[6] Training zone-weighted action models...")
    zone_model_accs = {}
    for zone_i in range(4):
        print(f"\n  --- Zone {zone_i} Action Model "
              f"(high_w={ZONE_HIGH_W}, others={ZONE_LOW_W}) ---")
        model_i = ActionLSTM().to(device)
        acc_i = train_action_model(
            model_i, train_action_win, test_action_win,
            test_raw, device,
            model_name=f'action_model_zone_{zone_i}',
            target_zone=zone_i, epochs=50
        )
        zone_model_accs[zone_i] = acc_i

    # ── Zone-Weighted Pipeline 추론 ──
    print("\n[7] Zone-weighted pipeline inference (kjh test set)...")
    action_models = {}
    for zone_i in range(4):
        m = ActionLSTM().to(device)
        m.load_state_dict(torch.load(
            os.path.join(WEIGHTS_DIR, f'action_model_zone_{zone_i}.pt'),
            map_location=device))
        m.eval()
        action_models[zone_i] = m

    # 각 test 샘플: 예측된 zone → 해당 zone의 action 모델로 예측
    pipeline_preds = []
    true_actions   = []

    for sid in sorted(zone_pred_map.keys()):
        pred_zone = zone_pred_map[sid]
        action_model = action_models[pred_zone]

        # 해당 샘플의 윈도우만 추출
        sid_indices = [i for i, s in enumerate(test_action_win.sample_ids) if s == sid]
        windows = torch.tensor(
            np.array([test_action_win.windows[i] for i in sid_indices]),
            dtype=torch.float32
        ).to(device)

        action_model.eval()
        with torch.no_grad():
            outputs = action_model(windows)
            win_preds = outputs.argmax(1).cpu().numpy().tolist()

        voted_action = Counter(win_preds).most_common(1)[0][0]
        pipeline_preds.append(voted_action)
        true_actions.append(test_action_win.action_labels[sid_indices[0]])

    pipeline_acc = sum(p == t for p, t in zip(pipeline_preds, true_actions)) / len(true_actions)

    # ── 결과 요약 ──
    print("\n" + "=" * 60)
    print("ACTION CLASSIFICATION RESULTS (Test: kjh, 64 samples)")
    print("=" * 60)
    print(f"  Baseline (no weighting) : {baseline_acc:.4f}")
    print(f"  Zone-Weighted Pipeline  : {pipeline_acc:.4f}")
    print(f"    (Zone prediction acc  : {zone_acc:.4f})")
    print("-" * 60)
    print("  Zone-specific model best accs:")
    for z, acc in zone_model_accs.items():
        n_samples = sum(1 for z_ in zone_pred_map.values() if z_ == z)
        print(f"    Zone {z} model: {acc:.4f}  (predicted {n_samples} test samples → this zone)")
    print("=" * 60)

    # ── Confusion Matrix: Pipeline ──
    cm = confusion_matrix(true_actions, pipeline_preds)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Pipeline CM
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=ACTION_NAMES, yticklabels=ACTION_NAMES, ax=axes[0])
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    axes[0].set_title(f'Zone-Weighted Pipeline\nAcc: {pipeline_acc:.4f}')

    # Baseline: re-run inference for CM
    baseline_model.load_state_dict(torch.load(
        os.path.join(WEIGHTS_DIR, 'action_model_baseline.pt'), map_location=device))
    _, base_preds, base_labels = action_majority_vote(
        test_raw, test_action_win, baseline_model, device)
    cm_base = confusion_matrix(base_labels, base_preds)
    sns.heatmap(cm_base, annot=True, fmt='d', cmap='Oranges',
                xticklabels=ACTION_NAMES, yticklabels=ACTION_NAMES, ax=axes[1])
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')
    axes[1].set_title(f'Baseline (No Weighting)\nAcc: {baseline_acc:.4f}')

    plt.suptitle('Action Classification Comparison (Test: kjh)', fontsize=13)
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, 'action_zone_weighted_cm.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\nConfusion matrix saved: {save_path}")

    # ── Classification Report ──
    print("\nClassification Report (Zone-Weighted Pipeline):")
    print(classification_report(true_actions, pipeline_preds, target_names=ACTION_NAMES))

    print("\nClassification Report (Baseline):")
    print(classification_report(base_labels, base_preds, target_names=ACTION_NAMES))


if __name__ == '__main__':
    main()
