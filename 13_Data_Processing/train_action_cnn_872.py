"""
1D-CNN Action 분류기 (handsup 제외, 3-class: sit/stand/walk)
슬라이딩 윈도우: window_size=90, stride=10
추론: 다수결 voting (majority voting)
Train: 11명 (kjh, kms 제외), Test: kjh + kms (평균 accuracy)
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from tqdm import tqdm

from dataset_872 import (
    CSIDataset872, normalize_data, get_device,
    FEATURE_DIM, ALL_SUBJECTS
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), 'weights')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)

# --- 설정 ---
TEST_SUBJECTS = ['kjh', 'kms']
TRAIN_SUBJECTS = [s for s in ALL_SUBJECTS if s not in TEST_SUBJECTS]

WINDOW_SIZE = 90
STRIDE = 10

# handsup 제외 → 3-class 리매핑
# 원본: handsup=0, sit=1, stand=2, walk=3
# 새로: sit=0, stand=1, walk=2
REMAP = {1: 0, 2: 1, 3: 2}
ACTION_NAMES = ['sit', 'stand', 'walk']
NUM_CLASSES = 3


def filter_handsup(dataset):
    """handsup(label=0) 제거 후 라벨 리매핑"""
    mask = dataset.action_labels != 0  # handsup=0 제외
    dataset.data = dataset.data[mask]
    dataset.zone_labels = dataset.zone_labels[mask]
    old_actions = dataset.action_labels[mask]
    dataset.action_labels = np.array([REMAP[a] for a in old_actions])
    dataset.samples = [s for s, m in zip(dataset.samples, mask) if m]
    print(f"  handsup 제거 후: {len(dataset.data)} samples "
          f"(sit={np.sum(dataset.action_labels==0)}, "
          f"stand={np.sum(dataset.action_labels==1)}, "
          f"walk={np.sum(dataset.action_labels==2)})")


class SlidingWindowDataset(Dataset):
    """CSIDataset872를 슬라이딩 윈도우로 분할 (action label 사용)"""
    def __init__(self, base_dataset, window_size=WINDOW_SIZE, stride=STRIDE):
        self.windows = []
        self.labels = []
        self.sample_ids = []

        for sample_idx in range(len(base_dataset)):
            x = base_dataset.data[sample_idx]  # (872, 456) numpy
            y = base_dataset.action_labels[sample_idx]
            seq_len = x.shape[0]

            start = 0
            while start + window_size <= seq_len:
                self.windows.append(x[start:start + window_size])
                self.labels.append(int(y))
                self.sample_ids.append(sample_idx)
                start += stride

        print(f"  원본 샘플 수: {len(base_dataset)} → 윈도우 수: {len(self.windows)} "
              f"(window={window_size}, stride={stride})")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        x = torch.tensor(self.windows[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


class ActionCNN(nn.Module):
    """1D-CNN Action 분류기"""
    def __init__(self, input_dim=FEATURE_DIM, num_classes=NUM_CLASSES):
        super().__init__()
        self.cnn = nn.Sequential(
            # (B, input_dim, 90)
            nn.Conv1d(input_dim, 256, kernel_size=7, padding=3),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Conv1d(256, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        # Global Average Pooling → (B, 128)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # x: (B, window_size, 456) → transpose → (B, 456, window_size)
        x = x.transpose(1, 2)
        x = self.cnn(x)               # (B, 128, window_size)
        x = x.mean(dim=2)             # Global Average Pooling → (B, 128)
        return self.classifier(x)     # (B, num_classes)


def majority_vote(base_dataset, window_dataset, model, device):
    """원본 샘플별 다수결 voting"""
    model.eval()
    sample_preds = {}
    sample_labels = {}

    loader = DataLoader(window_dataset, batch_size=256, shuffle=False)
    idx = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(1).cpu().numpy()
            batch_labels = labels.numpy()

            for i in range(len(preds)):
                sid = window_dataset.sample_ids[idx]
                if sid not in sample_preds:
                    sample_preds[sid] = []
                    sample_labels[sid] = batch_labels[i]
                sample_preds[sid].append(preds[i])
                idx += 1

    final_preds, final_labels = [], []
    for sid in sorted(sample_preds.keys()):
        voted = Counter(sample_preds[sid]).most_common(1)[0][0]
        final_preds.append(voted)
        final_labels.append(sample_labels[sid])

    correct = sum(p == l for p, l in zip(final_preds, final_labels))
    acc = correct / len(final_labels)
    return acc, final_preds, final_labels


def train_model():
    device = get_device()
    print(f"Device: {device}")

    # --- 데이터 로드 ---
    print("\nLoading train data...")
    train_raw = CSIDataset872(TRAIN_SUBJECTS, task='action')
    filter_handsup(train_raw)

    print("\nLoading test data...")
    test_datasets = {}
    for subj in TEST_SUBJECTS:
        test_raw = CSIDataset872([subj], task='action')
        filter_handsup(test_raw)
        test_datasets[subj] = test_raw

    # --- 정규화 (train 기준) ---
    # 각 테스트셋을 train 기준으로 정규화
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    train_flat = train_raw.data.reshape(-1, FEATURE_DIM)
    scaler.fit(train_flat)
    train_raw.data = scaler.transform(train_flat).reshape(-1, 872, FEATURE_DIM).astype(np.float32)

    for subj in TEST_SUBJECTS:
        td = test_datasets[subj]
        test_flat = td.data.reshape(-1, FEATURE_DIM)
        td.data = scaler.transform(test_flat).reshape(-1, 872, FEATURE_DIM).astype(np.float32)

    # --- 슬라이딩 윈도우 ---
    print("\nCreating sliding window datasets...")
    train_win = SlidingWindowDataset(train_raw)
    test_wins = {}
    for subj in TEST_SUBJECTS:
        print(f"  [{subj}]")
        test_wins[subj] = SlidingWindowDataset(test_datasets[subj])

    train_loader = DataLoader(train_win, batch_size=64, shuffle=True)

    # --- 모델 ---
    model = ActionCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    best_avg_acc = 0
    best_epoch = 0

    print(f"\n{'='*60}")
    print(f"Training 1D-CNN Action (3-class: sit/stand/walk)")
    print(f"Train: {len(TRAIN_SUBJECTS)}명 ({len(train_win)} windows)")
    for subj in TEST_SUBJECTS:
        print(f"Test [{subj}]: {len(test_datasets[subj])} samples ({len(test_wins[subj])} windows)")
    print(f"{'='*60}")

    for epoch in tqdm(range(50), desc="CNN Action"):
        # --- 학습 ---
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
        train_acc = train_correct / len(train_win)
        avg_loss = train_loss / len(train_win)

        # --- 평가: 각 테스트 피험자별 voting ---
        accs = {}
        for subj in TEST_SUBJECTS:
            acc, _, _ = majority_vote(test_datasets[subj], test_wins[subj], model, device)
            accs[subj] = acc
        avg_acc = np.mean(list(accs.values()))

        if avg_acc > best_avg_acc:
            best_avg_acc = avg_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(),
                       os.path.join(WEIGHTS_DIR, 'action_cnn_best.pt'))

        if (epoch + 1) % 5 == 0 or epoch == 0:
            acc_str = " | ".join(f"{s}: {accs[s]:.4f}" for s in TEST_SUBJECTS)
            print(f"Epoch [{epoch+1:>2}/50] Loss: {avg_loss:.4f} "
                  f"Train: {train_acc:.4f} | {acc_str} | Avg: {avg_acc:.4f}")

    print(f"\nBest Avg Test Acc (majority voting): {best_avg_acc:.4f} (epoch {best_epoch})")

    # --- 최종 평가 ---
    model.load_state_dict(torch.load(os.path.join(WEIGHTS_DIR, 'action_cnn_best.pt'),
                                     map_location=device))

    all_preds_combined = []
    all_labels_combined = []

    for subj in TEST_SUBJECTS:
        acc, preds, labels = majority_vote(test_datasets[subj], test_wins[subj], model, device)
        all_preds_combined.extend(preds)
        all_labels_combined.extend(labels)
        print(f"\n[{subj}] Voting Accuracy: {acc:.4f} ({sum(p==l for p,l in zip(preds,labels))}/{len(labels)})")

    final_avg = np.mean([
        sum(p == l for p, l in zip(*majority_vote(test_datasets[s], test_wins[s], model, device)[1:]))
        / len(test_datasets[s])
        for s in TEST_SUBJECTS
    ])
    print(f"\nAverage Accuracy: {final_avg:.4f}")

    # --- Confusion Matrix (2명 합산) ---
    cm = confusion_matrix(all_labels_combined, all_preds_combined)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=ACTION_NAMES,
                yticklabels=ACTION_NAMES, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'1D-CNN Action Confusion Matrix (3-class, no handsup)\n'
                 f'Test: {", ".join(TEST_SUBJECTS)} | Avg Acc: {best_avg_acc:.4f}')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'action_cnn_cm.png'), dpi=150)
    plt.close()
    print(f"\nConfusion matrix 저장: {os.path.join(RESULTS_DIR, 'action_cnn_cm.png')}")

    # --- Classification Report ---
    print("\nClassification Report (combined):")
    print(classification_report(all_labels_combined, all_preds_combined,
                                target_names=ACTION_NAMES))


if __name__ == '__main__':
    train_model()
