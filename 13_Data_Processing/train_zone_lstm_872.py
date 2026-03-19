"""
LSTM Zone 분류기 (872 x 114 x 4 데이터)
슬라이딩 윈도우: window_size=90, stride=10
추론: 다수결 voting (majority voting)
Train: 12명 (kjh 제외), Test: kjh
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
    TRAIN_SUBJECTS, TEST_SUBJECT, FEATURE_DIM
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), 'weights')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)

# --- 슬라이딩 윈도우 설정 ---
WINDOW_SIZE = 90   # 윈도우 크기 (패킷 수)
STRIDE = 10        # 슬라이드 간격
# 872 프레임 기준: (872 - 90) // 10 + 1 = 79 windows per sample


class SlidingWindowDataset(Dataset):
    """
    CSIDataset872를 슬라이딩 윈도우로 분할한 Dataset
    - 각 원본 샘플 (872, 456) → 79개의 (90, 456) 윈도우
    - 각 윈도우는 독립 학습 샘플로 취급
    - sample_id: 추론 시 majority voting을 위한 원본 샘플 인덱스
    """
    def __init__(self, base_dataset, window_size=WINDOW_SIZE, stride=STRIDE):
        self.windows = []    # (window_size, 456)
        self.labels = []     # zone label
        self.sample_ids = [] # 원본 샘플 인덱스 (voting용)

        for sample_idx in range(len(base_dataset)):
            x, y = base_dataset[sample_idx]  # (872, 456), scalar
            x_np = x.numpy()
            seq_len = x_np.shape[0]

            start = 0
            while start + window_size <= seq_len:
                window = x_np[start:start + window_size]  # (90, 456)
                self.windows.append(window)
                self.labels.append(y.item())
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


class ZoneLSTM(nn.Module):
    """BiLSTM Zone 분류기 (단일 모델)"""
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
        # bidirectional → hidden_dim * 2
        fc_input = hidden_dim * 2
        self.classifier = nn.Sequential(
            nn.Linear(fc_input, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: (B, window_size, 456)
        _, (hn, _) = self.lstm(x)
        # hn: (num_layers*2, B, hidden_dim)
        # 마지막 레이어의 forward/backward hidden 연결
        last_forward = hn[-2]  # (B, hidden_dim)
        last_backward = hn[-1] # (B, hidden_dim)
        out = torch.cat([last_forward, last_backward], dim=1)  # (B, hidden_dim*2)
        return self.classifier(out)


def majority_vote(base_dataset, window_dataset, model, device):
    """
    원본 샘플별 다수결 voting 정확도 계산
    - 각 샘플의 모든 윈도우 예측 → 가장 많이 나온 class 선택
    """
    model.eval()

    # sample_id별 예측 모음
    sample_preds = {}   # {sample_id: [pred1, pred2, ...]}
    sample_labels = {}  # {sample_id: true_label}

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

    # Majority voting
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

    # 원본 데이터셋 로드 및 정규화
    print("\nLoading data...")
    train_raw = CSIDataset872(TRAIN_SUBJECTS, task='zone')
    test_raw = CSIDataset872([TEST_SUBJECT], task='zone')
    normalize_data(train_raw, test_raw)

    # 슬라이딩 윈도우 데이터셋 생성
    print("\nCreating sliding window datasets...")
    train_win = SlidingWindowDataset(train_raw)
    test_win = SlidingWindowDataset(test_raw)

    train_loader = DataLoader(train_win, batch_size=64, shuffle=True)
    test_win_loader = DataLoader(test_win, batch_size=256, shuffle=False)

    # 모델, 옵티마이저, 스케줄러
    model = ZoneLSTM().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    best_acc = 0
    best_epoch = 0

    print(f"\n{'='*60}")
    print(f"Training BiLSTM (window={WINDOW_SIZE}, stride={STRIDE})")
    print(f"Train windows: {len(train_win)} | Test windows: {len(test_win)}")
    print(f"Test samples (voting): {len(test_raw)}")
    print(f"{'='*60}")

    for epoch in tqdm(range(50), desc="BiLSTM Zone"):
        # --- 학습 (윈도우 단위) ---
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
        train_acc_win = train_correct / len(train_win)
        avg_loss = train_loss / len(train_win)

        # --- 평가: majority voting (샘플 단위) ---
        vote_acc, all_preds, all_labels = majority_vote(test_raw, test_win, model, device)

        if vote_acc > best_acc:
            best_acc = vote_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(),
                       os.path.join(WEIGHTS_DIR, 'zone_lstm_best.pt'))

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:>2}/50] Loss: {avg_loss:.4f} "
                  f"Train(win): {train_acc_win:.4f} | "
                  f"Test(vote): {vote_acc:.4f}")

    print(f"\n✅ Best Test Acc (majority voting): {best_acc:.4f} (epoch {best_epoch})")

    # --- 최종 평가 ---
    model.load_state_dict(torch.load(os.path.join(WEIGHTS_DIR, 'zone_lstm_best.pt'),
                                     map_location=device))
    _, all_preds, all_labels = majority_vote(test_raw, test_win, model, device)

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[f'Zone {i}' for i in range(4)],
                yticklabels=[f'Zone {i}' for i in range(4)], ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'BiLSTM Zone Confusion Matrix (window={WINDOW_SIZE}, stride={STRIDE})\n'
                 f'Best Acc: {best_acc:.4f}')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'zone_lstm_cm.png'), dpi=150)
    plt.close()

    print("\nClassification Report (majority voting):")
    print(classification_report(all_labels, all_preds,
                                target_names=[f'Zone {i}' for i in range(4)]))


if __name__ == '__main__':
    train_model()
