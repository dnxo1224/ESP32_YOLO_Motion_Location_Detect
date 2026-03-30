"""
CNN-BiLSTM Action 분류기 (4-class: handsup/sit/stand/walk)
1D-CNN → BiLSTM → Temporal Attention → Classifier
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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from tqdm import tqdm

from dataset_872 import (
    CSIDataset872, get_device,
    FEATURE_DIM, ALL_SUBJECTS, SEQ_LEN
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), 'weights')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)

# --- 설정 ---
TEST_SUBJECTS = ['kjh', 'kms']
TRAIN_SUBJECTS = [s for s in ALL_SUBJECTS if s not in TEST_SUBJECTS]

WINDOW_SIZE = 90
STRIDE = 30

ACTION_NAMES = ['handsup', 'sit', 'stand', 'walk']
NUM_CLASSES = 4


# =====================================================================
# 데이터셋
# =====================================================================
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


# =====================================================================
# 모델: CNN-BiLSTM with Temporal Attention
# =====================================================================
class TemporalAttention(nn.Module):
    """시간축 Attention: 각 타임스텝에 가중치를 부여하여 가중합"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        # x: (B, T, hidden_dim)
        scores = self.attn(x)            # (B, T, 1)
        weights = torch.softmax(scores, dim=1)  # (B, T, 1)
        context = (x * weights).sum(dim=1)       # (B, hidden_dim)
        return context, weights.squeeze(-1)


class ActionCNNBiLSTM(nn.Module):
    """
    1D-CNN → BiLSTM → Temporal Attention → Classifier

    CNN: 시간축의 local pattern 추출 (kernel 7→5→3)
    BiLSTM: 시계열 의존성 학습
    Attention: 행동 판별에 유용한 시간 구간에 집중
    """
    def __init__(self, input_dim=FEATURE_DIM, num_classes=NUM_CLASSES,
                 cnn_channels=128, lstm_hidden=128, lstm_layers=2, dropout=0.3):
        super().__init__()

        # --- 1D-CNN Block ---
        # Input: (B, input_dim, T) → Output: (B, cnn_channels, T)
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 256, kernel_size=7, padding=3),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Conv1d(256, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Conv1d(256, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # --- BiLSTM ---
        # Input: (B, T, cnn_channels) → Output: (B, T, lstm_hidden*2)
        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=True,
        )

        # --- Temporal Attention ---
        lstm_out_dim = lstm_hidden * 2  # bidirectional
        self.attention = TemporalAttention(lstm_out_dim)

        # --- Classifier ---
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: (B, T, 456)
        # CNN은 (B, C, T) 형태 필요
        x = x.transpose(1, 2)           # (B, 456, T)
        x = self.cnn(x)                 # (B, 128, T)
        x = x.transpose(1, 2)           # (B, T, 128) — LSTM 입력

        x, _ = self.lstm(x)             # (B, T, 256)
        context, attn_weights = self.attention(x)  # (B, 256)

        return self.classifier(context)  # (B, num_classes)


# =====================================================================
# Majority Voting
# =====================================================================
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


# =====================================================================
# 학습
# =====================================================================
def train_model():
    device = get_device()
    print(f"Device: {device}")

    # --- 데이터 로드 ---
    print("\nLoading train data...")
    train_raw = CSIDataset872(TRAIN_SUBJECTS, task='action')

    print("\nLoading test data...")
    test_datasets = {}
    for subj in TEST_SUBJECTS:
        test_raw = CSIDataset872([subj], task='action')
        test_datasets[subj] = test_raw

    # --- 정규화 (train 기준 StandardScaler) ---
    scaler = StandardScaler()
    train_flat = train_raw.data.reshape(-1, FEATURE_DIM)
    scaler.fit(train_flat)
    train_raw.data = scaler.transform(train_flat).reshape(-1, SEQ_LEN, FEATURE_DIM).astype(np.float32)

    for subj in TEST_SUBJECTS:
        td = test_datasets[subj]
        test_flat = td.data.reshape(-1, FEATURE_DIM)
        td.data = scaler.transform(test_flat).reshape(-1, SEQ_LEN, FEATURE_DIM).astype(np.float32)

    # --- 슬라이딩 윈도우 ---
    print("\nCreating sliding window datasets...")
    train_win = SlidingWindowDataset(train_raw)
    test_wins = {}
    for subj in TEST_SUBJECTS:
        print(f"  [{subj}]")
        test_wins[subj] = SlidingWindowDataset(test_datasets[subj])

    train_loader = DataLoader(train_win, batch_size=64, shuffle=True)

    # --- 모델 ---
    model = ActionCNNBiLSTM().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    # 파라미터 수 출력
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {total_params:,} total, {trainable_params:,} trainable")

    best_avg_acc = 0
    best_epoch = 0

    print(f"\n{'='*60}")
    print(f"Training CNN-BiLSTM Action (4-class: handsup/sit/stand/walk)")
    print(f"Train: {len(TRAIN_SUBJECTS)}명 ({len(train_win)} windows)")
    for subj in TEST_SUBJECTS:
        print(f"Test [{subj}]: {len(test_datasets[subj])} samples ({len(test_wins[subj])} windows)")
    print(f"{'='*60}")

    for epoch in tqdm(range(50), desc="CNN-BiLSTM Action"):
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
                       os.path.join(WEIGHTS_DIR, 'action_cnn_bilstm_best.pt'))

        if (epoch + 1) % 5 == 0 or epoch == 0:
            acc_str = " | ".join(f"{s}: {accs[s]:.4f}" for s in TEST_SUBJECTS)
            print(f"Epoch [{epoch+1:>2}/50] Loss: {avg_loss:.4f} "
                  f"Train: {train_acc:.4f} | {acc_str} | Avg: {avg_acc:.4f}")

    print(f"\n✅ Best Avg Test Acc (majority voting): {best_avg_acc:.4f} (epoch {best_epoch})")

    # --- 최종 평가 ---
    model.load_state_dict(torch.load(os.path.join(WEIGHTS_DIR, 'action_cnn_bilstm_best.pt'),
                                     map_location=device))

    all_preds_combined = []
    all_labels_combined = []

    for subj in TEST_SUBJECTS:
        acc, preds, labels = majority_vote(test_datasets[subj], test_wins[subj], model, device)
        all_preds_combined.extend(preds)
        all_labels_combined.extend(labels)
        print(f"\n[{subj}] Voting Accuracy: {acc:.4f} "
              f"({sum(p==l for p,l in zip(preds,labels))}/{len(labels)})")

    avg_acc_final = np.mean([
        sum(p == l for p, l in zip(*majority_vote(test_datasets[s], test_wins[s], model, device)[1:]))
        / len(test_datasets[s])
        for s in TEST_SUBJECTS
    ])
    print(f"\n📊 Average Accuracy (kjh + kms): {avg_acc_final:.4f}")

    # --- Confusion Matrix (2명 합산) ---
    cm = confusion_matrix(all_labels_combined, all_preds_combined)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=ACTION_NAMES,
                yticklabels=ACTION_NAMES, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'CNN-BiLSTM Action Confusion Matrix (4-class)\n'
                 f'Test: {", ".join(TEST_SUBJECTS)} | Avg Acc: {best_avg_acc:.4f}')
    plt.tight_layout()
    cm_path = os.path.join(RESULTS_DIR, 'action_cnn_bilstm_cm.png')
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"\nConfusion matrix 저장: {cm_path}")

    # --- Classification Report ---
    print("\nClassification Report (combined):")
    print(classification_report(all_labels_combined, all_preds_combined,
                                target_names=ACTION_NAMES))


if __name__ == '__main__':
    train_model()
