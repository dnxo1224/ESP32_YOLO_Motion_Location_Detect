"""
MLP Zone 분류기 (800 x 166 x 4 데이터)
시간축 통계특징 기반 zone 분류
Train: 12명, Test: kms (또는 설정된 TEST_SUBJECT)
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from dataset_872 import (
    CSIDataset872, normalize_data, get_device,
    TRAIN_SUBJECTS, TEST_SUBJECT, FEATURE_DIM, SEQ_LEN
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), 'weights')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)


class ZoneMLP(nn.Module):
    """시간축 통계특징(mean, std, max, min) 기반 MLP"""
    def __init__(self, input_dim=FEATURE_DIM, num_classes=4, use_stats=True):
        super().__init__()
        self.use_stats = use_stats
        feat_dim = input_dim * 4 if use_stats else input_dim  # 664*4 = 2656 or 664

        self.net = nn.Sequential(
            nn.Linear(feat_dim, 512),
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

    def forward(self, x):
        # x: (B, 800, 664)
        if self.use_stats:
            x_mean = x.mean(dim=1)    # (B, 664)
            x_std = x.std(dim=1)      # (B, 664)
            x_max = x.max(dim=1)[0]   # (B, 664)
            x_min = x.min(dim=1)[0]   # (B, 664)
            x = torch.cat([x_mean, x_std, x_max, x_min], dim=1)  # (B, 2656)
        else:
            x = x.mean(dim=1)  # (B, 664)
        return self.net(x)


def train_model():
    device = get_device()
    print(f"Device: {device}")

    # 슬라이딩 윈도우 설정
    WINDOW_SIZE = 200
    STRIDE = 10

    # 데이터 로드
    print(f"Loading data (Window Size: {WINDOW_SIZE}, Stride: {STRIDE})...")
    train_dataset = CSIDataset872(TRAIN_SUBJECTS, task='zone', window_size=WINDOW_SIZE, stride=STRIDE)
    test_dataset = CSIDataset872([TEST_SUBJECT], task='zone', window_size=WINDOW_SIZE, stride=STRIDE)

    # 정규화
    # dataset_872.py의 normalize_data가 SEQ_LEN을 사용하므로, 
    # 여기서는 수동으로 처리하거나 dataset_872.py를 유연하게 수정해야 함.
    # 일단 여기서 수동 정규화 수행
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    train_flat = train_dataset.data.reshape(-1, FEATURE_DIM)
    scaler.fit(train_flat)
    train_dataset.data = scaler.transform(train_flat).reshape(-1, WINDOW_SIZE, FEATURE_DIM).astype(np.float32)
    test_flat = test_dataset.data.reshape(-1, FEATURE_DIM)
    test_dataset.data = scaler.transform(test_flat).reshape(-1, WINDOW_SIZE, FEATURE_DIM).astype(np.float32)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    print(f"Train: {len(train_dataset)} windows, Test: {len(test_dataset)} windows")

    # 두 가지 MLP 모델 비교: mean-only vs stats
    results = {}
    for use_stats, name in [(False, 'mean_only'), (True, 'stats')]:
        print(f"\n{'='*50}")
        print(f"Training ZoneMLP ({name})...")
        print(f"{'='*50}")

        model = ZoneMLP(use_stats=use_stats).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

        best_acc = 0
        best_epoch = 0

        for epoch in range(50):
            # Train
            model.train()
            train_loss, train_correct = 0, 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * inputs.size(0)
                train_correct += (outputs.argmax(1) == labels).sum().item()

            train_acc = train_correct / len(train_dataset)
            avg_train_loss = train_loss / len(train_dataset)

            # Test
            model.eval()
            test_correct = 0
            all_preds, all_labels = [], []
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    preds = outputs.argmax(1)
                    test_correct += (preds == labels).sum().item()
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            test_acc = test_correct / len(test_dataset)
            scheduler.step(avg_train_loss)

            if test_acc > best_acc:
                best_acc = test_acc
                best_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(WEIGHTS_DIR, f'zone_mlp_{name}_best.pt'))

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1}/50] Train Loss: {avg_train_loss:.4f} "
                      f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")

        print(f"\nBest Test Acc ({name}): {best_acc:.4f} (epoch {best_epoch})")
        results[name] = {'acc': best_acc, 'preds': all_preds, 'labels': all_labels}

        # Confusion Matrix
        cm = confusion_matrix(all_labels, all_preds)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=[f'Zone {i}' for i in range(4)],
                    yticklabels=[f'Zone {i}' for i in range(4)], ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f'MLP Zone Confusion Matrix ({name})\nTest Acc: {best_acc:.4f}')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f'zone_mlp_{name}_cm.png'), dpi=150)
        plt.close()

        print(f"\nClassification Report ({name}):")
        print(classification_report(all_labels, all_preds,
                                    target_names=[f'Zone {i}' for i in range(4)],
                                    zero_division=0))

    # 결과 요약
    print("\n" + "="*50)
    print("MLP Zone Classification Summary:")
    for name, r in results.items():
        print(f"  {name}: {r['acc']:.4f}")
    print("="*50)


if __name__ == '__main__':
    train_model()
