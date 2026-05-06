"""
Zone Expert Action Classifier — Vanilla RNN 학습 스크립트

[사용법]
  ZONE_ID 변수를 0 / 1 / 2 / 3 으로 직접 바꾼 뒤 실행하면
  해당 Zone 전문가 RNN 모델을 학습한다.

[평가 방식]
  윈도우 1개 → 즉시 예측 (실시간 추론 방식과 동일)
  Majority voting 미사용

[출력]
  weights_rnn/zone_expert_action_{ZONE_ID}_best.pt
  results/rnn_zone_expert_action_{ZONE_ID}_cm.png
"""
import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

# ─── 재현성을 위한 랜덤 시드 고정 ────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

from dataset import (
    CSIRawDataset, normalize_datasets, SlidingWindowDataset, get_device,
    TRAIN_SUBJECTS, TEST_SUBJECT,
)
from model_rnn import ZoneExpertRNN

# ─── 학습할 Zone 전문가 번호 ────────────────────────────────────────────────
# None → Zone 0~3 전체 순서대로 학습
# 특정 Zone만 학습하려면 0 / 1 / 2 / 3 으로 직접 지정
ZONE_ID = None

# ─── 하이퍼파라미터 ────────────────────────────────────────────────────────────
WINDOW_SIZE  = 200
STRIDE       = 20
BATCH_SIZE   = 64
EPOCHS       = 80        # vanilla RNN은 수렴이 느리므로 LSTM(50)보다 더 많은 에폭 사용
LR           = 5e-4      # vanilla RNN 학습 불안정성 방지를 위해 LSTM(1e-3)보다 낮게 설정
WEIGHT_DECAY = 1e-4
GRAD_CLIP    = 1.0       # vanishing/exploding gradient 방지 필수

ACTION_NAMES = ['handsup', 'sit', 'stand', 'walk']

# ─── 저장 경로 ────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(__file__)
WEIGHTS_DIR = os.path.join(BASE_DIR, 'weights_rnn')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ─── 윈도우 단위 평가 ─────────────────────────────────────────────────────────

def evaluate(model: nn.Module,
             win_ds: SlidingWindowDataset,
             device: torch.device):
    model.eval()
    loader = DataLoader(win_ds, batch_size=256, shuffle=False)

    all_preds, all_labels = [], []

    with torch.no_grad():
        for x, y_action, _, _ in loader:
            x = x.to(device)
            preds = model(x).argmax(1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(y_action.numpy().tolist())

    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    return acc, all_preds, all_labels


# ─── 혼동행렬 저장 ────────────────────────────────────────────────────────────

def save_confusion_matrix(preds, labels, zone_id, best_acc):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=ACTION_NAMES,
                yticklabels=ACTION_NAMES, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Zone {zone_id} Expert (RNN) — Action CM (window-level)\n'
                 f'Best Acc: {best_acc:.4f}')
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, f'rnn_zone_expert_action_{zone_id}_cm.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[CM saved] {out_path}")


# ─── 학습 루프 ────────────────────────────────────────────────────────────────

def train(zone_id: int = ZONE_ID):
    device = get_device()
    print(f"\n{'='*50}")
    print(f"Device : {device}")
    print(f"Zone ID: {zone_id}  (Zone {zone_id} 전문가 RNN 학습)")

    # ── 데이터 로드 ──────────────────────────────────────────────────────────
    print("\n[1] Loading raw datasets...")
    train_raw = CSIRawDataset(TRAIN_SUBJECTS)
    test_raw  = CSIRawDataset([TEST_SUBJECT])

    # ── 정규화 ───────────────────────────────────────────────────────────────
    print("\n[2] Normalizing (train-fit)...")
    normalize_datasets(train_raw, test_raw)

    # ── 슬라이딩 윈도우 ──────────────────────────────────────────────────────
    print(f"\n[3] Creating sliding windows (size={WINDOW_SIZE}, stride={STRIDE})...")
    train_win = SlidingWindowDataset(train_raw, WINDOW_SIZE, STRIDE)
    test_win  = SlidingWindowDataset(test_raw,  WINDOW_SIZE, STRIDE)

    train_win.oversample_zone(zone_id, factor=2)

    train_loader = DataLoader(train_win, batch_size=BATCH_SIZE, shuffle=True)

    print(f"  Train windows : {len(train_win)} (Zone {zone_id} ×2 오버샘플링 적용)")
    print(f"  Test  windows : {len(test_win)} (전체 Zone, 필터 없음)")

    # ── 모델 ─────────────────────────────────────────────────────────────────
    print(f"\n[4] Building ZoneExpertRNN (zone_id={zone_id})...")
    model = ZoneExpertRNN(zone_id=zone_id).to(device)
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_acc   = 0.0
    best_epoch = 0
    ckpt_path  = os.path.join(WEIGHTS_DIR, f'zone_expert_action_{zone_id}_best.pt')

    # ── 학습 ─────────────────────────────────────────────────────────────────
    print(f"\n[5] Training (epochs={EPOCHS})...")
    for epoch in tqdm(range(EPOCHS), desc=f'Zone{zone_id} Expert RNN'):
        model.train()
        total_loss, total_correct = 0.0, 0

        for x, y_action, _, _ in tqdm(train_loader,
                                      desc=f'  Epoch {epoch+1:02d}',
                                      leave=False):
            x, y_action = x.to(device), y_action.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss   = criterion(logits, y_action)
            loss.backward()
            clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            total_loss    += loss.item() * x.size(0)
            total_correct += (logits.argmax(1) == y_action).sum().item()

        scheduler.step()

        train_acc = total_correct / len(train_win)
        avg_loss  = total_loss    / len(train_win)

        test_acc, _, _ = evaluate(model, test_win, device)

        if test_acc > best_acc:
            best_acc   = test_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), ckpt_path)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            tqdm.write(
                f"  Epoch [{epoch+1:02d}/{EPOCHS}] "
                f"Loss: {avg_loss:.4f}  Train: {train_acc:.4f}  "
                f"Test(Win): {test_acc:.4f}  Best: {best_acc:.4f}"
            )

    print(f"\n[Best] Zone {zone_id} Expert RNN — Acc: {best_acc:.4f}  (epoch {best_epoch})")
    print(f"       Saved → {ckpt_path}")

    # ── 최종 평가 ─────────────────────────────────────────────────────────────
    print("\n[6] Final evaluation with best checkpoint...")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    _, final_preds, final_labels = evaluate(model, test_win, device)

    print(f"\nClassification Report (Zone {zone_id} Expert RNN, window-level, 전체 테스트):")
    print(classification_report(final_labels, final_preds,
                                target_names=ACTION_NAMES,
                                zero_division=0))

    save_confusion_matrix(final_preds, final_labels, zone_id, best_acc)


if __name__ == '__main__':
    zones = range(4) if ZONE_ID is None else [ZONE_ID]
    for z in zones:
        train(z)
