"""
WiDR 학습 스크립트 — Wi-Fi Dual-task Recognition

논문: "Wi-Fi CSI-Based Human Activity Recognition and Indoor Localization
       with Sampling Irregularity Mitigation" (Lee & Toh, IEEE IoT Journal 2025)

두 가지 평가 모드:
  Mode A: 11명 train / kjh+kms 2명 test (Subject-Split)
  Mode B: Subject-wise 5-Fold Cross-Validation
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from dataset import CSIDataset, normalize_data, get_device, ALL_SUBJECTS, NUM_ACTIONS, NUM_ZONES
from model import WiDRNet

# ============================================================
# 하이퍼파라미터 (논문 Section IV-B-2 최적값)
# ============================================================
EPOCHS = 50           # 논문 ~20 에폭 수렴, 여유 있게 50
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-4
LAMBDA_REG = 0.7      # 논문 최적 parameter sharing 강도
MIXUP_ALPHA = 0.4     # 논문 최적 Beta 분포 파라미터
GRAD_CLIP = 1.0
D_MODEL = 128
NUM_HEADS = 4         # 논문 최적
D_FFN = 256           # 논문 최적
DROPOUT = 0.1

# 경로
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
WEIGHTS_DIR = os.path.join(SCRIPT_DIR, 'weights')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)

ACTION_NAMES = ['handsup', 'sit', 'stand', 'walk']
ZONE_NAMES = ['Zone0', 'Zone1', 'Zone2', 'Zone3']


# ============================================================
# Mixup 데이터 증강 (논문 Eq.17)
# ============================================================
def mixup_data(x, y_action, y_zone, alpha=MIXUP_ALPHA):
    """
    Mixup: X̃ = η*X + (1-η)*X', Ỹ = η*Y + (1-η)*Y'
    η ~ Beta(α, α)
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_action_b = y_action[index]
    y_zone_b = y_zone[index]

    return mixed_x, y_action, y_zone, y_action_b, y_zone_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup 손실: λ*CE(pred, y_a) + (1-λ)*CE(pred, y_b)"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ============================================================
# 평가 함수
# ============================================================
@torch.no_grad()
def evaluate(model, loader, device):
    """모델 평가 — action/zone 정확도 및 예측 결과 반환"""
    model.eval()
    all_action_preds, all_action_labels = [], []
    all_zone_preds, all_zone_labels = [], []

    for x, y_act, y_zone in loader:
        x = x.to(device)
        action_logits, zone_logits = model(x)

        all_action_preds.append(action_logits.argmax(dim=1).cpu())
        all_action_labels.append(y_act)
        all_zone_preds.append(zone_logits.argmax(dim=1).cpu())
        all_zone_labels.append(y_zone)

    action_preds = torch.cat(all_action_preds).numpy()
    action_labels = torch.cat(all_action_labels).numpy()
    zone_preds = torch.cat(all_zone_preds).numpy()
    zone_labels = torch.cat(all_zone_labels).numpy()

    action_acc = (action_preds == action_labels).mean() * 100
    zone_acc = (zone_preds == zone_labels).mean() * 100

    return action_acc, zone_acc, action_preds, action_labels, zone_preds, zone_labels


# ============================================================
# Confusion Matrix 저장
# ============================================================
def save_confusion_matrix(y_true, y_pred, class_names, title, filepath):
    """Confusion matrix를 히트맵으로 저장"""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"  Confusion matrix saved: {filepath}")


# ============================================================
# 단일 학습 루프
# ============================================================
def train_one_split(train_subjects, test_subjects, device, tag="", save_best=True):
    """
    하나의 train/test split에 대해 학습 및 평가 수행

    Args:
        train_subjects: 학습 피험자 리스트
        test_subjects: 테스트 피험자 리스트
        device: torch device
        tag: 결과 저장 시 접두어 (e.g., "modeA", "fold1")
        save_best: 최고 모델 체크포인트 저장 여부

    Returns:
        best_action_acc, best_zone_acc
    """
    # --- 데이터 로드 ---
    print(f"\n{'='*60}")
    print(f"[{tag}] Train: {train_subjects}")
    print(f"[{tag}] Test:  {test_subjects}")
    print(f"{'='*60}")

    train_dataset = CSIDataset(train_subjects)
    test_dataset = CSIDataset(test_subjects)
    normalize_data(train_dataset, test_dataset)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- 모델 ---
    model = WiDRNet(
        d_model=D_MODEL, num_heads=NUM_HEADS, d_ffn=D_FFN,
        num_actions=NUM_ACTIONS, num_zones=NUM_ZONES, dropout=DROPOUT
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_mean_acc = 0
    best_action_acc = 0
    best_zone_acc = 0
    best_results = None

    # --- 학습 루프 ---
    for epoch in tqdm(range(EPOCHS), desc=f"[{tag}] Epochs"):
        model.train()
        total_loss = 0
        num_batches = 0

        for x, y_act, y_zone in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            x = x.to(device)
            y_act = y_act.to(device)
            y_zone = y_zone.to(device)

            # Mixup (논문 Eq.17)
            mixed_x, y_a_act, y_a_zone, y_b_act, y_b_zone, lam = mixup_data(x, y_act, y_zone)

            # Forward
            action_logits, zone_logits = model(mixed_x)

            # Total Loss (논문 Eq.8)
            loss_action = mixup_criterion(criterion, action_logits, y_a_act, y_b_act, lam)
            loss_zone = mixup_criterion(criterion, zone_logits, y_a_zone, y_b_zone, lam)
            loss_reg = model.param_reg_loss()
            loss = loss_action + loss_zone + LAMBDA_REG * loss_reg

            # Backward
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        scheduler.step()

        # --- Epoch 평가 ---
        avg_loss = total_loss / max(num_batches, 1)
        action_acc, zone_acc, a_pred, a_true, z_pred, z_true = evaluate(model, test_loader, device)
        mean_acc = (action_acc + zone_acc) / 2

        tqdm.write(f"  [{tag}] Epoch {epoch+1:3d} | Loss: {avg_loss:.4f} | "
                   f"Action: {action_acc:.2f}% | Zone: {zone_acc:.2f}% | Mean: {mean_acc:.2f}%")

        if mean_acc > best_mean_acc:
            best_mean_acc = mean_acc
            best_action_acc = action_acc
            best_zone_acc = zone_acc
            best_results = (a_pred, a_true, z_pred, z_true)
            if save_best:
                torch.save(model.state_dict(), os.path.join(WEIGHTS_DIR, f'widr_{tag}_best.pt'))

    # --- 최종 결과 출력 ---
    print(f"\n[{tag}] Best Results:")
    print(f"  Action Accuracy: {best_action_acc:.2f}%")
    print(f"  Zone Accuracy:   {best_zone_acc:.2f}%")
    print(f"  Mean Accuracy:   {best_mean_acc:.2f}%")

    if best_results is not None:
        a_pred, a_true, z_pred, z_true = best_results

        print(f"\n[{tag}] Action Classification Report:")
        print(classification_report(a_true, a_pred, target_names=ACTION_NAMES, zero_division=0))

        print(f"[{tag}] Zone Classification Report:")
        print(classification_report(z_true, z_pred, target_names=ZONE_NAMES, zero_division=0))

        if save_best:
            save_confusion_matrix(
                a_true, a_pred, ACTION_NAMES,
                f'WiDR Action CM ({tag})',
                os.path.join(RESULTS_DIR, f'widr_{tag}_action_cm.png')
            )
            save_confusion_matrix(
                z_true, z_pred, ZONE_NAMES,
                f'WiDR Zone CM ({tag})',
                os.path.join(RESULTS_DIR, f'widr_{tag}_zone_cm.png')
            )

    return best_action_acc, best_zone_acc


# ============================================================
# Mode A: Subject-Split (11 train / 2 test)
# ============================================================
def run_mode_a(device):
    """Mode A: 11명 train / kjh+kms 2명 test"""
    print("\n" + "=" * 70)
    print("  MODE A: Subject-Split (11 train / kjh+kms test)")
    print("=" * 70)

    test_subjects = ['kjh', 'kms']
    train_subjects = [s for s in ALL_SUBJECTS if s not in test_subjects]

    action_acc, zone_acc = train_one_split(
        train_subjects, test_subjects, device,
        tag="modeA", save_best=True
    )

    return action_acc, zone_acc


# ============================================================
# Mode B: Subject-wise 5-Fold CV
# ============================================================
def run_mode_b(device):
    """
    Mode B: Subject-wise 5-Fold Cross-Validation

    13명을 5그룹으로 나눠 교차 검증:
      Fold 1: test=[gyj, jhj, jkw]
      Fold 2: test=[kjh, kmh, kms]
      Fold 3: test=[kye, lsi, mhe]
      Fold 4: test=[phr, stk]
      Fold 5: test=[swt, ysj]
    """
    print("\n" + "=" * 70)
    print("  MODE B: Subject-wise 5-Fold Cross-Validation")
    print("=" * 70)

    folds = [
        ['gyj', 'jhj', 'jkw'],
        ['kjh', 'kmh', 'kms'],
        ['kye', 'lsi', 'mhe'],
        ['phr', 'stk'],
        ['swt', 'ysj'],
    ]

    action_accs = []
    zone_accs = []

    for i, test_subjects in enumerate(folds):
        train_subjects = [s for s in ALL_SUBJECTS if s not in test_subjects]
        action_acc, zone_acc = train_one_split(
            train_subjects, test_subjects, device,
            tag=f"fold{i+1}", save_best=False
        )
        action_accs.append(action_acc)
        zone_accs.append(zone_acc)

    # --- 5-Fold 결과 요약 ---
    print("\n" + "=" * 70)
    print("  5-Fold CV Results Summary")
    print("=" * 70)
    print(f"{'Fold':<8} {'Action(%)':<12} {'Zone(%)':<12} {'Mean(%)':<12}")
    print("-" * 44)
    for i in range(len(folds)):
        mean_acc = (action_accs[i] + zone_accs[i]) / 2
        print(f"Fold {i+1:<3} {action_accs[i]:<12.2f} {zone_accs[i]:<12.2f} {mean_acc:<12.2f}")

    action_mean = np.mean(action_accs)
    action_std = np.std(action_accs)
    zone_mean = np.mean(zone_accs)
    zone_std = np.std(zone_accs)
    overall_mean = (action_mean + zone_mean) / 2

    print("-" * 44)
    print(f"{'Mean':<8} {action_mean:<12.2f} {zone_mean:<12.2f} {overall_mean:<12.2f}")
    print(f"{'Std':<8} {action_std:<12.2f} {zone_std:<12.2f}")
    print(f"\nAction: {action_mean:.2f} +/- {action_std:.2f}%")
    print(f"Zone:   {zone_mean:.2f} +/- {zone_std:.2f}%")
    print(f"Mean:   {overall_mean:.2f}%")

    return action_mean, action_std, zone_mean, zone_std


# ============================================================
# Main
# ============================================================
def main():
    device = get_device()
    print(f"Device: {device}")
    print(f"Hyperparameters: epochs={EPOCHS}, batch={BATCH_SIZE}, lr={LR}, "
          f"lambda={LAMBDA_REG}, mixup_alpha={MIXUP_ALPHA}")
    print(f"Model: d_model={D_MODEL}, heads={NUM_HEADS}, d_ffn={D_FFN}, dropout={DROPOUT}")

    # Mode A: Subject-Split
    a_action, a_zone = run_mode_a(device)

    # Mode B: 5-Fold CV
    b_action_mean, b_action_std, b_zone_mean, b_zone_std = run_mode_b(device)

    # --- 최종 요약 ---
    print("\n" + "=" * 70)
    print("  FINAL SUMMARY")
    print("=" * 70)
    print(f"\nMode A (11 train / kjh+kms test):")
    print(f"  Action: {a_action:.2f}%")
    print(f"  Zone:   {a_zone:.2f}%")
    print(f"  Mean:   {(a_action + a_zone) / 2:.2f}%")

    print(f"\nMode B (5-Fold CV):")
    print(f"  Action: {b_action_mean:.2f} +/- {b_action_std:.2f}%")
    print(f"  Zone:   {b_zone_mean:.2f} +/- {b_zone_std:.2f}%")
    print(f"  Mean:   {(b_action_mean + b_zone_mean) / 2:.2f}%")


if __name__ == '__main__':
    main()
