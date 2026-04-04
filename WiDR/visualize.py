"""
WiDR 시각화 모듈

포함 기능:
  - Raw CSI 히트맵 / 시계열 (데이터 탐색)
  - 학습 곡선 (Loss, Action/Zone Accuracy)
  - Confusion Matrix

단독 실행 시 저장된 가중치로 Mode A 재평가 + 시각화:
    python visualize.py
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from config import (
    RAW_DATA_DIR, RESULTS_DIR, WEIGHTS_DIR,
    ACTION_NAMES, ZONE_NAMES,
    NUM_ACTIONS, NUM_ZONES,
    DEVICE, ALL_SUBJECTS,
)

# 모델 하이퍼파라미터 (train.py와 동일하게 유지)
D_MODEL = 128
NUM_HEADS = 4
D_FFN = 256


# ============================================================
# Raw CSI 탐색 시각화
# ============================================================
def plot_raw_csi(amp_matrix, filename=""):
    """
    Raw amplitude matrix 히트맵 + 서브캐리어 시계열 시각화

    Args:
        amp_matrix: (N, 114) amplitude 배열
        filename: 플롯 제목에 표시할 파일명
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    axes[0].imshow(amp_matrix.T, aspect='auto', cmap='viridis', origin='lower')
    axes[0].set_xlabel('Time (packet index)')
    axes[0].set_ylabel('Subcarrier index')
    axes[0].set_title(f'Raw Amplitude Heatmap\n{filename}')

    for sc in [10, 50, 100]:
        axes[1].plot(amp_matrix[:, sc], label=f'SC {sc}', alpha=0.7)
    axes[1].set_xlabel('Time (packet index)')
    axes[1].set_ylabel('Amplitude')
    axes[1].set_title('Subcarrier Time Series')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'raw_csi_sample.png'), dpi=150)
    plt.show()


def explore_raw_data():
    """RAW_DATA_DIR 첫 번째 파일로 Raw CSI 탐색 시각화 수행"""
    from preprocess import preprocess_raw_csv

    rx1_files = sorted(glob.glob(os.path.join(RAW_DATA_DIR, '*_rx1.csv')))
    if not rx1_files:
        print("Raw data 파일이 없습니다.")
        return

    sample_file = rx1_files[0]
    print(f"샘플 파일: {os.path.basename(sample_file)}")
    seq_ids, amp_matrix = preprocess_raw_csv(sample_file)
    print(f"Amplitude matrix shape: {amp_matrix.shape}")
    plot_raw_csi(amp_matrix, os.path.basename(sample_file))


# ============================================================
# 학습 곡선
# ============================================================
def plot_learning_curves(history, tag="modeA"):
    """
    학습 이력(dict)으로 Loss / Action Acc / Zone Acc / Mean Acc 곡선 저장

    Args:
        history: {'loss': [...], 'action_acc': [...], 'zone_acc': [...], 'mean_acc': [...]}
        tag: 저장 파일명 접두어
    """
    if not history:
        print("학습 이력이 없습니다.")
        return

    epochs_range = range(1, len(history['loss']) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].plot(epochs_range, history['loss'], 'b-')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'{tag}: Training Loss')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs_range, history['action_acc'], 'r-', label='Action')
    axes[1].plot(epochs_range, history['zone_acc'], 'g-', label='Zone')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title(f'{tag}: Test Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs_range, history['mean_acc'], 'm-')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Mean Accuracy (%)')
    axes[2].set_title(f'{tag}: Mean Dual-task Accuracy')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, f'{tag}_learning_curve.png')
    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f"학습 곡선 저장: {out_path}")


def plot_fold_curves(fold_histories):
    """5-Fold CV 각 fold 학습 곡선 비교"""
    if not fold_histories or not fold_histories[0]:
        print("Fold 학습 이력이 없습니다.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for i, hist in enumerate(fold_histories):
        if hist:
            epochs_range = range(1, len(hist['action_acc']) + 1)
            axes[0].plot(epochs_range, hist['action_acc'], label=f'Fold {i+1}', alpha=0.7)
            axes[1].plot(epochs_range, hist['zone_acc'], label=f'Fold {i+1}', alpha=0.7)

    for ax, title in zip(axes, ['5-Fold CV: Action Recognition', '5-Fold CV: Zone Classification']):
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, 'modeB_5fold_curves.png')
    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f"5-Fold 곡선 저장: {out_path}")


# ============================================================
# Confusion Matrix
# ============================================================
def plot_confusion_matrix(y_true, y_pred, class_names, title, save_path=None):
    """Confusion Matrix 시각화 및 저장"""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Confusion matrix 저장: {save_path}")
    plt.show()


# ============================================================
# 단독 실행: 저장된 Mode A 가중치로 재평가 + 시각화
# ============================================================
if __name__ == '__main__':
    import torch
    from torch.utils.data import DataLoader
    from dataset import CSIDataset, normalize_data
    from model import WiDRNet
    from train import evaluate, BATCH_SIZE

    weight_path = os.path.join(WEIGHTS_DIR, 'widr_modeA_best.pt')
    if not os.path.exists(weight_path):
        print(f"가중치 파일 없음: {weight_path}")
        print("먼저 train.py를 실행하세요.")
        exit(1)

    test_subjects = ['kjh', 'kms']
    train_subjects = [s for s in ALL_SUBJECTS if s not in test_subjects]

    train_ds = CSIDataset(train_subjects)
    test_ds = CSIDataset(test_subjects)
    normalize_data(train_ds, test_ds)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = WiDRNet(
        d_model=D_MODEL, num_heads=NUM_HEADS, d_ffn=D_FFN,
        num_actions=NUM_ACTIONS, num_zones=NUM_ZONES,
    ).to(DEVICE)
    model.load_state_dict(torch.load(weight_path, map_location=DEVICE, weights_only=True))

    a_acc, z_acc, a_pred, a_true, z_pred, z_true = evaluate(model, test_loader, DEVICE)
    print(f"Mode A — Action: {a_acc:.2f}%, Zone: {z_acc:.2f}%")

    plot_confusion_matrix(
        a_true, a_pred, ACTION_NAMES,
        f'Action CM (Mode A) | Acc: {a_acc:.2f}%',
        os.path.join(RESULTS_DIR, 'modeA_action_cm.png'),
    )
    plot_confusion_matrix(
        z_true, z_pred, ZONE_NAMES,
        f'Zone CM (Mode A) | Acc: {z_acc:.2f}%',
        os.path.join(RESULTS_DIR, 'modeA_zone_cm.png'),
    )
