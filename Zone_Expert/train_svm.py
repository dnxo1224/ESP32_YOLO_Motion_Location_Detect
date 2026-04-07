"""
Zone Expert Action Classifier — SVM 학습 스크립트

[피처 추출 방식]
  윈도우 (200, 664) → RX별 슬라이싱 (200, 166) × 4
  RX당 3가지 피처 추출:
    1) 시간축 평균     : (166,)  — 정적 채널 특성
    2) 서브캐리어 그래디언트 평균 : (165,)  — 주파수 도메인 공간 지문
    3) 시간축 그래디언트 평균     : (166,)  — 프레임 간 변화량
  4 RX 연결: 4 × 497 = 1,988D 벡터

[Zone 가중치]
  해당 Zone의 RX 피처 블록(497D)에 ×2.0 스케일링

[사용법]
  ZONE_ID 변수를 0 / 1 / 2 / 3 으로 직접 바꾼 뒤 실행

[출력]
  weights_svm/zone_expert_action_{ZONE_ID}.pkl  (SVM + feature_scaler)
  weights_svm/data_scaler.pkl                   (normalize_datasets 스케일러, Zone 0 학습 시 1회 저장)
  results/svm_zone_expert_action_{ZONE_ID}_cm.png
"""
import os
import sys
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(__file__))

from dataset import (
    CSIRawDataset, normalize_datasets, SlidingWindowDataset,
    TRAIN_SUBJECTS, TEST_SUBJECT,
)

# ─── 학습할 Zone 전문가 번호 ────────────────────────────────────────────────
# None → Zone 0~3 전체 순서대로 학습
# 특정 Zone만 학습하려면 0 / 1 / 2 / 3 으로 직접 지정
ZONE_ID = None

# ─── SVM 하이퍼파라미터 ───────────────────────────────────────────────────────
SVM_C     = 10.0
SVM_GAMMA = 'scale'   # 1 / (n_features * X.var()) 자동 계산

# ─── Zone RX 매핑 ─────────────────────────────────────────────────────────────
ZONE_RX_MAP  = {0: 0, 1: 1, 2: 2, 3: 3}
RX_HIGH_W    = 1.2    # 해당 Zone RX 피처 블록 강조 배율
FEATURES_PER_RX = 497 # 166 + 165 + 166

# ─── 슬라이딩 윈도우 설정 ─────────────────────────────────────────────────────
WINDOW_SIZE = 200
STRIDE      = 20

ACTION_NAMES = ['handsup', 'sit', 'stand', 'walk']

# ─── 저장 경로 ────────────────────────────────────────────────────────────────
BASE_DIR         = os.path.dirname(__file__)
WEIGHTS_DIR      = os.path.join(BASE_DIR, 'weights_svm')
RESULTS_DIR      = os.path.join(BASE_DIR, 'results')
DATA_SCALER_PATH = os.path.join(WEIGHTS_DIR, 'data_scaler.pkl')
os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ─── 피처 추출 ────────────────────────────────────────────────────────────────

def extract_features(windows: np.ndarray) -> np.ndarray:
    """
    윈도우 배열에서 SVM 입력 피처 벡터를 추출한다.

    Args:
        windows: (N, 200, 664) — 정규화 + mean removal 완료된 슬라이딩 윈도우

    Returns:
        features: (N, 1988) — 4 RX × 497 피처
    """
    N = windows.shape[0]
    rx_features = []

    for rx_idx in range(4):
        # RX별 슬라이싱: stride=4 인터리브 구조
        rx_win = windows[:, :, rx_idx::4]  # (N, 200, 166)

        # 1) 시간축 평균 (N, 166)
        mean_feat = rx_win.mean(axis=1)

        # 2) 서브캐리어 그래디언트 평균 (N, 165)
        #    mean_feat의 인접 서브캐리어 간 차분 → 주파수 도메인 공간 지문
        subcarrier_grad = np.diff(mean_feat, axis=1)  # (N, 165)

        # 3) 시간축 그래디언트 평균 (N, 166)
        #    인접 프레임 간 차분의 시간 평균 → 프레임 간 변화량
        temporal_grad = np.diff(rx_win, axis=1).mean(axis=1)  # (N, 166)

        rx_feat = np.concatenate([mean_feat, subcarrier_grad, temporal_grad], axis=1)  # (N, 497)
        rx_features.append(rx_feat)

    return np.concatenate(rx_features, axis=1)  # (N, 1988)


def apply_zone_rx_weight(X_feat: np.ndarray, zone_id: int,
                         rx_high_w: float = RX_HIGH_W) -> np.ndarray:
    """
    해당 Zone의 RX 피처 블록에 강조 가중치를 곱한다.

    Args:
        X_feat  : (N, 1988) 피처 배열
        zone_id : 강조할 Zone 번호
        rx_high_w: 강조 배율 (기본 2.0)

    Returns:
        X_weighted: (N, 1988) — Zone RX 블록이 강조된 피처 배열
    """
    X_weighted = X_feat.copy()
    rx_code_idx = ZONE_RX_MAP[zone_id]
    start = rx_code_idx * FEATURES_PER_RX
    end   = start + FEATURES_PER_RX
    X_weighted[:, start:end] *= rx_high_w
    return X_weighted


# ─── 혼동행렬 저장 ────────────────────────────────────────────────────────────

def save_confusion_matrix(preds, labels, zone_id, acc):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=ACTION_NAMES,
                yticklabels=ACTION_NAMES, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Zone {zone_id} Expert (SVM) — Action CM (window-level)\n'
                 f'Acc: {acc:.4f}')
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, f'svm_zone_expert_action_{zone_id}_cm.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[CM saved] {out_path}")


# ─── 학습 루프 ────────────────────────────────────────────────────────────────

def train(zone_id: int = ZONE_ID):
    print(f"\n{'='*50}")
    print(f"Zone ID: {zone_id}  (Zone {zone_id} 전문가 SVM 학습)")

    # ── 데이터 로드 ──────────────────────────────────────────────────────────
    print("\n[1] Loading raw datasets...")
    train_raw = CSIRawDataset(TRAIN_SUBJECTS)
    test_raw  = CSIRawDataset([TEST_SUBJECT])

    # ── 정규화 ───────────────────────────────────────────────────────────────
    print("\n[2] Normalizing (train-fit)...")
    data_scaler = normalize_datasets(train_raw, test_raw)

    # data_scaler는 Zone 0 학습 시 1회만 저장 (모든 Zone이 동일한 스케일러 공유)
    if not os.path.exists(DATA_SCALER_PATH):
        joblib.dump(data_scaler, DATA_SCALER_PATH)
        print(f"  Data scaler saved → {DATA_SCALER_PATH}")
    else:
        print(f"  Data scaler already exists, skipping save.")

    # ── 슬라이딩 윈도우 ──────────────────────────────────────────────────────
    print(f"\n[3] Creating sliding windows (size={WINDOW_SIZE}, stride={STRIDE})...")
    train_win = SlidingWindowDataset(train_raw, WINDOW_SIZE, STRIDE)
    test_win  = SlidingWindowDataset(test_raw,  WINDOW_SIZE, STRIDE)

    train_win.oversample_zone(zone_id, factor=2)

    print(f"  Train windows : {len(train_win)} (Zone {zone_id} ×2 오버샘플링 적용)")
    print(f"  Test  windows : {len(test_win)} (전체 Zone, 필터 없음)")

    # ── 윈도우 배열 추출 ─────────────────────────────────────────────────────
    # SlidingWindowDataset.windows: (N, window_size, 664) numpy array
    X_train_win = train_win.windows   # (N_train, 200, 664)
    y_train     = np.array(train_win.action_labels)

    X_test_win  = test_win.windows    # (N_test, 200, 664)
    y_test      = np.array(test_win.action_labels)

    # ── 피처 추출 ─────────────────────────────────────────────────────────────
    print("\n[4] Extracting features...")
    X_train_feat = extract_features(X_train_win)  # (N_train, 1988)
    X_test_feat  = extract_features(X_test_win)   # (N_test,  1988)
    print(f"  Feature shape: {X_train_feat.shape}")

    # ── Zone RX 가중치 적용 ───────────────────────────────────────────────────
    X_train_feat = apply_zone_rx_weight(X_train_feat, zone_id)
    X_test_feat  = apply_zone_rx_weight(X_test_feat,  zone_id)

    # ── 피처 정규화 (SVM 전용 StandardScaler) ────────────────────────────────
    feature_scaler = StandardScaler()
    X_train_scaled = feature_scaler.fit_transform(X_train_feat)
    X_test_scaled  = feature_scaler.transform(X_test_feat)

    # ── SVM 학습 ─────────────────────────────────────────────────────────────
    print(f"\n[5] Training SVM (C={SVM_C}, gamma={SVM_GAMMA})...")
    svm = SVC(kernel='rbf', C=SVM_C, gamma=SVM_GAMMA)
    svm.fit(X_train_scaled, y_train)
    print("  SVM training complete.")

    # ── 평가 ─────────────────────────────────────────────────────────────────
    print("\n[6] Evaluating...")
    y_pred = svm.predict(X_test_scaled)
    acc = (y_pred == y_test).mean()

    print(f"\nTest Accuracy (Zone {zone_id} Expert SVM, window-level, 전체 테스트): {acc:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=ACTION_NAMES,
                                zero_division=0))

    # ── 저장 ─────────────────────────────────────────────────────────────────
    ckpt_path = os.path.join(WEIGHTS_DIR, f'zone_expert_action_{zone_id}.pkl')
    joblib.dump({'svm': svm, 'feature_scaler': feature_scaler, 'zone_id': zone_id},
                ckpt_path)
    print(f"\n[Saved] {ckpt_path}")

    save_confusion_matrix(y_pred.tolist(), y_test.tolist(), zone_id, acc)


if __name__ == '__main__':
    zones = range(4) if ZONE_ID is None else [ZONE_ID]
    for z in zones:
        train(z)
