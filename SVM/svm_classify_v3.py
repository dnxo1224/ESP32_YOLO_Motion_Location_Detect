"""
SVM Baseline — Wi-Fi CSI Zone & Activity Classification

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
버전 히스토리
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
v1 — 기본 SVM
  - 샘플 단위: 녹음 1개 → 특징 벡터 1개
  - 전처리: seq_id 기반 800-그리드 정렬 + outlier 제거 + 선형 보간
  - 특징: mean, std, min, max, median, IQR, FFT 저주파, gradient
          (2821-dim / RX → 11284-dim, Zone/Action 공통)
  - 모델: StandardScaler + SVC(C=10, gamma='scale') 고정
  - 결과: Zone 88.28% / Action 86.72%

v2 — 슬라이딩 윈도우 + 태스크별 특징 + GridSearchCV
  - 샘플 단위: 녹음 1개 → 52개 윈도우 (WINDOW_SIZE=30, WINDOW_STRIDE=15)
  - 특징: Zone/Action 태스크별 분리
      Zone   : mean + std + subcarrier gradient (위치 특화)
               497-dim / RX → 1988-dim
      Action : mean + std + FFT 저주파 에너지 (행동 특화)
               1992-dim / RX → 7968-dim
  - 모델: GridSearchCV(C=[0.1,1,10,100], gamma=['scale','auto',1e-3,1e-4])
  - 결과: Zone 93.77% / Action (CV 79.41%)

v3 — GPU 가속 (cuML) + 하이퍼파라미터 고정
  - cuML 설치 시 GPU(CUDA) SVC 자동 사용, 미설치 시 sklearn CPU fallback
  - Pipeline 제거 → StandardScaler(CPU) + cuSVC(GPU) 분리 구조
  - GridSearchCV 제거 → C=100, gamma=1e-4 고정
  - 학습/예측 단계별 진행바 추가
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

현재 파이프라인 (v3):
  1. seq_id 기반 800-그리드 정렬 (처음 50 시퀀스 제거, 패킷 손실 → NaN)
  2. 3-sigma outlier 제거 (채널별 spike → NaN 마킹)
  3. 선형 보간으로 NaN 채우기 (ffill/bfill fallback)
  4. 슬라이딩 윈도우 (WINDOW_SIZE=30, WINDOW_STRIDE=15)
  5. 태스크별 특징 추출:
       Zone   : mean + std + subcarrier gradient  (497-dim / RX → 1988-dim)
       Action : mean + std + FFT 저주파 에너지    (1992-dim / RX → 7968-dim)
  6. StandardScaler(CPU) + GridSearchCV(cuML/sklearn SVM RBF)

실험 구성:
  - test  : kjh, kms (2명 고정)
  - train : 나머지 11명

실행:
    python svm_classify.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
)

# GPU(cuML) 사용 가능 시 우선 사용, 없으면 sklearn CPU fallback
try:
    from cuml.svm import SVC
    USING_GPU = True
except ImportError:
    from sklearn.svm import SVC
    USING_GPU = False

warnings.filterwarnings('ignore')

# ── 상수 정의 ──────────────────────────────────────────────────────────────────

# Null subcarrier 인덱스 (LTF + HT-LTF, 26개 제거)
_NULL_IDX  = (list(range(0, 6)) + [32] +
              list(range(59, 66)) + list(range(123, 134)) + [191])
VALID_IDX    = [i for i in range(192) if i not in _NULL_IDX]
NUM_FEATURES = len(VALID_IDX)  # 166

ACTION_MAP = {'handsup': 0, 'sit': 1, 'stand': 2, 'walk': 3}
ZONE_MAP   = {
    1: 0,  2: 0,  5: 0,  6: 0,
    3: 1,  4: 1,  7: 1,  8: 1,
    9: 2, 10: 2, 13: 2, 14: 2,
   11: 3, 12: 3, 15: 3, 16: 3,
}

TEST_SUBJECTS = ['kjh', 'kms']

# ── 설정 ──────────────────────────────────────────────────────────────────────

RAW_DIR     = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'svm')

GRID_SIZE     = 800    # 정렬 타겟 길이
SKIP_HEAD     = 50     # 처음 제거할 시퀀스 수
NAN_THRESHOLD = 0.5    # NaN 비율이 이 값 초과 시 경고
MAX_SEQ       = 65536  # ESP32 seq_id 최대값 (uint16 기준)
OUTLIER_SIGMA = 3.0    # outlier 제거 임계값
K_LOW         = 10     # FFT 저주파 빈 수 (DC 제외 1~K_LOW번 빈)

WINDOW_SIZE   = 30     # 슬라이딩 윈도우 길이 (timestep)
WINDOW_STRIDE = 15     # 슬라이딩 윈도우 이동 간격
SVM_MAX_ITER  = -1    # SVM 최대 반복 횟수
SVM_C         = 10.0   # 고정 하이퍼파라미터
SVM_GAMMA     = 1e-3    # 고정 하이퍼파라미터

ACTION_NAMES = ['handsup', 'sit', 'stand', 'walk']
ZONE_NAMES   = ['Zone0', 'Zone1', 'Zone2', 'Zone3']


# ── Raw CSV 로딩 ───────────────────────────────────────────────────────────────

def load_raw_rx(filepath):
    """Raw CSV 1개 RX → amplitude (L_raw, 166) + seq_ids (L_raw,)"""
    df = pd.read_csv(filepath, header=None, low_memory=False)
    seq_ids = df.iloc[:, 2].values.astype(np.int64)

    n_rows = len(df)
    csi_raw = np.zeros((n_rows, 384), dtype=np.float32)
    csi_raw[:, 1:383] = df.iloc[:, 26:408].values.astype(np.float32)

    real = csi_raw[:, 0::2]
    imag = csi_raw[:, 1::2]
    amplitude = np.sqrt(real ** 2 + imag ** 2)[:, VALID_IDX]

    return amplitude, seq_ids


def load_all_samples(raw_dir):
    """모든 피험자의 Raw 데이터를 로드."""
    samples = []
    subjects = sorted([
        d for d in os.listdir(raw_dir)
        if os.path.isdir(os.path.join(raw_dir, d))
    ])

    for subj in tqdm(subjects, desc='Loading subjects'):
        subj_dir = os.path.join(raw_dir, subj)
        for action_name, action_idx in ACTION_MAP.items():
            for pos in range(1, 17):
                files = [
                    os.path.join(subj_dir, f'{subj}_{action_name}_{pos}_rx{rx}.csv')
                    for rx in range(1, 5)
                ]
                if not all(os.path.exists(f) for f in files):
                    continue

                amps, sids = [], []
                for f in files:
                    amp, sid = load_raw_rx(f)
                    amps.append(amp)
                    sids.append(sid)

                samples.append({
                    'amplitudes': amps,
                    'seq_ids':    sids,
                    'action':     action_idx,
                    'zone':       ZONE_MAP[pos],
                    'subject':    subj,
                })

    return samples


def split_by_subjects(samples, test_subjects):
    test_set = set(test_subjects)
    train = [s for s in samples if s['subject'] not in test_set]
    test  = [s for s in samples if s['subject'] in test_set]
    return train, test


# ── Step 1: seq_id 기반 그리드 정렬 ───────────────────────────────────────────

def seq_slot(sid, start_seq, max_seq=MAX_SEQ):
    """wrap-around를 고려한 슬롯 인덱스 계산."""
    d = int(sid - start_seq) % max_seq
    if d > max_seq // 2:
        return -1
    return d


def align_to_grid(amplitude, seq_ids, start_seq, grid_size=GRID_SIZE):
    """
    RX 1개 raw amplitude를 seq_id 기반 800-슬롯 그리드에 배치.

    Returns
    -------
    grid      : (grid_size, 166) float32, NaN 포함 가능
    nan_ratio : float
    """
    grid = np.full((grid_size, NUM_FEATURES), np.nan, dtype=np.float32)

    for i, sid in enumerate(seq_ids):
        slot = seq_slot(sid, start_seq)
        if 0 <= slot < grid_size:
            grid[slot] = amplitude[i]

    nan_ratio = np.isnan(grid).mean()
    return grid, nan_ratio


def compute_start_seq(seq_ids_list, skip_head=SKIP_HEAD):
    """
    4개 RX의 seq_ids 리스트로부터 공통 기준 seq_id 결정.
    각 RX에서 skip_head 번째 고유 seq_id를 찾고 4개 중 최댓값 사용.
    """
    start_seqs = []
    for seq_ids in seq_ids_list:
        unique_sorted = np.sort(np.unique(seq_ids))
        if len(unique_sorted) <= skip_head:
            start_seqs.append(int(unique_sorted[0]))
        else:
            start_seqs.append(int(unique_sorted[skip_head]))
    return max(start_seqs)


# ── Step 2: Outlier 제거 ──────────────────────────────────────────────────────

def remove_outliers(grid, sigma=OUTLIER_SIGMA):
    """채널별 sigma 초과 패킷을 NaN으로 교체."""
    mean = np.nanmean(grid, axis=0)
    std  = np.nanstd(grid, axis=0)
    std[std == 0] = 1.0
    spike = np.abs(grid - mean) > sigma * std
    cleaned = grid.copy()
    cleaned[spike] = np.nan
    return cleaned


# ── Step 3: 선형 보간 ─────────────────────────────────────────────────────────

def interpolate_grid(grid):
    """
    (grid_size, 166) NaN 포함 배열 → 보간 완료 배열.
    내부 NaN → 선형 보간, leading/trailing NaN → ffill/bfill, 전채널 NaN → 0
    """
    df = pd.DataFrame(grid)
    df = df.interpolate(method='linear', axis=0)
    df = df.ffill(axis=0).bfill(axis=0)
    df = df.fillna(0.0)
    return df.values.astype(np.float32)


# ── Step 4: 태스크별 특징 추출 ────────────────────────────────────────────────

def extract_zone_features_window(window):
    """
    (WINDOW_SIZE, 166) → Zone 특화 특징 벡터 (497-dim).

    mean     : (166,)  — 채널별 평균 (위치별 정적 채널 패턴)
    std      : (166,)  — 채널별 표준편차
    gradient : (165,)  — 인접 서브캐리어 간 진폭 차이 평균 (멀티패스 공간 지문)
    """
    mean = window.mean(axis=0)
    std  = window.std(axis=0)
    grad = np.abs(np.diff(window, axis=1)).mean(axis=0)  # (165,)
    return np.concatenate([mean, std, grad])              # (497,)


def extract_action_features_window(window):
    """
    (WINDOW_SIZE, 166) → Action 특화 특징 벡터 (1992-dim).

    mean          : (166,)        — 채널별 평균
    std           : (166,)        — 채널별 표준편차
    FFT 저주파 에너지: (K_LOW×166,)  — 느린 신체 움직임 Doppler 성분
    """
    mean = window.mean(axis=0)
    std  = window.std(axis=0)
    fft_power   = np.abs(np.fft.rfft(window, axis=0)) ** 2   # (WINDOW_SIZE//2+1, 166)
    fft_lowfreq = fft_power[1:K_LOW + 1, :].flatten()        # (K_LOW×166,)
    return np.concatenate([mean, std, fft_lowfreq])           # (332 + K_LOW×166,)


# ── Step 5: 슬라이딩 윈도우 + 데이터셋 빌드 ──────────────────────────────────

def build_windows_from_sample(sample):
    """
    녹음 1개 → 슬라이딩 윈도우 단위 특징 리스트.

    Returns
    -------
    zone_feats   : (W, 1988) float32   — 4 RX × 497
    action_feats : (W, 7968) float32   — 4 RX × 1992
    y_zone       : (W,) int32
    y_action     : (W,) int32
    """
    start_seq = compute_start_seq(sample['seq_ids'])

    # 4 RX 각각 전처리 → (GRID_SIZE, 166)
    grids = []
    for rx in range(4):
        grid, nan_ratio = align_to_grid(
            sample['amplitudes'][rx], sample['seq_ids'][rx], start_seq
        )
        if nan_ratio > NAN_THRESHOLD:
            warnings.warn(
                f"[{sample['subject']} action={sample['action']} "
                f"zone={sample['zone']} RX{rx+1}] NaN ratio={nan_ratio:.1%}"
            )
        grid = remove_outliers(grid)
        grid = interpolate_grid(grid)
        grids.append(grid)

    # 슬라이딩 윈도우
    zone_feats, action_feats = [], []
    for start in range(0, GRID_SIZE - WINDOW_SIZE + 1, WINDOW_STRIDE):
        rx_zone, rx_action = [], []
        for grid in grids:
            w = grid[start:start + WINDOW_SIZE]        # (WINDOW_SIZE, 166)
            rx_zone.append(extract_zone_features_window(w))
            rx_action.append(extract_action_features_window(w))
        zone_feats.append(np.concatenate(rx_zone))    # (1988,)
        action_feats.append(np.concatenate(rx_action))  # (7968,)

    W = len(zone_feats)
    return (
        np.array(zone_feats,   dtype=np.float32),
        np.array(action_feats, dtype=np.float32),
        np.full(W, sample['zone'],   dtype=np.int32),
        np.full(W, sample['action'], dtype=np.int32),
    )


def build_dataset(samples, desc='Extracting features'):
    """
    샘플 리스트 → 윈도우 단위 특징 행렬.

    Returns
    -------
    X_zone   : (N_win, 1988) float32
    X_action : (N_win, 7968) float32
    y_zone   : (N_win,) int32
    y_action : (N_win,) int32
    """
    Xz_list, Xa_list, yz_list, ya_list = [], [], [], []

    for s in tqdm(samples, desc=desc):
        xz, xa, yz, ya = build_windows_from_sample(s)
        Xz_list.append(xz)
        Xa_list.append(xa)
        yz_list.append(yz)
        ya_list.append(ya)

    return (
        np.vstack(Xz_list),
        np.vstack(Xa_list),
        np.concatenate(yz_list),
        np.concatenate(ya_list),
    )


# ── SVM 학습 & 평가 ────────────────────────────────────────────────────────────

def train_and_evaluate(X_train, y_train, X_test, y_test,
                       label_names, task_name):
    """SVM 학습 → 평가 → (accuracy, confusion_matrix) 반환."""
    print(f'\n  [{task_name}] Using {"GPU (cuML)" if USING_GPU else "CPU (sklearn)"}')
    print(f'  [{task_name}] C={SVM_C}, gamma={SVM_GAMMA}')

    # StandardScaler: CPU에서 fit → cuML/sklearn 모두 numpy 입력 허용
    print(f'  [{task_name}] Scaling features ...')
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # SVM 학습
    svm = SVC(kernel='rbf', C=SVM_C, gamma=SVM_GAMMA,
              decision_function_shape='ovr', max_iter=SVM_MAX_ITER,
              verbose=True)
    print(f'  [{task_name}] Training SVM  (train={X_train_sc.shape[0]} samples) ...')
    with tqdm(total=1, desc=f'  [{task_name}] SVM fit', bar_format='{l_bar}{bar}| [{elapsed}]') as pbar:
        svm.fit(X_train_sc, y_train)
        pbar.update(1)

    # 예측
    print(f'  [{task_name}] Predicting    (test={X_test_sc.shape[0]} samples) ...')
    with tqdm(total=1, desc=f'  [{task_name}] SVM predict', bar_format='{l_bar}{bar}| [{elapsed}]') as pbar:
        y_pred = svm.predict(X_test_sc)
        pbar.update(1)

    acc = accuracy_score(y_test, y_pred)
    print(f'  [{task_name}] Test accuracy : {acc*100:.2f}%')
    print(classification_report(y_test, y_pred,
                                target_names=label_names, zero_division=0))
    return acc, confusion_matrix(y_test, y_pred)


def save_confusion_matrix(cm, label_names, task_name, acc):
    """혼동 행렬 PNG 저장."""
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=label_names)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(f'{task_name}  (acc={acc*100:.1f}%)')
    plt.tight_layout()

    path = os.path.join(RESULTS_DIR, f'{task_name.lower()}_cm.png')
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f'  Confusion matrix saved → {path}')


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    n_windows = (GRID_SIZE - WINDOW_SIZE) // WINDOW_STRIDE + 1
    print(f'Window size={WINDOW_SIZE}, stride={WINDOW_STRIDE} → {n_windows} windows/recording')

    print('\nLoading raw CSI data ...')
    all_samples = load_all_samples(RAW_DIR)
    print(f'Total recordings: {len(all_samples)}')

    train_samples, test_samples = split_by_subjects(all_samples, TEST_SUBJECTS)
    print(f'Train: {len(train_samples)} recordings  |  Test: {len(test_samples)} recordings')
    print(f'Test subjects: {TEST_SUBJECTS}')

    X_tr_z, X_tr_a, yz_tr, ya_tr = build_dataset(
        train_samples, desc='Extracting features (train)'
    )
    X_te_z, X_te_a, yz_te, ya_te = build_dataset(
        test_samples, desc='Extracting features (test) '
    )

    print(f'\nZone   feature shape — train: {X_tr_z.shape}, test: {X_te_z.shape}')
    print(f'Action feature shape — train: {X_tr_a.shape}, test: {X_te_a.shape}')

    # Zone 분류
    print('\n' + '='*60)
    print('Zone Classification (GridSearchCV SVM RBF)')
    print('='*60)
    acc_zone, cm_zone = train_and_evaluate(
        X_tr_z, yz_tr, X_te_z, yz_te, ZONE_NAMES, 'Zone'
    )
    save_confusion_matrix(cm_zone, ZONE_NAMES, 'Zone', acc_zone)

    # Action 분류
    print('\n' + '='*60)
    print('Action Classification (GridSearchCV SVM RBF)')
    print('='*60)
    acc_action, cm_action = train_and_evaluate(
        X_tr_a, ya_tr, X_te_a, ya_te, ACTION_NAMES, 'Action'
    )
    save_confusion_matrix(cm_action, ACTION_NAMES, 'Action', acc_action)

    # 요약 CSV
    summary_path = os.path.join(RESULTS_DIR, 'accuracy_summary.csv')
    with open(summary_path, 'w') as f:
        f.write('task,test_subjects,accuracy\n')
        f.write(f'zone,"{"+".join(TEST_SUBJECTS)}",{acc_zone*100:.4f}\n')
        f.write(f'action,"{"+".join(TEST_SUBJECTS)}",{acc_action*100:.4f}\n')
    print(f'\nSummary saved → {summary_path}')

    print('\n' + '='*60)
    print('RESULTS')
    print('='*60)
    print(f'  Zone   Accuracy : {acc_zone*100:.2f}%')
    print(f'  Action Accuracy : {acc_action*100:.2f}%')


if __name__ == '__main__':
    main()
