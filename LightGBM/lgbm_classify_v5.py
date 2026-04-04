"""
LightGBM — Wi-Fi CSI Zone & Activity Classification

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
버전 히스토리
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
v1 — LightGBM 기본
  - 전처리 캐시(NPZ) 로드 구조 (preprocess.py 1회 실행 후 재사용)
  - 슬라이딩 윈도우 (WINDOW_SIZE=30, WINDOW_STRIDE=15) → 52 windows/recording
  - 태스크별 특징:
      Zone   : mean+std+min+max+median+IQR+FFT+gradient  (2821-dim/RX → 11284-dim)
      Action : mean+std+min+max+median+IQR+FFT+gradient  (2821-dim/RX → 11284-dim)
  - LightGBM multiclass, GPU 지원 (device='gpu'), CPU fallback

v2 — ANOVA 기반 유효 주파수 빈 선택 + Zone Cross-correlation
  - Action: ANOVA Top-10 빈 [5,6,7,32,33,35,36,37,40,44] 사용
            (저주파 클러스터 bins 5~7 + 중주파 클러스터 bins 32~44)
  - Zone  : ANOVA Top-5 빈 [1,2,3,26,49] 사용
            + RX 쌍 간 Cross-correlation (peak_val + peak_lag) 추가
  - 특징 차원:
      Zone   : (1991-dim/RX × 4RX) + cross-corr 1992 = 9956-dim
      Action : 2821-dim/RX × 4RX = 11284-dim (빈 구성만 변경)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

실험 구성:
  - test  : kjh, kms (2명 고정)
  - train : 나머지 11명

사전 실행:
    python preprocess.py   # 최초 1회

실행:
    python lgbm_classify.py
"""

import os
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

import lightgbm as lgb
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
)

warnings.filterwarnings('ignore')

# ── 상수 ──────────────────────────────────────────────────────────────────────

NUM_FEATURES  = 166
TEST_SUBJECTS = ['kjh', 'kms']

PROCESSED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'processed')
RESULTS_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'lgbm')

WINDOW_SIZE   = 800
WINDOW_STRIDE = 100
K_LOW         = 10

# ANOVA F-statistic 기반 선택된 유효 주파수 빈
ACTION_FFT_BINS = [5, 6, 7, 32, 33, 35, 36, 37, 40, 44]   # Top-10 (저주파+중주파 클러스터)
ZONE_FFT_BINS   = [1, 2, 3, 26, 49]                         # Top-5

# Cross-correlation RX 쌍 (6 pairs)
XCORR_PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

ACTION_NAMES = ['handsup', 'sit', 'stand', 'walk']
ZONE_NAMES   = ['Zone0', 'Zone1', 'Zone2', 'Zone3']

LGBM_PARAMS_BASE = {
    'objective':     'multiclass',
    'num_class':     4,
    'metric':        'multi_logloss',
    'n_estimators':  1000,
    'learning_rate': 0.05,
    'num_leaves':    63,
    'verbose':       -1,
}

# ── 특징 차원 (참조용) ─────────────────────────────────────────────────────────
#
# Zone per-RX  : 6×166 + 5×166 + 165 = 1991-dim
# Zone 4RX     : 1991 × 4 = 7964-dim
# Zone crosscorr: 6 pairs × 2 stats × 166 = 1992-dim
# Zone total   : 7964 + 1992 = 9956-dim
#
# Action per-RX: 6×166 + 10×166 + 165 = 2821-dim
# Action total : 2821 × 4 = 11284-dim


# ── 전처리 캐시 로드 ──────────────────────────────────────────────────────────

def load_processed(processed_dir):
    """processed/ 폴더의 NPZ 파일 전체 로드."""
    samples = []
    npz_files = sorted([f for f in os.listdir(processed_dir) if f.endswith('.npz')])
    if not npz_files:
        raise FileNotFoundError(
            f'No NPZ files found in {processed_dir}. Run preprocess.py first.'
        )
    for fname in tqdm(npz_files, desc='Loading processed data'):
        data = np.load(os.path.join(processed_dir, fname), allow_pickle=True)
        samples.append({
            'grids':   data['grids'],
            'action':  int(data['action']),
            'zone':    int(data['zone']),
            'subject': str(data['subject']),
        })
    return samples


def split_by_subjects(samples, test_subjects):
    test_set = set(test_subjects)
    train = [s for s in samples if s['subject'] not in test_set]
    test  = [s for s in samples if s['subject'] in test_set]
    return train, test


# ── 특징 추출 ─────────────────────────────────────────────────────────────────

def _base_stats(window):
    """공통 통계 특징: mean+std+min+max+median+IQR → (996,)"""
    mean   = window.mean(axis=0)
    std    = window.std(axis=0)
    mn     = window.min(axis=0)
    mx     = window.max(axis=0)
    median = np.median(window, axis=0)
    iqr    = np.percentile(window, 75, axis=0) - np.percentile(window, 25, axis=0)
    return np.concatenate([mean, std, mn, mx, median, iqr])   # (996,)


def _fft_bins(window, bins):
    """지정된 빈의 FFT 파워 → (len(bins)×166,)"""
    fft_power = np.abs(np.fft.rfft(window, axis=0)) ** 2   # (WINDOW_SIZE//2+1, 166)
    selected  = fft_power[bins, :]                          # (len(bins), 166)
    return selected.flatten()


def _gradient(window):
    """서브캐리어 축 gradient → (165,)"""
    return np.abs(np.diff(window, axis=1)).mean(axis=0)


def extract_zone_features_window(window):
    """
    (WINDOW_SIZE, 166) → 1991-dim  (Zone 단일 RX 특징)

      기본 통계  : 6×166 = 996-dim
      FFT top-5  : 5×166 = 830-dim  (ANOVA 유효 빈: 1,2,3,26,49)
      gradient   :     165-dim
    """
    return np.concatenate([
        _base_stats(window),
        _fft_bins(window, ZONE_FFT_BINS),
        _gradient(window),
    ])   # (1991,)


def extract_action_features_window(window):
    """
    (WINDOW_SIZE, 166) → 2821-dim  (Action 단일 RX 특징)

      기본 통계  : 6×166 =  996-dim
      FFT top-10 : 10×166 = 1660-dim  (ANOVA 유효 빈: 5,6,7,32,33,35,36,37,40,44)
      gradient   :      165-dim
    """
    return np.concatenate([
        _base_stats(window),
        _fft_bins(window, ACTION_FFT_BINS),
        _gradient(window),
    ])   # (2821,)


def extract_crosscorr_features(windows):
    """
    4개 RX 윈도우 간 Cross-correlation 특징 (Zone 특화).

    Parameters
    ----------
    windows : list of 4 arrays, each (WINDOW_SIZE, 166)

    Returns
    -------
    feats : (1992,)  = 6 pairs × 2 stats (peak_val, peak_lag) × 166 channels

    FFT 기반 벡터화 cross-correlation 사용 (채널 루프 없음).
    """
    n      = WINDOW_SIZE
    fft_n  = 2 * n - 1         # full cross-correlation 길이
    lags   = np.arange(-(n - 1), n, dtype=np.float32)   # (fft_n,)

    feats = []
    for i, j in XCORR_PAIRS:
        wi = windows[i] - windows[i].mean(axis=0)   # zero-mean (WINDOW_SIZE, 166)
        wj = windows[j] - windows[j].mean(axis=0)

        # FFT 기반 cross-correlation: (fft_n, 166)
        Wi    = np.fft.rfft(wi, n=fft_n, axis=0)
        Wj    = np.fft.rfft(wj, n=fft_n, axis=0)
        xcorr = np.fft.irfft(Wi * np.conj(Wj), n=fft_n, axis=0)  # (fft_n, 166)

        # 채널별 피크 인덱스 → peak_val, peak_lag
        peak_idx = np.abs(xcorr).argmax(axis=0)            # (166,)
        peak_val = xcorr[peak_idx, np.arange(NUM_FEATURES)]  # (166,)
        peak_lag = lags[peak_idx]                            # (166,)

        feats.append(peak_val)
        feats.append(peak_lag)

    return np.concatenate(feats)   # (1992,)


def build_windows_from_grids(sample):
    """
    grids (4, 800, 166) → 슬라이딩 윈도우 특징.

    Returns
    -------
    zone_feats   : (W, 9956)  float32
    action_feats : (W, 11284) float32
    y_zone       : (W,) int32
    y_action     : (W,) int32
    """
    grids = sample['grids']   # (4, 800, 166)

    zone_feats, action_feats = [], []
    for start in range(0, 800 - WINDOW_SIZE + 1, WINDOW_STRIDE):
        # 4 RX 윈도우 추출
        windows = [grids[rx, start:start + WINDOW_SIZE] for rx in range(4)]

        # Zone: per-RX 특징 + cross-correlation
        rx_zone = [extract_zone_features_window(w) for w in windows]
        crosscorr = extract_crosscorr_features(windows)
        zone_feats.append(np.concatenate(rx_zone + [crosscorr]))   # (9956,)

        # Action: per-RX 특징만
        rx_action = [extract_action_features_window(w) for w in windows]
        action_feats.append(np.concatenate(rx_action))              # (11284,)

    W = len(zone_feats)
    return (
        np.array(zone_feats,   dtype=np.float32),
        np.array(action_feats, dtype=np.float32),
        np.full(W, sample['zone'],   dtype=np.int32),
        np.full(W, sample['action'], dtype=np.int32),
    )


def build_dataset(samples, desc='Extracting features'):
    """
    Returns
    -------
    X_zone   : (N_win, 9956)
    X_action : (N_win, 11284)
    y_zone   : (N_win,)
    y_action : (N_win,)
    """
    Xz, Xa, yz, ya = [], [], [], []
    for s in tqdm(samples, desc=desc):
        xz, xa, lz, la = build_windows_from_grids(s)
        Xz.append(xz); Xa.append(xa)
        yz.append(lz); ya.append(la)
    return (
        np.vstack(Xz), np.vstack(Xa),
        np.concatenate(yz), np.concatenate(ya),
    )


# ── LightGBM 학습 & 평가 ──────────────────────────────────────────────────────

def try_gpu():
    try:
        probe = lgb.LGBMClassifier(n_estimators=1, device='gpu', verbose=-1)
        probe.fit(np.zeros((10, 4)), np.arange(4).repeat(3)[:10])
        return True
    except Exception:
        return False


def train_and_evaluate(X_train, y_train, X_test, y_test,
                       label_names, task_name, using_gpu):
    """LightGBM 학습 → 평가 → (accuracy, confusion_matrix) 반환."""
    device = 'gpu' if using_gpu else 'cpu'
    print(f'\n  [{task_name}] device={device}  '
          f'train={X_train.shape[0]:,}  test={X_test.shape[0]:,}')

    clf = lgb.LGBMClassifier(**{**LGBM_PARAMS_BASE, 'device': device})
    clf.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=100),
        ],
    )

    y_pred = clf.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)

    print(f'  [{task_name}] Best iteration : {clf.best_iteration_}')
    print(f'  [{task_name}] Test accuracy  : {acc * 100:.2f}%')
    print(classification_report(y_test, y_pred,
                                target_names=label_names, zero_division=0))

    plot_feature_importance(clf, task_name)
    return acc, confusion_matrix(y_test, y_pred)


def plot_feature_importance(clf, task_name):
    """
    그룹별 Feature Importance 집계 → 막대그래프 저장.

    Zone   (9956-dim) : per-RX 1991 × 4 + cross-corr 1992
    Action (11284-dim): per-RX 2821 × 4
    """
    importances = clf.feature_importances_

    if task_name == 'Zone':
        feat_per_rx = 1991
        per_rx_groups = [
            ('mean',     0,    166),
            ('std',      166,  332),
            ('min',      332,  498),
            ('max',      498,  664),
            ('median',   664,  830),
            ('IQR',      830,  996),
            ('FFT',      996,  1826),   # 5×166=830
            ('gradient', 1826, 1991),   # 165
        ]
        group_sums = {}
        for name, s, e in per_rx_groups:
            total = sum(importances[rx * feat_per_rx + s:
                                    rx * feat_per_rx + e].sum()
                        for rx in range(4))
            group_sums[name] = total
        # Cross-correlation (마지막 1992)
        group_sums['cross-corr'] = importances[7964:9956].sum()

    else:  # Action
        feat_per_rx = 2821
        per_rx_groups = [
            ('mean',     0,    166),
            ('std',      166,  332),
            ('min',      332,  498),
            ('max',      498,  664),
            ('median',   664,  830),
            ('IQR',      830,  996),
            ('FFT',      996,  2656),   # 10×166=1660
            ('gradient', 2656, 2821),   # 165
        ]
        group_sums = {}
        for name, s, e in per_rx_groups:
            total = sum(importances[rx * feat_per_rx + s:
                                    rx * feat_per_rx + e].sum()
                        for rx in range(4))
            group_sums[name] = total

    # 비율 변환
    total_imp  = sum(group_sums.values()) + 1e-8
    group_pcts = {k: v / total_imp * 100 for k, v in group_sums.items()}

    # 막대그래프
    names  = list(group_pcts.keys())
    values = list(group_pcts.values())
    colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52',
              '#8172B2', '#937860', '#DA8BC3', '#8C8C8C', '#B8C0CC']

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(names, values, color=colors[:len(names)],
                  edgecolor='white', linewidth=0.8)
    ax.set_title(f'Feature Importance — {task_name}', fontsize=13, pad=12)
    ax.set_ylabel('Importance (%)')
    ax.set_ylim(0, max(values) * 1.18)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()

    path = os.path.join(RESULTS_DIR, f'{task_name.lower()}_feature_importance.png')
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f'  Feature importance saved → {path}')

    print(f'\n  [{task_name}] Feature importance:')
    for name, pct in sorted(group_pcts.items(), key=lambda x: -x[1]):
        print(f'    {name:<12} {"█" * int(pct / 2)}  {pct:.1f}%')


def save_confusion_matrix(cm, label_names, task_name, acc):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(f'{task_name}  (acc={acc * 100:.1f}%)')
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f'{task_name.lower()}_cm.png')
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f'  Confusion matrix saved → {path}')


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    n_windows = (800 - WINDOW_SIZE) // WINDOW_STRIDE + 1
    print(f'Window size={WINDOW_SIZE}, stride={WINDOW_STRIDE} '
          f'→ {n_windows} windows/recording')
    print(f'Zone   FFT bins : {ZONE_FFT_BINS}')
    print(f'Action FFT bins : {ACTION_FFT_BINS}')

    print('\nLoading preprocessed data ...')
    all_samples = load_processed(PROCESSED_DIR)
    print(f'Total recordings: {len(all_samples)}')

    train_samples, test_samples = split_by_subjects(all_samples, TEST_SUBJECTS)
    print(f'Train: {len(train_samples)}  |  Test: {len(test_samples)}  '
          f'(test subjects: {TEST_SUBJECTS})')

    X_tr_z, X_tr_a, yz_tr, ya_tr = build_dataset(
        train_samples, desc='Extracting features (train)'
    )
    X_te_z, X_te_a, yz_te, ya_te = build_dataset(
        test_samples, desc='Extracting features (test) '
    )
    print(f'\nZone   feature — train: {X_tr_z.shape}, test: {X_te_z.shape}')
    print(f'Action feature — train: {X_tr_a.shape}, test: {X_te_a.shape}')

    print('\nChecking GPU availability ...')
    using_gpu = try_gpu()
    print(f'  → {"GPU (LightGBM CUDA)" if using_gpu else "CPU"}')

    # Zone 분류
    print('\n' + '=' * 60)
    print('Zone Classification (LightGBM)')
    print('=' * 60)
    acc_zone, cm_zone = train_and_evaluate(
        X_tr_z, yz_tr, X_te_z, yz_te, ZONE_NAMES, 'Zone', using_gpu
    )
    save_confusion_matrix(cm_zone, ZONE_NAMES, 'Zone', acc_zone)

    # Action 분류
    print('\n' + '=' * 60)
    print('Action Classification (LightGBM)')
    print('=' * 60)
    acc_action, cm_action = train_and_evaluate(
        X_tr_a, ya_tr, X_te_a, ya_te, ACTION_NAMES, 'Action', using_gpu
    )
    save_confusion_matrix(cm_action, ACTION_NAMES, 'Action', acc_action)

    # 요약 CSV
    summary_path = os.path.join(RESULTS_DIR, 'accuracy_summary.csv')
    with open(summary_path, 'w') as f:
        f.write('task,test_subjects,accuracy\n')
        f.write(f'zone,"{"+".join(TEST_SUBJECTS)}",{acc_zone * 100:.4f}\n')
        f.write(f'action,"{"+".join(TEST_SUBJECTS)}",{acc_action * 100:.4f}\n')
    print(f'\nSummary saved → {summary_path}')

    print('\n' + '=' * 60)
    print('RESULTS')
    print('=' * 60)
    print(f'  Zone   Accuracy : {acc_zone * 100:.2f}%')
    print(f'  Action Accuracy : {acc_action * 100:.2f}%')


if __name__ == '__main__':
    main()
