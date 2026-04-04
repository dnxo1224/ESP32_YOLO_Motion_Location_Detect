"""
LightGBM — Wi-Fi CSI Zone & Activity Classification

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
버전 히스토리
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
v1 — LightGBM 기본
  - 전처리 캐시(NPZ) 로드 구조 (preprocess.py 1회 실행 후 재사용)
  - 슬라이딩 윈도우 (WINDOW_SIZE=30, WINDOW_STRIDE=15) → 52 windows/recording
  - 태스크별 특징:
      Zone   : mean + std + subcarrier gradient  (497-dim / RX → 1988-dim)
      Action : mean + std + DFS centroid + DFS spread  (664-dim / RX → 2656-dim)
  - LightGBM multiclass, GPU 지원 (device='gpu'), CPU fallback
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
ACTION_MAP    = {'handsup': 0, 'sit': 1, 'stand': 2, 'walk': 3}
TEST_SUBJECTS = ['kms']

PROCESSED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'processed')
RESULTS_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'lgbm')

WINDOW_SIZE   = 200
WINDOW_STRIDE = 30
K_LOW         = 10

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


# ── 전처리 캐시 로드 ──────────────────────────────────────────────────────────

def load_processed(processed_dir):
    """
    processed/ 폴더의 NPZ 파일 전체 로드.

    Returns
    -------
    samples : list of dict
        {
          'grids'   : (4, 800, 166) float32,
          'action'  : int,
          'zone'    : int,
          'subject' : str,
        }
    """
    samples = []
    npz_files = sorted([
        f for f in os.listdir(processed_dir) if f.endswith('.npz')
    ])
    if not npz_files:
        raise FileNotFoundError(
            f'No NPZ files found in {processed_dir}. '
            'Run preprocess.py first.'
        )
    for fname in tqdm(npz_files, desc='Loading processed data'):
        data = np.load(os.path.join(processed_dir, fname), allow_pickle=True)
        samples.append({
            'grids':   data['grids'],             # (4, 800, 166)
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

def extract_zone_features_window(window):
    """(WINDOW_SIZE, 166) → 497-dim (zone 특화: mean + std + gradient)"""
    mean = window.mean(axis=0)
    std  = window.std(axis=0)
    # grad = np.abs(np.diff(window, axis=1)).mean(axis=0)   # (165,)
    return np.concatenate([mean, std])


def extract_action_features_window(window):
    """
    (WINDOW_SIZE, 166) → 664-dim (action 특화: mean + std + DFS centroid + DFS spread)

    DFS (Doppler Frequency Shift):
      - centroid : 주파수 무게중심 = Σ(f × |F(f)|²) / Σ|F(f)|²  → 움직임 속도 반영
      - spread   : 주파수 분포의 표준편차                          → 움직임 복잡도 반영
    """
    mean = window.mean(axis=0)   # (166,)
    std  = window.std(axis=0)    # (166,)

    # FFT (DC 제외, 양의 주파수만)
    fft_power = np.abs(np.fft.rfft(window, axis=0)) ** 2  # (WINDOW_SIZE//2+1, 166)
    freqs     = np.arange(fft_power.shape[0], dtype=np.float32)  # 빈 인덱스 [0,1,2,...]
    power_dc  = fft_power[1:, :]   # DC(0번 빈) 제외 → (WINDOW_SIZE//2, 166)
    freqs_dc  = freqs[1:]          # (WINDOW_SIZE//2,)

    total_power = power_dc.sum(axis=0) + 1e-8  # (166,) — 0-division 방지

    # DFS centroid: 에너지 무게중심 주파수
    centroid = (freqs_dc[:, None] * power_dc).sum(axis=0) / total_power  # (166,)

    # DFS spread: 주파수 분포의 표준편차
    spread = np.sqrt(
        (((freqs_dc[:, None] - centroid[None, :]) ** 2) * power_dc
         ).sum(axis=0) / total_power
    )  # (166,)

    return np.concatenate([mean, std, centroid, spread])  # (664,)


def build_windows_from_grids(sample):
    """
    전처리 완료된 grids (4, 800, 166) → 슬라이딩 윈도우 특징.

    Returns
    -------
    zone_feats   : (W, 1988) float32
    action_feats : (W, 2656) float32
    y_zone       : (W,) int32
    y_action     : (W,) int32
    """
    grids = sample['grids']   # (4, 800, 166)

    zone_feats, action_feats = [], []
    for start in range(0, WINDOW_SIZE * ((800 - WINDOW_SIZE) // WINDOW_STRIDE + 1),
                       WINDOW_STRIDE):
        if start + WINDOW_SIZE > 800:
            break
        rx_zone, rx_action = [], []
        for rx in range(4):
            w = grids[rx, start:start + WINDOW_SIZE]     # (WINDOW_SIZE, 166)
            rx_zone.append(extract_zone_features_window(w))
            rx_action.append(extract_action_features_window(w))
        zone_feats.append(np.concatenate(rx_zone))       # (1988,)
        action_feats.append(np.concatenate(rx_action))   # (2656,)

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
    X_zone   : (N_win, 1988)
    X_action : (N_win, 2656)
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
    """GPU 사용 가능 여부 확인."""
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

    params = {**LGBM_PARAMS_BASE, 'device': device}
    clf = lgb.LGBMClassifier(**params)

    clf.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=100),
        ],
    )

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f'  [{task_name}] Best iteration : {clf.best_iteration_}')
    print(f'  [{task_name}] Test accuracy  : {acc * 100:.2f}%')
    print(classification_report(y_test, y_pred,
                                target_names=label_names, zero_division=0))
    return acc, confusion_matrix(y_test, y_pred)


def save_confusion_matrix(cm, label_names, task_name, acc):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=label_names)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
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
