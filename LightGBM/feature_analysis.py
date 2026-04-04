"""
클래스별 평균 파워 스펙트럼 시각화 — 유효 주파수 대역 탐색

분석 내용:
  1. Action 클래스별 평균 파워 스펙트럼 (전체 / 저주파 확대)
  2. Zone 클래스별 평균 파워 스펙트럼 (전체 / 저주파 확대)
  3. 클래스 간 스펙트럼 차이 (Difference map) — 어느 대역이 분류에 유효한지 파악
  4. ANOVA F-statistic per bin — 통계적으로 유의미한 대역 강조

사전 실행:
    python preprocess.py

실행:
    python feature_analysis.py
"""

import os
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import f_oneway
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ── 상수 ──────────────────────────────────────────────────────────────────────

PROCESSED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'processed')
RESULTS_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'analysis')

ACTION_NAMES = ['handsup', 'sit', 'stand', 'walk']
ZONE_NAMES   = ['Zone0',   'Zone1', 'Zone2', 'Zone3']
COLORS       = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']

GRID_SIZE    = 800   # 시간축 길이
N_FREQS      = GRID_SIZE // 2 + 1   # rfft 결과 빈 수 (401)
ZOOM_MAX_BIN = 50    # 저주파 확대 범위


# ── 데이터 로드 ────────────────────────────────────────────────────────────────

def load_processed(processed_dir):
    samples = []
    npz_files = sorted([f for f in os.listdir(processed_dir) if f.endswith('.npz')])
    if not npz_files:
        raise FileNotFoundError(f'No NPZ files in {processed_dir}. Run preprocess.py first.')
    for fname in tqdm(npz_files, desc='Loading'):
        data = np.load(os.path.join(processed_dir, fname), allow_pickle=True)
        samples.append({
            'grids':   data['grids'],
            'action':  int(data['action']),
            'zone':    int(data['zone']),
        })
    return samples


# ── 파워 스펙트럼 계산 ─────────────────────────────────────────────────────────

def compute_mean_power_spectrum(samples):
    """
    샘플별 평균 파워 스펙트럼 계산.

    각 샘플: grids (4, 800, 166)
      → 4 RX × 166 채널 각각 rfft(axis=time)
      → 파워 스펙트럼 평균 → (401,) 대표 스펙트럼

    Returns
    -------
    spectra  : (N, 401) float32  — 샘플별 평균 파워 스펙트럼
    y_action : (N,) int
    y_zone   : (N,) int
    """
    spectra, y_action, y_zone = [], [], []

    for s in tqdm(samples, desc='Computing spectra'):
        grids = s['grids']   # (4, 800, 166)

        # 4 RX × 166 채널 모두 rfft → 파워 → 평균
        fft_power = np.abs(np.fft.rfft(grids, axis=1)) ** 2  # (4, 401, 166)
        mean_spec  = fft_power.mean(axis=(0, 2))               # (401,)

        spectra.append(mean_spec)
        y_action.append(s['action'])
        y_zone.append(s['zone'])

    return (np.array(spectra, dtype=np.float32),
            np.array(y_action, dtype=np.int32),
            np.array(y_zone,   dtype=np.int32))


# ── 플롯 함수 ─────────────────────────────────────────────────────────────────

def plot_class_spectra(spectra, labels, class_names, task_name, zoom_max=ZOOM_MAX_BIN):
    """
    클래스별 평균 파워 스펙트럼 — 전체 & 저주파 확대 subplot.
    """
    bins = np.arange(N_FREQS)

    # 클래스별 평균 스펙트럼
    class_means = []
    for c in range(len(class_names)):
        idx = np.where(labels == c)[0]
        class_means.append(spectra[idx].mean(axis=0))  # (401,)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'{task_name} — 클래스별 평균 파워 스펙트럼', fontsize=13)

    for ax, (x_start, x_end, subtitle) in zip(
        axes,
        [(0, N_FREQS, '전체 주파수 대역'),
         (1, zoom_max, f'저주파 확대 (bin 1~{zoom_max-1})')],
    ):
        for c, (mean_spec, color, name) in enumerate(
            zip(class_means, COLORS, class_names)
        ):
            x = bins[x_start:x_end]
            y = mean_spec[x_start:x_end]
            ax.plot(x, y, color=color, label=name, linewidth=1.6, alpha=0.85)

        ax.set_xlabel('Frequency bin')
        ax.set_ylabel('Mean Power')
        ax.set_title(subtitle)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f'{task_name.lower()}_spectrum.png')
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f'  Saved → {path}')


def plot_difference_map(spectra, labels, class_names, task_name, zoom_max=ZOOM_MAX_BIN):
    """
    클래스 간 스펙트럼 차이 히트맵 — 어느 bin에서 클래스 간 차이가 큰지 시각화.
    행: 클래스 쌍 (A - B), 열: 주파수 빈 (저주파 범위)
    """
    n_classes = len(class_names)
    bins = np.arange(1, zoom_max)   # DC 제외

    class_means = []
    for c in range(n_classes):
        idx = np.where(labels == c)[0]
        class_means.append(spectra[idx, 1:zoom_max].mean(axis=0))   # (zoom_max-1,)

    # 모든 클래스 쌍의 차이
    pairs, diff_matrix = [], []
    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            pairs.append(f'{class_names[i]} − {class_names[j]}')
            diff = class_means[i] - class_means[j]
            diff_matrix.append(diff)

    diff_matrix = np.array(diff_matrix)  # (n_pairs, zoom_max-1)

    # 정규화 (행별 최대 절댓값 기준)
    row_max = np.abs(diff_matrix).max(axis=1, keepdims=True) + 1e-8
    diff_norm = diff_matrix / row_max

    fig, ax = plt.subplots(figsize=(14, 4))
    im = ax.imshow(diff_norm, aspect='auto', cmap='RdBu_r',
                   vmin=-1, vmax=1,
                   extent=[bins[0], bins[-1], len(pairs) - 0.5, -0.5])
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels(pairs, fontsize=9)
    ax.set_xlabel('Frequency bin')
    ax.set_title(f'{task_name} — 클래스 간 스펙트럼 차이 (정규화, 빨강=양/파랑=음)')
    plt.colorbar(im, ax=ax, fraction=0.02)
    plt.tight_layout()

    path = os.path.join(RESULTS_DIR, f'{task_name.lower()}_diff_map.png')
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f'  Saved → {path}')


def plot_anova_fstat(spectra, labels, class_names, task_name, zoom_max=ZOOM_MAX_BIN):
    """
    ANOVA F-statistic per bin — 높을수록 해당 bin이 클래스 구분에 유효.
    """
    bins = np.arange(1, zoom_max)
    f_stats = []

    for b in bins:
        groups = [spectra[labels == c, b] for c in range(len(class_names))]
        f_val, _ = f_oneway(*groups)
        f_stats.append(f_val if np.isfinite(f_val) else 0.0)

    f_stats = np.array(f_stats)

    # 상위 5개 bin 강조
    top5_idx = np.argsort(f_stats)[-5:][::-1]

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.bar(bins, f_stats, color='#4C72B0', alpha=0.7, width=0.8)
    for idx in top5_idx:
        ax.bar(bins[idx], f_stats[idx], color='#C44E52', alpha=0.9, width=0.8)
        ax.text(bins[idx], f_stats[idx] + f_stats.max() * 0.01,
                f'bin {bins[idx]}', ha='center', va='bottom', fontsize=7.5,
                color='#C44E52')

    ax.set_xlabel('Frequency bin')
    ax.set_ylabel('F-statistic')
    ax.set_title(f'{task_name} — ANOVA F-statistic per bin '
                 f'(빨강: top-5 유효 대역, bin 1~{zoom_max-1})')
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()

    path = os.path.join(RESULTS_DIR, f'{task_name.lower()}_anova.png')
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f'  Saved → {path}')

    # 콘솔 출력
    print(f'\n  [{task_name}] Top-10 유효 주파수 빈 (F-statistic 기준):')
    top10 = np.argsort(f_stats)[-10:][::-1]
    for rank, i in enumerate(top10, 1):
        print(f'    {rank:2d}. bin {bins[i]:3d}  F={f_stats[i]:.1f}')


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print('Loading preprocessed data ...')
    samples = load_processed(PROCESSED_DIR)
    print(f'Total samples: {len(samples)}')

    print('\nComputing power spectra ...')
    spectra, y_action, y_zone = compute_mean_power_spectrum(samples)
    print(f'Spectra shape: {spectra.shape}')

    # ── Action 분석 ──────────────────────────────────────────────────────────
    print('\n[Action]')
    plot_class_spectra(spectra, y_action, ACTION_NAMES, 'Action')
    plot_difference_map(spectra, y_action, ACTION_NAMES, 'Action')
    plot_anova_fstat(spectra, y_action, ACTION_NAMES, 'Action')

    # ── Zone 분석 ────────────────────────────────────────────────────────────
    print('\n[Zone]')
    plot_class_spectra(spectra, y_zone, ZONE_NAMES, 'Zone')
    plot_difference_map(spectra, y_zone, ZONE_NAMES, 'Zone')
    plot_anova_fstat(spectra, y_zone, ZONE_NAMES, 'Zone')

    print(f'\n모든 분석 결과 → {RESULTS_DIR}')
    print('\n생성된 파일:')
    for f in sorted(os.listdir(RESULTS_DIR)):
        print(f'  {f}')


if __name__ == '__main__':
    main()
