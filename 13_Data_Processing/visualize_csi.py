"""
CSI 데이터 시각화 스크립트
data_interpolated_linear_872/ 데이터를 활용한 행동별/Zone별 패턴 분석
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# --- 설정 ---
BASE_DIR = os.path.join(os.path.dirname(__file__), '..', 'data_interpolated_linear_872')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

ACTIONS = ['handsup', 'sit', 'stand', 'walk']
ZONE_MAP = {
    1: 0, 2: 0, 5: 0, 6: 0,
    3: 1, 4: 1, 7: 1, 8: 1,
    9: 2, 10: 2, 13: 2, 14: 2,
    11: 3, 12: 3, 15: 3, 16: 3,
}
ZONE_POSITIONS = {0: [1, 2, 5, 6], 1: [3, 4, 7, 8], 2: [9, 10, 13, 14], 3: [11, 12, 15, 16]}
ZONE_REPR_POS = {0: 1, 1: 3, 2: 9, 3: 11}  # 각 zone 대표 위치
ALL_SUBJECTS = ['gyj', 'jhj', 'jkw', 'kjh', 'kmh', 'kms', 'kye', 'lsi', 'mhe', 'phr', 'stk', 'swt', 'ysj']


def load_sample(base_dir, subject, action, position):
    """4개 RX CSV를 로드하여 (872, 456) 배열로 반환"""
    rx_list = []
    for rx in range(1, 5):
        path = os.path.join(base_dir, f"{subject}_{action}_{position}_rx{rx}_872.csv")
        df = pd.read_csv(path, index_col=0)
        rx_list.append(df.values)  # (872, 114)
    return np.hstack(rx_list)  # (872, 456)


def plot_action_heatmaps(subject='gyj', position=1):
    """Plot 1: 행동별 CSI Amplitude Heatmap (고정 피험자/위치)"""
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    fig.suptitle(f'CSI Heatmap by Action (Subject: {subject}, Position: {position})', fontsize=14)

    for i, action in enumerate(ACTIONS):
        data = load_sample(BASE_DIR, subject, action, position)
        im = axes[i].imshow(data, aspect='auto', cmap='viridis', interpolation='nearest')
        axes[i].set_title(action)
        axes[i].set_xlabel('Subcarrier Index (114×4 RX)')
        axes[i].set_ylabel('Time Frame (872)')
        plt.colorbar(im, ax=axes[i], fraction=0.046)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'plot1_action_heatmaps.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_zone_heatmaps(subject='gyj', action='walk'):
    """Plot 2: Zone별 CSI Amplitude Heatmap (고정 피험자/행동)"""
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    fig.suptitle(f'CSI Heatmap by Zone (Subject: {subject}, Action: {action})', fontsize=14)

    for zone_id in range(4):
        pos = ZONE_REPR_POS[zone_id]
        data = load_sample(BASE_DIR, subject, action, pos)
        im = axes[zone_id].imshow(data, aspect='auto', cmap='viridis', interpolation='nearest')
        axes[zone_id].set_title(f'Zone {zone_id} (pos {pos})')
        axes[zone_id].set_xlabel('Subcarrier Index')
        axes[zone_id].set_ylabel('Time Frame')
        plt.colorbar(im, ax=axes[zone_id], fraction=0.046)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'plot2_zone_heatmaps.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_timeseries_comparison(subject='gyj', subcarrier_idx=50):
    """Plot 3: 시계열 비교 (특정 서브캐리어)"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # 행동별 비교 (고정 위치 1)
    for action in ACTIONS:
        data = load_sample(BASE_DIR, subject, action, 1)
        axes[0].plot(data[:, subcarrier_idx], label=action, alpha=0.8)
    axes[0].set_title(f'Action Comparison (pos=1, sub_idx={subcarrier_idx})')
    axes[0].set_xlabel('Time Frame')
    axes[0].set_ylabel('CSI Amplitude')
    axes[0].legend()

    # Zone별 비교 (고정 행동 walk)
    for zone_id, pos in ZONE_REPR_POS.items():
        data = load_sample(BASE_DIR, subject, 'walk', pos)
        axes[1].plot(data[:, subcarrier_idx], label=f'Zone {zone_id} (pos {pos})', alpha=0.8)
    axes[1].set_title(f'Zone Comparison (action=walk, sub_idx={subcarrier_idx})')
    axes[1].set_xlabel('Time Frame')
    axes[1].set_ylabel('CSI Amplitude')
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'plot3_timeseries_comparison.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_zone_mean_profiles():
    """Plot 4: Zone별 평균 진폭 프로파일 (전체 피험자/행동 평균)"""
    zone_profiles = {z: [] for z in range(4)}

    for subject in ALL_SUBJECTS:
        for action in ACTIONS:
            for pos in range(1, 17):
                zone = ZONE_MAP[pos]
                try:
                    data = load_sample(BASE_DIR, subject, action, pos)
                    mean_profile = data.mean(axis=0)  # (456,)
                    zone_profiles[zone].append(mean_profile)
                except FileNotFoundError:
                    continue

    fig, ax = plt.subplots(figsize=(14, 5))
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    for zone_id in range(4):
        profiles = np.array(zone_profiles[zone_id])
        mean = profiles.mean(axis=0)
        std = profiles.std(axis=0)
        ax.plot(mean, label=f'Zone {zone_id} (n={len(profiles)})', color=colors[zone_id], alpha=0.9)
        ax.fill_between(range(456), mean - std, mean + std, color=colors[zone_id], alpha=0.15)

    ax.set_title('Mean CSI Amplitude Profile by Zone (all subjects, all actions)')
    ax.set_xlabel('Feature Index (114 subcarriers × 4 RX)')
    ax.set_ylabel('Mean Amplitude')
    ax.legend()

    # RX 경계선 표시
    for rx_boundary in [114, 228, 342]:
        ax.axvline(x=rx_boundary, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'plot4_zone_mean_profiles.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_action_mean_profiles():
    """Plot 5 (보너스): 행동별 평균 진폭 프로파일"""
    action_profiles = {a: [] for a in ACTIONS}

    for subject in ALL_SUBJECTS:
        for action in ACTIONS:
            for pos in range(1, 17):
                try:
                    data = load_sample(BASE_DIR, subject, action, pos)
                    mean_profile = data.mean(axis=0)
                    action_profiles[action].append(mean_profile)
                except FileNotFoundError:
                    continue

    fig, ax = plt.subplots(figsize=(14, 5))
    for action in ACTIONS:
        profiles = np.array(action_profiles[action])
        mean = profiles.mean(axis=0)
        std = profiles.std(axis=0)
        ax.plot(mean, label=f'{action} (n={len(profiles)})', alpha=0.9)
        ax.fill_between(range(456), mean - std, mean + std, alpha=0.15)

    ax.set_title('Mean CSI Amplitude Profile by Action (all subjects, all positions)')
    ax.set_xlabel('Feature Index (114 subcarriers × 4 RX)')
    ax.set_ylabel('Mean Amplitude')
    ax.legend()

    for rx_boundary in [114, 228, 342]:
        ax.axvline(x=rx_boundary, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'plot5_action_mean_profiles.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


if __name__ == '__main__':
    print("=== CSI Data Visualization ===")
    print(f"Data dir: {BASE_DIR}")
    print(f"Results dir: {RESULTS_DIR}")

    plot_action_heatmaps()
    plot_zone_heatmaps()
    plot_timeseries_comparison()
    plot_zone_mean_profiles()
    plot_action_mean_profiles()

    print("\nAll plots saved!")
