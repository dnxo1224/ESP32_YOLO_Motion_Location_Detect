"""
BiLSTM 슬라이딩 윈도우 개별 예측 결과 분석
- 학습된 zone_lstm_best.pt 로드
- 테스트 데이터(kjh)의 각 샘플에 대해 79개 윈도우의 개별 예측 출력
- 윈도우 단위 정확도 vs voting 정확도 비교
"""
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from collections import Counter
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from dataset_872 import (
    CSIDataset872, normalize_data, get_device,
    TRAIN_SUBJECTS, TEST_SUBJECT, FEATURE_DIM
)
from train_zone_lstm_872 import SlidingWindowDataset, ZoneLSTM, WINDOW_SIZE, STRIDE

ZONE_NAMES = ['Zone0', 'Zone1', 'Zone2', 'Zone3']
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), 'weights')
CHECKPOINT = os.path.join(WEIGHTS_DIR, 'zone_lstm_best.pt')


def evaluate_windows():
    device = get_device()
    print(f"Device: {device}")

    # --- 데이터 로드 ---
    print("\nLoading data...")
    train_raw = CSIDataset872(TRAIN_SUBJECTS, task='zone')
    test_raw = CSIDataset872([TEST_SUBJECT], task='zone')
    normalize_data(train_raw, test_raw)

    print("\nCreating sliding window dataset...")
    test_win = SlidingWindowDataset(test_raw)

    # --- 모델 로드 ---
    model = ZoneLSTM().to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    model.eval()
    print(f"Loaded checkpoint: {CHECKPOINT}")

    # --- 윈도우별 예측 수집 ---
    loader = DataLoader(test_win, batch_size=256, shuffle=False)
    all_preds = []

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Inference"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(1).cpu().numpy()
            all_preds.extend(preds)

    all_preds = np.array(all_preds)

    # --- 샘플별 그룹핑 ---
    sample_ids = np.array(test_win.sample_ids)
    sample_labels_arr = np.array(test_win.labels)
    unique_samples = sorted(set(sample_ids))
    num_samples = len(unique_samples)

    print(f"\n{'='*80}")
    print(f"테스트 피험자: {TEST_SUBJECT} | 총 샘플: {num_samples} | "
          f"윈도우/샘플: {WINDOW_SIZE}프레임, stride={STRIDE}")
    print(f"{'='*80}")

    total_window_correct = 0
    total_windows = 0
    vote_correct = 0
    sample_win_accs = []
    heatmap_data = []  # 히트맵용: 각 샘플의 윈도우별 정답/오답

    for sid in unique_samples:
        mask = sample_ids == sid
        win_preds = all_preds[mask]
        true_label = sample_labels_arr[mask][0]

        # 샘플 정보
        subject, action, position = test_raw.samples[sid]
        n_windows = len(win_preds)

        # 윈도우별 정답/오답
        correct_mask = (win_preds == true_label)
        n_correct = correct_mask.sum()
        win_acc = n_correct / n_windows

        # 다수결
        voted = Counter(win_preds.tolist()).most_common(1)[0][0]
        vote_result = "O" if voted == true_label else "X"

        # 통계 누적
        total_window_correct += n_correct
        total_windows += n_windows
        if voted == true_label:
            vote_correct += 1
        sample_win_accs.append(win_acc)

        # 히트맵 데이터
        heatmap_data.append(correct_mask.astype(int))

        # --- 콘솔 출력 ---
        # 예측 분포
        pred_counts = Counter(win_preds.tolist())
        dist_str = " | ".join(f"{ZONE_NAMES[z]}:{pred_counts.get(z, 0)}" for z in range(4))

        print(f"\n[Sample {sid:>3}] {subject}_{action}_{position} "
              f"| 정답: {ZONE_NAMES[true_label]} | "
              f"Voting: {ZONE_NAMES[voted]} [{vote_result}]")
        print(f"  윈도우 정확도: {n_correct}/{n_windows} ({win_acc:.1%})")
        print(f"  예측 분포: {dist_str}")

        # 윈도우별 예측 시각화 (한 줄로)
        pred_chars = []
        for p in win_preds:
            if p == true_label:
                pred_chars.append(str(p))
            else:
                pred_chars.append(f"[{p}]")  # 오답은 괄호로 표시
        print(f"  윈도우 예측: {' '.join(pred_chars)}")

    # --- 전체 통계 ---
    overall_win_acc = total_window_correct / total_windows
    overall_vote_acc = vote_correct / num_samples
    sample_win_accs = np.array(sample_win_accs)

    print(f"\n{'='*80}")
    print(f"전체 통계")
    print(f"{'='*80}")
    print(f"  윈도우 단위 정확도: {total_window_correct}/{total_windows} ({overall_win_acc:.4f})")
    print(f"  Voting 정확도:      {vote_correct}/{num_samples} ({overall_vote_acc:.4f})")
    print(f"  다수결 효과:        {overall_vote_acc - overall_win_acc:+.4f}")
    print(f"\n  샘플별 윈도우 정확도:")
    print(f"    평균: {sample_win_accs.mean():.4f}")
    print(f"    최소: {sample_win_accs.min():.4f}")
    print(f"    최대: {sample_win_accs.max():.4f}")
    print(f"    표준편차: {sample_win_accs.std():.4f}")

    # --- 시각화 1: 윈도우 정확도 히스토그램 ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(sample_win_accs, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(sample_win_accs.mean(), color='red', linestyle='--',
               label=f'Mean: {sample_win_accs.mean():.3f}')
    ax.set_xlabel('Window Accuracy per Sample')
    ax.set_ylabel('Count')
    ax.set_title(f'BiLSTM Window-level Accuracy Distribution\n'
                 f'Window Acc: {overall_win_acc:.4f} → Vote Acc: {overall_vote_acc:.4f}')
    ax.legend()
    plt.tight_layout()
    hist_path = os.path.join(RESULTS_DIR, 'window_accuracy_histogram.png')
    plt.savefig(hist_path, dpi=150)
    plt.close()
    print(f"\n히스토그램 저장: {hist_path}")

    # --- 시각화 2: 윈도우 예측 히트맵 (정답=1, 오답=0) ---
    n_windows_per = len(heatmap_data[0])
    heatmap_arr = np.array(heatmap_data)  # (num_samples, n_windows)

    fig, ax = plt.subplots(figsize=(16, max(6, num_samples * 0.25)))
    cmap = plt.cm.colors.ListedColormap(['#e74c3c', '#2ecc71'])  # 빨강=오답, 초록=정답
    im = ax.imshow(heatmap_arr, aspect='auto', cmap=cmap, interpolation='none')
    
    # y축: 샘플 이름
    y_labels = [f"{test_raw.samples[sid][1]}_{test_raw.samples[sid][2]} "
                f"({ZONE_NAMES[sample_labels_arr[sample_ids == sid][0]]})"
                for sid in unique_samples]
    ax.set_yticks(range(num_samples))
    ax.set_yticklabels(y_labels, fontsize=6)
    ax.set_xlabel(f'Window Index (0-{n_windows_per-1})')
    ax.set_ylabel('Sample (action_position)')
    ax.set_title(f'Window Predictions Heatmap (Green=Correct, Red=Wrong)\n'
                 f'Test Subject: {TEST_SUBJECT}')

    # 컬러바
    cbar = plt.colorbar(im, ax=ax, ticks=[0.25, 0.75])
    cbar.ax.set_yticklabels(['Wrong', 'Correct'])

    plt.tight_layout()
    heatmap_path = os.path.join(RESULTS_DIR, 'window_predictions_heatmap.png')
    plt.savefig(heatmap_path, dpi=150)
    plt.close()
    print(f"히트맵 저장: {heatmap_path}")


if __name__ == '__main__':
    evaluate_windows()
