"""
전처리 캐시 생성기 — Raw CSI → NPZ 저장

처리 파이프라인:
  1. Raw CSV 로드 (seq_id + amplitude)
  2. seq_id 기반 800-그리드 정렬 (처음 50 시퀀스 제거, 패킷 손실 → NaN)
  3. 3-sigma outlier 제거
  4. 선형 보간 (ffill/bfill fallback)
  5. processed/{subject}_{action}_{position}.npz 저장
     → grids: (4, 800, 166) float32  (4 RX 보간 완료 그리드)

이미 저장된 파일은 자동 skip.

실행:
    python preprocess.py
"""

import os
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ── 상수 ──────────────────────────────────────────────────────────────────────

_NULL_IDX    = (list(range(0, 6)) + [32] +
                list(range(59, 66)) + list(range(123, 134)) + [191])
VALID_IDX    = [i for i in range(192) if i not in _NULL_IDX]
NUM_FEATURES = len(VALID_IDX)   # 166

ACTION_MAP = {'handsup': 0, 'sit': 1, 'stand': 2, 'walk': 3}
ZONE_MAP   = {
     1: 0,  2: 0,  5: 0,  6: 0,
     3: 1,  4: 1,  7: 1,  8: 1,
     9: 2, 10: 2, 13: 2, 14: 2,
    11: 3, 12: 3, 15: 3, 16: 3,
}

RAW_DIR       = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
PROCESSED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'processed')

GRID_SIZE     = 800
SKIP_HEAD     = 50
MAX_SEQ       = 65536
OUTLIER_SIGMA = 3.0
NAN_THRESHOLD = 0.5


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
                    'position':   pos,
                })
    return samples


# ── 전처리 함수 ────────────────────────────────────────────────────────────────

def seq_slot(sid, start_seq, max_seq=MAX_SEQ):
    d = int(sid - start_seq) % max_seq
    if d > max_seq // 2:
        return -1
    return d


def compute_start_seq(seq_ids_list, skip_head=SKIP_HEAD):
    start_seqs = []
    for seq_ids in seq_ids_list:
        unique_sorted = np.sort(np.unique(seq_ids))
        if len(unique_sorted) <= skip_head:
            start_seqs.append(int(unique_sorted[0]))
        else:
            start_seqs.append(int(unique_sorted[skip_head]))
    return max(start_seqs)


def align_to_grid(amplitude, seq_ids, start_seq, grid_size=GRID_SIZE):
    grid = np.full((grid_size, NUM_FEATURES), np.nan, dtype=np.float32)
    for i, sid in enumerate(seq_ids):
        slot = seq_slot(sid, start_seq)
        if 0 <= slot < grid_size:
            grid[slot] = amplitude[i]
    nan_ratio = np.isnan(grid).mean()
    return grid, nan_ratio


def remove_outliers(grid, sigma=OUTLIER_SIGMA):
    mean = np.nanmean(grid, axis=0)
    std  = np.nanstd(grid, axis=0)
    std[std == 0] = 1.0
    spike = np.abs(grid - mean) > sigma * std
    cleaned = grid.copy()
    cleaned[spike] = np.nan
    return cleaned


def interpolate_grid(grid):
    df = pd.DataFrame(grid)
    df = df.interpolate(method='linear', axis=0)
    df = df.ffill(axis=0).bfill(axis=0)
    df = df.fillna(0.0)
    return df.values.astype(np.float32)


def preprocess_sample(sample):
    """
    샘플 딕셔너리 → 전처리 완료된 4 RX 그리드.

    Returns
    -------
    grids : (4, 800, 166) float32
    """
    start_seq = compute_start_seq(sample['seq_ids'])
    grids = []
    for rx in range(4):
        grid, nan_ratio = align_to_grid(
            sample['amplitudes'][rx], sample['seq_ids'][rx], start_seq
        )
        if nan_ratio > NAN_THRESHOLD:
            warnings.warn(
                f"[{sample['subject']} action={sample['action']} "
                f"pos={sample['position']} RX{rx+1}] NaN ratio={nan_ratio:.1%}"
            )
        grid = remove_outliers(grid)
        grid = interpolate_grid(grid)
        grids.append(grid)
    return np.stack(grids, axis=0)   # (4, 800, 166)


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print('Loading raw CSI data ...')
    all_samples = load_all_samples(RAW_DIR)
    print(f'Total samples: {len(all_samples)}')

    skipped, saved = 0, 0
    for s in tqdm(all_samples, desc='Preprocessing'):
        action_name = [k for k, v in ACTION_MAP.items() if v == s['action']][0]
        fname = f"{s['subject']}_{action_name}_{s['position']}.npz"
        save_path = os.path.join(PROCESSED_DIR, fname)

        if os.path.exists(save_path):
            skipped += 1
            continue

        grids = preprocess_sample(s)
        np.savez_compressed(
            save_path,
            grids   = grids,             # (4, 800, 166) float32
            action  = np.int32(s['action']),
            zone    = np.int32(s['zone']),
            subject = np.str_(s['subject']),
            position= np.int32(s['position']),
        )
        saved += 1

    print(f'\nDone. saved={saved}, skipped(already exists)={skipped}')
    print(f'Processed files → {PROCESSED_DIR}')


if __name__ == '__main__':
    main()
