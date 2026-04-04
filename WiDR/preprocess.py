"""
전처리 파이프라인
  1. Raw CSI → 진폭 추출 + HT-LTF null 제거
  2. 4-RX seq_id 동기화 + 872-frame 정렬  → data_aligned_872/
  3. 선형 보간 (NaN 채우기)               → data_interpolated_linear_872/

단독 실행:
    python preprocess.py
"""

import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import (
    RAW_DATA_DIR, ALIGNED_DIR, INTERPOLATED_DIR,
    VALID_HT_LTF_IDX, NUM_RX, SEQ_LEN, SKIP_FIRST,
    NULL_IN_HT_LTF, NUM_SUBCARRIERS,
)


# ============================================================
# 1. 진폭 추출
# ============================================================
def extract_amplitude(csi_string):
    """
    Raw CSI 문자열 '[R0 I0 R1 I1 ...]' → amplitude 배열
    Returns None if invalid
    """
    try:
        s = str(csi_string)
        for ch in ['"', '[', ']', ',']:
            s = s.replace(ch, ' ')
        vals = np.array([float(v) for v in s.split()])
        if len(vals) < 2 or len(vals) % 2 != 0:
            return None
        real_parts = vals[0::2]
        imag_parts = vals[1::2]
        return np.sqrt(real_parts ** 2 + imag_parts ** 2)
    except Exception:
        return None


def find_csi_column(df):
    """'[' 을 포함하는 CSI 컬럼 인덱스 반환"""
    for col_idx in range(len(df.columns)):
        if '[' in str(df.iloc[0, col_idx]):
            return col_idx
    return None


def preprocess_raw_csv(filepath):
    """
    Raw CSI CSV → (seq_ids, amp_matrix)
    amp_matrix: (N, 114) HT-LTF null 제거 완료
    """
    df = pd.read_csv(filepath, header=None, low_memory=False)
    seq_ids = df.iloc[:, 2].values.astype(int)

    csi_col = find_csi_column(df)
    if csi_col is None:
        raise ValueError(f"CSI column not found in {filepath}")

    amplitudes = []
    valid_seq_ids = []
    for i in range(len(df)):
        amp = extract_amplitude(df.iloc[i, csi_col])
        if amp is None:
            continue
        n_sub = len(amp)
        if n_sub == 192:
            amp = amp[VALID_HT_LTF_IDX]       # 192 → 114
        elif n_sub == 64:
            valid_64 = list(range(6, 32)) + list(range(33, 59))
            amp = amp[valid_64]                # 64 → 52
        else:
            continue
        amplitudes.append(amp)
        valid_seq_ids.append(seq_ids[i])

    return np.array(valid_seq_ids), np.array(amplitudes, dtype=np.float32)


# ============================================================
# 2. 872-Frame 정렬
# ============================================================
def align_872_frames(subject, action, position):
    """
    Raw 4-RX CSV → 872-frame 정렬 CSV 4개를 data_aligned_872/ 에 저장
    Returns True on success
    """
    rx_data = {}
    all_seq_ids = set()

    for rx in range(1, NUM_RX + 1):
        filepath = os.path.join(RAW_DATA_DIR, f"{subject}_{action}_{position}_rx{rx}.csv")
        if not os.path.exists(filepath):
            return False
        seq_ids, amp_matrix = preprocess_raw_csv(filepath)
        if len(seq_ids) == 0:
            return False
        col_names = [f"sub_{j}" for j in range(amp_matrix.shape[1])]
        rx_df = pd.DataFrame(amp_matrix, index=seq_ids, columns=col_names)
        rx_df = rx_df[~rx_df.index.duplicated(keep='first')]
        rx_data[rx] = rx_df
        all_seq_ids.update(seq_ids)

    start_seq = min(all_seq_ids) + SKIP_FIRST
    target_indices = list(range(start_seq, start_seq + SEQ_LEN))

    for rx in range(1, NUM_RX + 1):
        aligned_df = rx_data[rx].reindex(target_indices)
        out_path = os.path.join(ALIGNED_DIR, f"{subject}_{action}_{position}_rx{rx}_872.csv")
        aligned_df.to_csv(out_path)

    return True


def run_alignment():
    """RawData/ 전체를 순회하며 872-frame 정렬 수행"""
    rx1_files = sorted(glob.glob(os.path.join(RAW_DATA_DIR, '*_rx1.csv')))
    if not rx1_files:
        print("Raw data 파일이 없습니다.")
        return 0

    success, fail = 0, 0
    for f in tqdm(rx1_files, desc="872-Frame 정렬"):
        basename = os.path.basename(f).replace('_rx1.csv', '')
        parts = basename.split('_')
        if len(parts) < 3:
            continue
        subject, action, position = parts[0], parts[1], parts[2]

        out_path = os.path.join(ALIGNED_DIR, f"{subject}_{action}_{position}_rx1_872.csv")
        if os.path.exists(out_path):
            success += 1
            continue

        if align_872_frames(subject, action, position):
            success += 1
        else:
            fail += 1

    print(f"정렬 완료: {success} 성공, {fail} 실패")
    return success


# ============================================================
# 3. 선형 보간
# ============================================================
def run_linear_interpolation():
    """data_aligned_872/ → data_interpolated_linear_872/ 선형 보간"""
    aligned_files = sorted(glob.glob(os.path.join(ALIGNED_DIR, '*_872.csv')))
    if not aligned_files:
        print("정렬된 데이터가 없습니다. run_alignment()를 먼저 실행하세요.")
        return 0

    count = 0
    for f in tqdm(aligned_files, desc="선형 보간"):
        out_path = os.path.join(INTERPOLATED_DIR, os.path.basename(f))
        if os.path.exists(out_path):
            count += 1
            continue
        df = pd.read_csv(f, index_col=0)
        if df.isna().sum().sum() > 0:
            df = df.interpolate(method='linear').bfill().ffill()
        df.to_csv(out_path)
        count += 1

    print(f"보간 완료: {count}개 파일")
    return count


# ============================================================
# 단독 실행
# ============================================================
if __name__ == '__main__':
    existing_aligned = glob.glob(os.path.join(ALIGNED_DIR, '*_rx1_872.csv'))
    if existing_aligned:
        print(f"정렬된 데이터 이미 존재: {len(existing_aligned)}개 — 스킵")
    else:
        run_alignment()

    existing_interp = glob.glob(os.path.join(INTERPOLATED_DIR, '*_rx1_872.csv'))
    if existing_interp:
        print(f"보간된 데이터 이미 존재: {len(existing_interp)}개 — 스킵")
    else:
        run_linear_interpolation()
