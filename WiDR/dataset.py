"""
WiDR 데이터셋 모듈
data_interpolated_linear_872/ CSV 파일을 로드하여 PyTorch Dataset으로 변환
Dual-task (Action Recognition + Zone Classification) 전용
"""
import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# --- 상수 정의 ---
SEQ_LEN = 872
NUM_SUBCARRIERS = 114
NUM_RX = 4
FEATURE_DIM = NUM_SUBCARRIERS * NUM_RX  # 456

ACTION_MAP = {'handsup': 0, 'sit': 1, 'stand': 2, 'walk': 3}
NUM_ACTIONS = len(ACTION_MAP)

ZONE_MAP = {
    1: 0, 2: 0, 5: 0, 6: 0,
    3: 1, 4: 1, 7: 1, 8: 1,
    9: 2, 10: 2, 13: 2, 14: 2,
    11: 3, 12: 3, 15: 3, 16: 3,
}
NUM_ZONES = 4

ALL_SUBJECTS = [
    'gyj', 'jhj', 'jkw', 'kjh', 'kmh', 'kms',
    'kye', 'lsi', 'mhe', 'phr', 'stk', 'swt', 'ysj'
]

# 기본 데이터 디렉토리
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data_interpolated_linear_872')


def load_sample(base_dir, subject, action, position):
    """4개 RX CSV를 로드하여 (872, 456) numpy 배열로 반환"""
    rx_list = []
    for rx in range(1, NUM_RX + 1):
        path = os.path.join(base_dir, f"{subject}_{action}_{position}_rx{rx}_872.csv")
        df = pd.read_csv(path, index_col=0)
        rx_list.append(df.values)  # (872, 114)
    return np.hstack(rx_list).astype(np.float32)  # (872, 456)


class CSIDataset(Dataset):
    """
    Wi-Fi CSI 데이터셋 (872 x 456) — Dual-task 전용

    항상 (x, y_action, y_zone) 반환
    """
    def __init__(self, subjects, base_dir=None):
        if base_dir is None:
            base_dir = DATA_DIR
        self.base_dir = base_dir
        self.samples = []
        self.data = []
        self.action_labels = []
        self.zone_labels = []

        rx1_files = glob.glob(os.path.join(base_dir, "*_rx1_872.csv"))
        for f in sorted(rx1_files):
            basename = os.path.basename(f)
            prefix = basename.replace('_rx1_872.csv', '')
            parts = prefix.split('_')
            subject = parts[0]
            action = parts[1]
            position = int(parts[2])

            if subject not in subjects:
                continue
            if action not in ACTION_MAP:
                continue

            try:
                data = load_sample(base_dir, subject, action, position)
                if np.isnan(data).any():
                    print(f"Warning: NaN in {prefix}, skipping")
                    continue
                self.data.append(data)
                self.action_labels.append(ACTION_MAP[action])
                self.zone_labels.append(ZONE_MAP[position])
                self.samples.append((subject, action, position))
            except FileNotFoundError as e:
                print(f"Warning: Missing file for {prefix}: {e}")
                continue

        self.data = np.array(self.data)  # (N, 872, 456)
        self.action_labels = np.array(self.action_labels)
        self.zone_labels = np.array(self.zone_labels)
        print(f"[CSIDataset] Loaded {len(self.data)} samples from {len(subjects)} subjects")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float32)
        y_action = torch.tensor(self.action_labels[idx], dtype=torch.long)
        y_zone = torch.tensor(self.zone_labels[idx], dtype=torch.long)
        return x, y_action, y_zone


def normalize_data(train_dataset, test_dataset):
    """StandardScaler로 train 기준 정규화 (in-place)"""
    scaler = StandardScaler()
    train_flat = train_dataset.data.reshape(-1, FEATURE_DIM)
    scaler.fit(train_flat)

    train_dataset.data = scaler.transform(train_flat).reshape(-1, SEQ_LEN, FEATURE_DIM).astype(np.float32)
    test_flat = test_dataset.data.reshape(-1, FEATURE_DIM)
    test_dataset.data = scaler.transform(test_flat).reshape(-1, SEQ_LEN, FEATURE_DIM).astype(np.float32)
    return scaler


def get_device():
    """CUDA > CPU 순서로 디바이스 선택"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


# ============================================================
# Raw CSI Dataset (Neural CDE용) — 13_raw_data/ 직접 로드
# ============================================================

# 상수 (config.py와 동기화)
_RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '13_raw_data')
_SKIP_FIRST   = 50
_L_FIXED      = SEQ_LEN   # 872


class RawCSIDataset(Dataset):
    """
    Raw CSI 데이터셋 — Neural CDE 입력용

    13_raw_data/ 에서 직접 로드하여 NaN 그리드로 변환.
    torchcde의 natural_cubic_spline_coeffs 가 NaN을 자동 보간.

    반환: (x, y_action, y_zone)
      x: (4, L_fixed, 115) — 4 RX × 872 프레임 × (114 진폭 + 1 시간채널)
         진폭은 누락 패킷 위치가 NaN, 시간채널은 항상 채워짐
    """
    def __init__(self, subjects, raw_dir=None, l_fixed=_L_FIXED, skip_first=_SKIP_FIRST):
        from preprocess import preprocess_raw_csv  # 순환 import 방지

        if raw_dir is None:
            raw_dir = _RAW_DATA_DIR

        self.data          = []
        self.action_labels = []
        self.zone_labels   = []

        # rx1 파일 기준으로 샘플 목록 구성
        rx1_files = sorted(glob.glob(os.path.join(raw_dir, '*_rx1.csv')))
        if not rx1_files:
            print(f"[RawCSIDataset] Raw data 없음: {raw_dir}")

        for f in tqdm(rx1_files, desc='[RawCSIDataset] 로딩'):
            basename = os.path.basename(f).replace('_rx1.csv', '')
            parts    = basename.split('_')
            subject, action, position = parts[0], parts[1], int(parts[2])

            if subject not in subjects or action not in ACTION_MAP:
                continue

            try:
                # ── 공통 시간 그리드 결정 (rx1 기준) ──────────────────
                seq_ids_rx1, _ = preprocess_raw_csv(f)
                if len(seq_ids_rx1) == 0:
                    continue
                start_seq = int(seq_ids_rx1.min()) + skip_first
                grid      = np.arange(start_seq, start_seq + l_fixed, dtype=np.int64)

                # ── 4 RX 로드 → NaN 그리드 생성 ─────────────────────
                rx_grids = []
                skip_sample = False
                for rx in range(1, NUM_RX + 1):
                    rx_path = os.path.join(raw_dir, f"{subject}_{action}_{position}_rx{rx}.csv")
                    seq_ids, amp_matrix = preprocess_raw_csv(rx_path)

                    # 중복 seq_id 제거 (재전송 패킷)
                    _, unique_idx = np.unique(seq_ids, return_index=True)
                    seq_ids    = seq_ids[unique_idx]
                    amp_matrix = amp_matrix[unique_idx]

                    # 유효 패킷이 2개 미만이면 보간 불가
                    if len(seq_ids) < 2:
                        skip_sample = True
                        break

                    # NaN 그리드 구성 (872, 114)
                    amp_grid = np.full((l_fixed, NUM_SUBCARRIERS), np.nan, dtype=np.float32)
                    sid_to_idx = {int(sid): i for i, sid in enumerate(seq_ids)}
                    for j, sid in enumerate(grid):
                        if sid in sid_to_idx:
                            amp_grid[j] = amp_matrix[sid_to_idx[sid]]

                    # 시간 채널 추가: i/30.0 (항상 채워짐)
                    t_rel = (np.arange(l_fixed, dtype=np.float32) / 30.0).reshape(-1, 1)
                    rx_grids.append(np.hstack([amp_grid, t_rel]))  # (872, 115)

                if skip_sample:
                    continue

                # (4, 872, 115)
                self.data.append(np.stack(rx_grids, axis=0))
                self.action_labels.append(ACTION_MAP[action])
                self.zone_labels.append(ZONE_MAP[position])

            except (FileNotFoundError, Exception) as e:
                print(f"[RawCSIDataset] 스킵 {basename}: {e}")
                continue

        self.data          = np.array(self.data, dtype=np.float32)   # (N, 4, 872, 115)
        self.action_labels = np.array(self.action_labels)
        self.zone_labels   = np.array(self.zone_labels)
        print(f"[RawCSIDataset] {len(self.data)}개 샘플 로드 완료")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x        = torch.tensor(self.data[idx], dtype=torch.float32)  # (4, 872, 115)
        y_action = torch.tensor(self.action_labels[idx], dtype=torch.long)
        y_zone   = torch.tensor(self.zone_labels[idx],   dtype=torch.long)
        return x, y_action, y_zone


def normalize_raw_data(train_dataset: RawCSIDataset,
                       test_dataset:  RawCSIDataset) -> StandardScaler:
    """
    RawCSIDataset 진폭 채널(0:114)만 정규화 (in-place).
    시간 채널(index 114)은 건드리지 않음.

    Returns:
        train 기준으로 fit된 StandardScaler
    """
    scaler = StandardScaler()

    # train 진폭만 추출하여 fit
    # data shape: (N, 4, 872, 115) → 진폭: (N, 4, 872, 114)
    train_amp  = train_dataset.data[:, :, :, :NUM_SUBCARRIERS]
    train_flat = train_amp.reshape(-1, NUM_SUBCARRIERS)
    scaler.fit(train_flat)

    # train 적용
    train_dataset.data[:, :, :, :NUM_SUBCARRIERS] = (
        scaler.transform(train_flat)
        .reshape(-1, NUM_RX, SEQ_LEN, NUM_SUBCARRIERS)
        .astype(np.float32)
    )

    # test 적용
    test_amp  = test_dataset.data[:, :, :, :NUM_SUBCARRIERS]
    test_flat = test_amp.reshape(-1, NUM_SUBCARRIERS)
    test_dataset.data[:, :, :, :NUM_SUBCARRIERS] = (
        scaler.transform(test_flat)
        .reshape(-1, NUM_RX, SEQ_LEN, NUM_SUBCARRIERS)
        .astype(np.float32)
    )

    return scaler
