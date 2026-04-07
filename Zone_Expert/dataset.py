"""
Zone Expert Action Classifier — 데이터셋 모듈
NPZ 파일(LightGBM/processed/*.npz)을 로드하여
슬라이딩 윈도우 Dataset을 생성한다.
"""
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# ─── 상수 ─────────────────────────────────────────────────────────────────────
SEQ_LEN         = 800
NUM_SUBCARRIERS = 166
NUM_RX          = 4
FEATURE_DIM     = NUM_SUBCARRIERS * NUM_RX   # 664

ACTION_MAP = {'handsup': 0, 'sit': 1, 'stand': 2, 'walk': 3}
ZONE_MAP   = {
    1: 0, 2: 0, 5: 0, 6: 0,
    3: 1, 4: 1, 7: 1, 8: 1,
    9: 2, 10: 2, 13: 2, 14: 2,
    11: 3, 12: 3, 15: 3, 16: 3,
}

ALL_SUBJECTS   = ['gyj', 'jhj', 'jkw', 'kjh', 'kmh', 'kms',
                  'kye', 'lsi', 'mhe', 'phr', 'stk', 'swt', 'ysj']
TEST_SUBJECT   = 'kms'
TRAIN_SUBJECTS = [s for s in ALL_SUBJECTS if s != TEST_SUBJECT]

# LightGBM/processed 디렉토리 (이 파일 기준 ../../LightGBM/processed)
DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'LightGBM', 'processed')
)


# ─── 원시 샘플 Dataset ────────────────────────────────────────────────────────

class CSIRawDataset(Dataset):
    """
    NPZ 파일을 로드하여 전체 시퀀스(800, 664) 단위로 반환.
    정규화 전 데이터로 사용 → normalize_datasets() 후 슬라이딩 윈도우 적용.

    __getitem__ 반환: (x: FloatTensor(800, 664), y_action: long, y_zone: long)
    """
    def __init__(self, subjects, base_dir=None):
        if base_dir is None:
            base_dir = DATA_DIR

        self.data          = []
        self.action_labels = []
        self.zone_labels   = []

        npz_files = sorted(glob.glob(os.path.join(base_dir, '*.npz')))
        print(f"[CSIRawDataset] subjects={subjects}, dir={base_dir}")

        for fpath in tqdm(npz_files, desc='Loading NPZ'):
            npz = np.load(fpath, allow_pickle=True)
            subject = str(npz['subject'])
            if subject not in subjects:
                continue

            # grids: (4, 800, 166) → (800, 166, 4) → (800, 664)
            grids     = npz['grids']  # (4, 800, 166)
            full_data = grids.transpose(1, 2, 0).reshape(SEQ_LEN, FEATURE_DIM).astype(np.float32)

            self.data.append(full_data)
            self.action_labels.append(int(npz['action']))
            self.zone_labels.append(int(npz['zone']))

        self.data          = np.array(self.data)           # (N, 800, 664)
        self.action_labels = np.array(self.action_labels)  # (N,)
        self.zone_labels   = np.array(self.zone_labels)    # (N,)
        print(f"  → {len(self.data)} samples loaded")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x       = torch.tensor(self.data[idx], dtype=torch.float32)
        y_action = torch.tensor(self.action_labels[idx], dtype=torch.long)
        y_zone  = torch.tensor(self.zone_labels[idx], dtype=torch.long)
        return x, y_action, y_zone


# ─── 정규화 ───────────────────────────────────────────────────────────────────

def normalize_datasets(train_ds: CSIRawDataset,
                       test_ds:  CSIRawDataset) -> StandardScaler:
    """
    train 데이터 기준 StandardScaler 적합 후 train/test 모두 변환.
    in-place로 .data 배열을 교체한다.
    """
    scaler    = StandardScaler()
    train_flat = train_ds.data.reshape(-1, FEATURE_DIM)
    scaler.fit(train_flat)

    train_ds.data = scaler.transform(train_flat) \
                          .reshape(-1, SEQ_LEN, FEATURE_DIM) \
                          .astype(np.float32)

    test_flat = test_ds.data.reshape(-1, FEATURE_DIM)
    test_ds.data  = scaler.transform(test_flat) \
                          .reshape(-1, SEQ_LEN, FEATURE_DIM) \
                          .astype(np.float32)

    print(f"[normalize_datasets] scaler fitted on {len(train_flat)} frames")
    return scaler


# ─── 슬라이딩 윈도우 Dataset ──────────────────────────────────────────────────

class SlidingWindowDataset(Dataset):
    """
    정규화된 CSIRawDataset에서 슬라이딩 윈도우를 생성한다.

    __getitem__ 반환:
        x        : FloatTensor(window_size, 664)
        y_action : long
        y_zone   : long
        sample_id: int  (원본 샘플 인덱스, majority voting에 사용)
    """
    def __init__(self, raw_ds: CSIRawDataset,
                 window_size: int = 50,
                 stride:      int = 25):
        self.window_size  = window_size
        self.stride       = stride

        self.windows      = []
        self.action_labels = []
        self.zone_labels  = []
        self.sample_ids   = []

        for idx in range(len(raw_ds)):
            seq    = raw_ds.data[idx]          # (800, 664)
            a_lbl  = int(raw_ds.action_labels[idx])
            z_lbl  = int(raw_ds.zone_labels[idx])
            seq_len = seq.shape[0]

            start = 0
            while start + window_size <= seq_len:
                window = seq[start:start + window_size]
                # 윈도우별 평균 제거 (추론 시에도 동일하게 적용 가능)
                window = window - window.mean(axis=0)
                self.windows.append(window)
                self.action_labels.append(a_lbl)
                self.zone_labels.append(z_lbl)
                self.sample_ids.append(idx)
                start += stride

        self.windows = np.array(self.windows, dtype=np.float32)
        print(f"[SlidingWindowDataset] {len(raw_ds)} samples "
              f"→ {len(self.windows)} windows "
              f"(window={window_size}, stride={stride})")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        x        = torch.tensor(self.windows[idx],       dtype=torch.float32)
        y_action = torch.tensor(self.action_labels[idx], dtype=torch.long)
        y_zone   = torch.tensor(self.zone_labels[idx],   dtype=torch.long)
        return x, y_action, y_zone, self.sample_ids[idx]

    def oversample_zone(self, zone_id: int, factor: int = 2) -> None:
        """
        zone_id 데이터를 factor배 복제하여 학습 데이터에 추가한다. (in-place)

        Args:
            zone_id: 복제할 Zone 번호
            factor : 복제 배수 (기본 2 → 원본 포함 총 2배)
        """
        mask = np.array(self.zone_labels) == zone_id
        extra = factor - 1  # 추가할 복제 횟수 (factor=2 → 1번 추가)

        zone_windows = self.windows[mask]
        zone_actions = [v for v, m in zip(self.action_labels, mask) if m]
        zone_zones   = [v for v, m in zip(self.zone_labels,   mask) if m]
        zone_ids     = [v for v, m in zip(self.sample_ids,    mask) if m]

        self.windows       = np.concatenate(
            [self.windows] + [zone_windows] * extra, axis=0
        )
        self.action_labels = self.action_labels + zone_actions * extra
        self.zone_labels   = self.zone_labels   + zone_zones   * extra
        self.sample_ids    = self.sample_ids    + zone_ids     * extra

        before = mask.sum()
        after  = before * factor
        print(f"[oversample_zone={zone_id}] {before} → {after} windows "
              f"(×{factor}, total: {len(self.windows)})")

    def filter_by_zone(self, zone_id: int) -> 'SlidingWindowDataset':
        """
        zone_id에 해당하는 윈도우만 남긴 새 Dataset을 반환한다.
        원본 Dataset은 변경되지 않는다.
        """
        mask = np.array(self.zone_labels) == zone_id
        new_ds = object.__new__(SlidingWindowDataset)
        new_ds.window_size   = self.window_size
        new_ds.stride        = self.stride
        new_ds.windows       = self.windows[mask]
        new_ds.action_labels = [v for v, m in zip(self.action_labels, mask) if m]
        new_ds.zone_labels   = [v for v, m in zip(self.zone_labels,   mask) if m]
        new_ds.sample_ids    = [v for v, m in zip(self.sample_ids,    mask) if m]
        print(f"[filter_by_zone={zone_id}] {mask.sum()} / {len(mask)} windows selected")
        return new_ds


# ─── 디바이스 선택 ─────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    """CUDA → MPS → CPU 순서로 선택"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')
