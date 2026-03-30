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
    """MPS > CUDA > CPU 순서로 디바이스 선택"""
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')
