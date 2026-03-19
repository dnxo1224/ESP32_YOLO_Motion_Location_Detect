"""
공통 데이터셋 모듈 (872 x 114 x 4)
data_interpolated_linear_872/ CSV 파일을 로드하여 PyTorch Dataset으로 변환
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
ZONE_MAP = {
    1: 0, 2: 0, 5: 0, 6: 0,
    3: 1, 4: 1, 7: 1, 8: 1,
    9: 2, 10: 2, 13: 2, 14: 2,
    11: 3, 12: 3, 15: 3, 16: 3,
}

ALL_SUBJECTS = ['gyj', 'jhj', 'jkw', 'kjh', 'kmh', 'kms', 'kye', 'lsi', 'mhe', 'phr', 'stk', 'swt', 'ysj']
TEST_SUBJECT = 'kjh'
TRAIN_SUBJECTS = [s for s in ALL_SUBJECTS if s != TEST_SUBJECT]

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data_interpolated_linear_872')


def load_sample(base_dir, subject, action, position):
    """4개 RX CSV를 로드하여 (872, 456) numpy 배열로 반환"""
    rx_list = []
    for rx in range(1, 5):
        path = os.path.join(base_dir, f"{subject}_{action}_{position}_rx{rx}_872.csv")
        df = pd.read_csv(path, index_col=0)
        rx_list.append(df.values)  # (872, 114)
    return np.hstack(rx_list).astype(np.float32)  # (872, 456)


class CSIDataset872(Dataset):
    """
    CSI 데이터셋 (872 x 456)

    Args:
        subjects: 포함할 피험자 리스트
        base_dir: data_interpolated_linear_872 디렉토리 경로
        task: 'zone' (zone label만), 'action' (action label만), 'both' (둘 다)
    """
    def __init__(self, subjects, base_dir=None, task='zone'):
        if base_dir is None:
            base_dir = DATA_DIR
        self.base_dir = base_dir
        self.task = task
        self.samples = []  # (subject, action, position)
        self.data = []
        self.zone_labels = []
        self.action_labels = []

        # rx1 파일 스캔으로 샘플 목록 생성
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
                    print(f"Warning: NaN found in {prefix}, skipping")
                    continue
                self.data.append(data)
                self.zone_labels.append(ZONE_MAP[position])
                self.action_labels.append(ACTION_MAP[action])
                self.samples.append((subject, action, position))
            except FileNotFoundError as e:
                print(f"Warning: Missing file for {prefix}: {e}")
                continue

        self.data = np.array(self.data)  # (N, 872, 456)
        self.zone_labels = np.array(self.zone_labels)
        self.action_labels = np.array(self.action_labels)
        print(f"Loaded {len(self.data)} samples from {len(subjects)} subjects (task={task})")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float32)

        if self.task == 'zone':
            y = torch.tensor(self.zone_labels[idx], dtype=torch.long)
            return x, y
        elif self.task == 'action':
            y = torch.tensor(self.action_labels[idx], dtype=torch.long)
            return x, y
        else:  # 'both'
            y_zone = torch.tensor(self.zone_labels[idx], dtype=torch.long)
            y_action = torch.tensor(self.action_labels[idx], dtype=torch.long)
            return x, y_action, y_zone


def normalize_data(train_dataset, test_dataset):
    """
    StandardScaler로 train 데이터 기준 정규화
    데이터를 (N*872, 456)으로 reshape하여 피팅 후 원래 shape으로 복원
    """
    scaler = StandardScaler()

    # Train 데이터로 fit
    train_flat = train_dataset.data.reshape(-1, FEATURE_DIM)
    scaler.fit(train_flat)

    # Transform
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
