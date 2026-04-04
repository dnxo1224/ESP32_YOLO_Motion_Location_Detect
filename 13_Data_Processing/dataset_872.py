"""
공통 데이터셋 모듈 (800 x 166 x 4)
LightGBM/processed/ NPZ 파일을 로드하여 PyTorch Dataset으로 변환
"""
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# --- 상수 정의 ---
SEQ_LEN = 800
NUM_SUBCARRIERS = 166
NUM_RX = 4
FEATURE_DIM = NUM_SUBCARRIERS * NUM_RX  # 664

ACTION_MAP = {'handsup': 0, 'sit': 1, 'stand': 2, 'walk': 3}
# Zone mapping (1-16 -> 0-3)
ZONE_MAP = {
    1: 0, 2: 0, 5: 0, 6: 0,
    3: 1, 4: 1, 7: 1, 8: 1,
    9: 2, 10: 2, 13: 2, 14: 2,
    11: 3, 12: 3, 15: 3, 16: 3,
}

ALL_SUBJECTS = ['gyj', 'jhj', 'jkw', 'kjh', 'kmh', 'kms', 'kye', 'lsi', 'mhe', 'phr', 'stk', 'swt', 'ysj', 'kjh']
TEST_SUBJECTS = ['kms']  # LightGBM 설정에 맞춰 kjh, kms 2명으로 설정 가능
TEST_SUBJECT = 'kms' # 기본값
TRAIN_SUBJECTS = [s for s in ALL_SUBJECTS if s != TEST_SUBJECT]

# 데이터 경로 (LightGBM/processed)
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'LightGBM', 'processed'))


class CSIDataset872(Dataset):
    """
    CSI 데이터셋 (800 x 664)
    LightGBM/processed/*.npz 파일을 로드하여 슬라이딩 윈도우 생성

    Args:
        subjects: 포함할 피험자 리스트
        base_dir: LightGBM/processed 디렉토리 경로
        task: 'zone' (zone label만), 'action' (action label만), 'both' (둘 다)
        window_size: 슬라이딩 윈도우 크기 (None이면 전체 800 사용)
        stride: 슬라이딩 윈도우 이동 간격
    """
    def __init__(self, subjects, base_dir=None, task='zone', window_size=None, stride=None):
        if base_dir is None:
            base_dir = DATA_DIR
        self.base_dir = base_dir
        self.task = task
        self.window_size = window_size
        self.stride = stride
        
        self.data = []
        self.zone_labels = []
        self.action_labels = []
        self.subjects_info = []

        print(f"Loading samples for subjects: {subjects} from {base_dir}")
        npz_files = glob.glob(os.path.join(base_dir, "*.npz"))
        
        for f in tqdm(sorted(npz_files), desc="Loading NPZ"):
            data_npz = np.load(f, allow_pickle=True)
            subject = str(data_npz['subject'])
            
            if subject not in subjects:
                continue
            
            # grids: (4, 800, 166) -> (800, 166, 4) -> (800, 664)
            grids = data_npz['grids'] # (4, 800, 166)
            full_data = grids.transpose(1, 2, 0).reshape(SEQ_LEN, FEATURE_DIM).astype(np.float32)
            
            action = int(data_npz['action'])
            zone = int(data_npz['zone'])
            
            # 슬라이딩 윈도우 적용
            if window_size is not None and stride is not None:
                start = 0
                while start + window_size <= SEQ_LEN:
                    window = full_data[start : start + window_size]
                    self.data.append(window)
                    self.zone_labels.append(zone)
                    self.action_labels.append(action)
                    self.subjects_info.append(subject)
                    start += stride
            else:
                self.data.append(full_data)
                self.zone_labels.append(zone)
                self.action_labels.append(action)
                self.subjects_info.append(subject)

        self.data = np.array(self.data)  # (N_windows, window_size, 664)
        self.zone_labels = np.array(self.zone_labels)
        self.action_labels = np.array(self.action_labels)
        print(f"Loaded {len(self.data)} samples (task={task}, windows per sample={'all' if window_size is None else (SEQ_LEN-window_size)//stride + 1})")

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
    """StandardScaler로 train 데이터 기준 정규화"""
    if len(train_dataset) == 0 or len(test_dataset) == 0:
        return None
        
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
