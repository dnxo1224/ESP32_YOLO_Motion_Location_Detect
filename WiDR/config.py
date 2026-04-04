import os
import torch

# ============================================================
# 서브캐리어
# ============================================================
NULL_IN_HT_LTF = [64, 65] + list(range(123, 134)) + [191]  # 14개 null
VALID_HT_LTF_IDX = [i for i in range(64, 192) if i not in NULL_IN_HT_LTF]
NUM_SUBCARRIERS = len(VALID_HT_LTF_IDX)  # 114

# ============================================================
# 시퀀스
# ============================================================
SEQ_LEN = 872
SKIP_FIRST = 50   # 초기 과도 상태 패킷 스킵
NUM_RX = 4
FEATURE_DIM = NUM_SUBCARRIERS * NUM_RX  # 456

# ============================================================
# 레이블
# ============================================================
ACTION_MAP = {'handsup': 0, 'sit': 1, 'stand': 2, 'walk': 3}
NUM_ACTIONS = len(ACTION_MAP)

ZONE_MAP = {
    1: 0, 2: 0, 5: 0, 6: 0,
    3: 1, 4: 1, 7: 1, 8: 1,
    9: 2, 10: 2, 13: 2, 14: 2,
    11: 3, 12: 3, 15: 3, 16: 3,
}
NUM_ZONES = 4

ALL_SUBJECTS = ['gyj', 'jhj', 'jkw', 'kjh', 'kmh', 'kms',
                'kye', 'lsi', 'mhe', 'phr', 'stk', 'swt', 'ysj']

ACTION_NAMES = ['handsup', 'sit', 'stand', 'walk']
ZONE_NAMES = ['Zone0', 'Zone1', 'Zone2', 'Zone3']

# ============================================================
# 경로
# ============================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, '13_raw_data')
ALIGNED_DIR = os.path.join(PROJECT_ROOT, 'data_aligned_872')
INTERPOLATED_DIR = os.path.join(PROJECT_ROOT, 'data_interpolated_linear_872')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), 'weights')

os.makedirs(ALIGNED_DIR, exist_ok=True)
os.makedirs(INTERPOLATED_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)

# ============================================================
# 디바이스
# ============================================================
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

DEVICE = get_device()
