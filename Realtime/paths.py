"""
Realtime 파이프라인 공용 경로 및 sys.path 설정.

Realtime 모듈들은 항상 `import paths`를 가장 먼저 수행한다.
→ Zone_Expert(dataset, model_cnn)와 LightGBM(preprocess)을 import 가능하게 만든다.
"""
import os
import sys

REALTIME_DIR    = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT       = os.path.dirname(REALTIME_DIR)

ZONE_EXPERT_DIR = os.path.join(REPO_ROOT, 'Zone_Expert')
LIGHTGBM_DIR    = os.path.join(REPO_ROOT, 'LightGBM')
PROCESSED_DIR   = os.path.join(LIGHTGBM_DIR, 'processed')
RAW_DATA_DIR    = os.path.join(LIGHTGBM_DIR, 'data')

ARTIFACTS_DIR   = os.path.join(REALTIME_DIR, 'artifacts')
SCALER_PATH     = os.path.join(ARTIFACTS_DIR, 'scaler_train.npz')

ZONE_MLP_WEIGHTS   = os.path.join(
    REPO_ROOT, '13_Data_Processing', 'weights', 'zone_mlp_mean_only_best.pt'
)
EXPERT_WEIGHTS_DIR = os.path.join(ZONE_EXPERT_DIR, 'weights_cnn_norx_noos')

# Zone_Expert가 LightGBM보다 먼저 검색되도록 역순 삽입
for _p in (LIGHTGBM_DIR, ZONE_EXPERT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
