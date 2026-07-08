"""
윈도우 단위 실시간 전처리.

오프라인 LightGBM/preprocess.py의 remove_outliers + interpolate_grid를
200프레임 윈도우에 적용한 뒤, 저장된 train StandardScaler를 적용한다.

[의도적 근사] 오프라인은 3σ 통계를 800프레임 그리드 전체에서 계산하지만,
실시간은 예측 시점의 200프레임 윈도우에서 계산한다. 영향은
verify_replay.py에서 정량화한다.
"""
import warnings

import numpy as np
import pandas as pd


def preprocess_window(win: np.ndarray,
                      scaler_mean: np.ndarray,
                      scaler_scale: np.ndarray,
                      sigma: float = 3.0) -> np.ndarray:
    """
    win: (200, 664) float32, 결측 프레임/RX는 NaN.
    반환: (200, 664) float32, 이상치 제거 + 보간 + 정규화 완료.
    """
    with warnings.catch_warnings():
        # 전 구간 NaN 컬럼(RX 무응답)에서 nanmean 경고 억제 → fillna(0)로 처리됨
        warnings.simplefilter('ignore', category=RuntimeWarning)
        col_mean = np.nanmean(win, axis=0)
        col_std = np.nanstd(win, axis=0)
    col_std[~np.isfinite(col_std)] = 1.0
    col_std[col_std == 0] = 1.0

    cleaned = win.copy()
    with np.errstate(invalid='ignore'):
        spike = np.abs(win - col_mean) > sigma * col_std
    cleaned[spike] = np.nan

    # 선형 보간 + ffill/bfill + 0 채움 (interpolate_grid와 동일)
    df = pd.DataFrame(cleaned)
    df = df.interpolate(method='linear', axis=0)
    df = df.ffill(axis=0).bfill(axis=0)
    df = df.fillna(0.0)
    filled = df.values.astype(np.float32)

    return ((filled - scaler_mean) / scaler_scale).astype(np.float32)
