"""
[1회 실행] Train StandardScaler 영속화.

Zone_Expert/dataset.py의 normalize_datasets()와 동일하게
TRAIN_SUBJECTS(kms 제외 12명)의 NPZ 전체 프레임으로 StandardScaler를 fit한 뒤
mean_/scale_만 npz로 저장한다. (sklearn 버전에 독립적 — 실시간에서는
(x - mean) / scale 산술 연산만 수행)

실행:
    python Realtime/export_scaler.py
"""
import os

import numpy as np
from sklearn.preprocessing import StandardScaler

import paths
from dataset import CSIRawDataset, TRAIN_SUBJECTS, FEATURE_DIM  # Zone_Expert


def main():
    os.makedirs(paths.ARTIFACTS_DIR, exist_ok=True)

    print(f'[export_scaler] loading train subjects: {TRAIN_SUBJECTS}')
    train = CSIRawDataset(TRAIN_SUBJECTS)

    flat = train.data.reshape(-1, FEATURE_DIM)
    print(f'[export_scaler] fitting StandardScaler on {flat.shape[0]} frames ...')
    scaler = StandardScaler()
    scaler.fit(flat)

    np.savez(
        paths.SCALER_PATH,
        mean=scaler.mean_.astype(np.float64),
        scale=scaler.scale_.astype(np.float64),
        n_frames=np.int64(flat.shape[0]),
        subjects=np.array(TRAIN_SUBJECTS),
    )
    print(f'[export_scaler] saved -> {paths.SCALER_PATH}')

    # sanity check: 산술 적용 == scaler.transform (float64 기준 비교)
    sample = flat[:1000].astype(np.float64)
    manual = (sample - scaler.mean_) / scaler.scale_
    ref = scaler.transform(sample)
    err = np.abs(manual - ref).max()
    print(f'[export_scaler] sanity check max|manual - transform| = {err:.2e}')
    assert err < 1e-6, 'manual scaling does not match scaler.transform'
    print('[export_scaler] OK')


def load_scaler(path: str = None):
    """저장된 스케일러 → (mean(664,), scale(664,)) float32."""
    if path is None:
        path = paths.SCALER_PATH
    z = np.load(path)
    return z['mean'].astype(np.float32), z['scale'].astype(np.float32)


if __name__ == '__main__':
    main()
