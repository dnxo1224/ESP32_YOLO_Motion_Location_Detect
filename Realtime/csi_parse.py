"""
CSI_DATA CSV 라인 → (seq_id, amplitude(166,)) 파서.

LightGBM/preprocess.py의 load_raw_rx(배치/pandas 기반)와 동일한 결과를
라인 단위(스트리밍)로 생성한다. VALID_IDX 등 상수는 preprocess.py에서
import하여 중복 정의를 피한다.

라인 포맷 (csi_recv 펌웨어 출력 기준, 클래식 ESP32 타깃):
  field[0]      = 'CSI_DATA'
  field[2]      = seq_id (TX 패킷 payload의 공유 카운터 → 4 RX 간 정렬 기준)
  field[4]      = RSSI (수집 스크립트와 동일하게 0 초과/-100 미만은 폐기)
  field[23]     = len (반드시 384)
  field[26:408] = CSI real/imag 인터리브 382개 (csi_raw[1:383]에 대응)
  field[409]    = 수신 wall-clock timestamp
                  (multi_rx_collect_csi가 저장 시 덧붙임 — 라이브 시리얼
                   라인은 409필드, 녹화 CSV는 410필드. 둘 다 처리 가능)

주의: ESP32-C5/C6 타깃 펌웨어는 메타 필드 수가 달라(15개) 이 파서가
모든 라인을 거부한다. 클래식 ESP32 csi_recv 전용.
"""
import numpy as np

import paths  # noqa: F401  (sys.path 설정)
from preprocess import VALID_IDX, NUM_FEATURES  # LightGBM/preprocess.py

_VALID_IDX = np.asarray(VALID_IDX, dtype=np.int64)


def parse_line(line: str):
    """CSV 라인 1개 → (seq_id, amp(166,) float32) | 손상 라인이면 None."""
    if not line.startswith('CSI_DATA'):
        return None
    parts = line.rstrip('\r\n').split(',')
    if len(parts) < 409:
        return None
    try:
        seq_id = int(parts[2])
        rssi = int(parts[4])
        if rssi > 0 or rssi < -100:     # 수집 스크립트와 동일한 RSSI 필터
            return None
        if int(parts[23]) != 384:       # CSI 길이 384 고정 (다른 포맷 거부)
            return None
        vals = np.array(parts[26:408], dtype=np.float32)   # 382개
    except ValueError:
        return None
    if vals.shape[0] != 382:
        return None

    # load_raw_rx와 동일: csi_raw(384,)의 [1:383]에 배치
    csi = np.zeros(384, dtype=np.float32)
    csi[1:383] = vals
    real = csi[0::2]
    imag = csi[1::2]
    amp = np.sqrt(real ** 2 + imag ** 2)[_VALID_IDX]
    return seq_id, amp.astype(np.float32)


def parse_timestamp(line: str):
    """라인 마지막 필드의 wall-clock timestamp. 없으면 None."""
    try:
        return float(line.rstrip('\r\n').rsplit(',', 1)[-1])
    except ValueError:
        return None


def iter_csv_packets(filepath: str):
    """녹화된 raw CSV → (seq_id, amp(166,), timestamp) 제너레이터."""
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            parsed = parse_line(line)
            if parsed is None:
                continue
            seq_id, amp = parsed
            ts = parse_timestamp(line)
            yield seq_id, amp, ts


# ── 자가 검증: load_raw_rx와 결과 일치 확인 ───────────────────────────────────
if __name__ == '__main__':
    import glob
    import os
    from preprocess import load_raw_rx

    sample = sorted(glob.glob(
        os.path.join(paths.RAW_DATA_DIR, 'kms', 'kms_walk_*_rx1.csv')
    ))[0]
    print(f'[self-check] file: {sample}')

    ref_amp, ref_sid = load_raw_rx(sample)

    amps, sids = [], []
    for sid, amp, _ts in iter_csv_packets(sample):
        sids.append(sid)
        amps.append(amp)
    amps = np.stack(amps)
    sids = np.array(sids, dtype=np.int64)

    assert amps.shape == ref_amp.shape, f'{amps.shape} != {ref_amp.shape}'
    assert np.array_equal(sids, ref_sid), 'seq_id mismatch'
    assert np.allclose(amps, ref_amp, atol=1e-5), 'amplitude mismatch'
    print(f'[self-check] OK - {len(sids)} packets, '
          f'amp shape {amps.shape}, max|diff|={np.abs(amps - ref_amp).max():.2e}')
