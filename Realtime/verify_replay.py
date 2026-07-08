"""
[검증 B] 엔드투엔드 리플레이 vs 오프라인.

kms의 raw CSV 4개를 ReplaySource(speed=0)로 전체 파이프라인
(파싱→동기화→링버퍼→윈도우 전처리→추론)에 통과시키고,
동일 샘플의 NPZ(오프라인 전처리 완료본)에 대한 오프라인 예측과 비교한다.

동기화기의 base_seq 계산이 오프라인 compute_start_seq와 동일하므로
리플레이 슬롯 s == NPZ 그리드 행 s. 윈도우 시작 인덱스로 정렬해 비교한다.

100% 일치는 기대하지 않는다(3σ 통계: 800그리드 vs 200윈도우 근사).
zone 일치율 90% 미만이면 슬롯 정렬부터 조사할 것.

실행: python Realtime/verify_replay.py [--device cpu]
"""
import argparse
import glob
import os

import numpy as np

import paths
from dataset import CSIRawDataset, TEST_SUBJECT  # Zone_Expert
from engine import InferenceEngine
from export_scaler import load_scaler
from main import run_pipeline
from models import ACTION_NAMES
from postprocess import MajorityVoteSmoother, RecordingSink
from sources import ReplaySource
from verify_engine import make_windows, offline_predict_windows

# 행동별 1개씩 검증 (존재하는 position 자동 탐색)
ACTIONS = ['handsup', 'sit', 'stand', 'walk']


def find_position(subject: str, action: str) -> int:
    files = glob.glob(os.path.join(
        paths.RAW_DATA_DIR, subject, f'{subject}_{action}_*_rx1.csv'))
    positions = sorted(int(os.path.basename(f).split('_')[2]) for f in files)
    if not positions:
        raise FileNotFoundError(f'{subject}/{action} raw CSV 없음')
    return positions[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()

    scaler_mean, scaler_scale = load_scaler()
    engine = InferenceEngine(device=args.device)
    test = CSIRawDataset([TEST_SUBJECT])

    # NPZ 인덱스: (action, position) → 샘플 인덱스
    # CSIRawDataset은 position을 보존하지 않으므로 NPZ에서 직접 읽는다
    npz_index = {}
    for f in glob.glob(os.path.join(paths.PROCESSED_DIR, f'{TEST_SUBJECT}_*.npz')):
        z = np.load(f, allow_pickle=True)
        key = (int(z['action']), int(z['position']))
        npz_index[key] = f

    grand_total = grand_zone = grand_action = 0

    for action in ACTIONS:
        pos = find_position(TEST_SUBJECT, action)
        a_idx = ACTIONS.index(action)
        npz_path = npz_index.get((a_idx, pos))
        if npz_path is None:
            print(f'[skip] {action} pos={pos}: NPZ 없음')
            continue

        print(f'\n--- {TEST_SUBJECT} / {action} / pos {pos} ---')

        # 1) 오프라인 기준 예측 (NPZ 그리드 → 61 윈도우)
        z = np.load(npz_path, allow_pickle=True)
        grid = z['grids'].transpose(1, 2, 0).reshape(800, 664).astype(np.float32)
        grid_norm = ((grid - scaler_mean) / scaler_scale).astype(np.float32)
        wins = make_windows(grid_norm)
        z_off, a_off = offline_predict_windows(engine, wins)
        off_by_start = {s * 10: (z_off[s], a_off[s]) for s in range(len(z_off))}

        # 2) 리플레이 파이프라인
        source = ReplaySource.from_sample(
            paths.RAW_DATA_DIR, TEST_SUBJECT, action, pos, speed=0)
        sink = RecordingSink()
        run_pipeline(source, sink, engine=engine,
                     smoother=MajorityVoteSmoother(k=1), verbose=False)

        # 3) 윈도우 시작 인덱스로 정렬 비교
        n = zm = am = 0
        for pred in sink.predictions:
            ref = off_by_start.get(pred.window_start)
            if ref is None:          # 오프라인 800그리드 범위 밖 윈도우
                continue
            n += 1
            zm += int(pred.zone == ref[0])
            am += int(pred.action == ref[1])

        rt_actions = [p.action for p in sink.predictions]
        maj = max(set(rt_actions), key=rt_actions.count) if rt_actions else -1
        print(f'  리플레이 예측 {len(sink.predictions)}개 '
              f'(오프라인과 겹치는 윈도우 {n}개)')
        if n:
            print(f'  zone 일치율   : {zm / n:.4f}')
            print(f'  action 일치율 : {am / n:.4f}')
        print(f'  실시간 다수결 action: {ACTION_NAMES[maj]} (정답: {action})')

        grand_total += n
        grand_zone += zm
        grand_action += am

    print(f'\n{"=" * 55}')
    print(f'[검증 B] 전체 - {grand_total} windows')
    if grand_total:
        zr = grand_zone / grand_total
        ar = grand_action / grand_total
        print(f'  zone 일치율   : {zr:.4f}')
        print(f'  action 일치율 : {ar:.4f}')
        verdict = 'PASS' if zr >= 0.9 else 'CHECK (슬롯 정렬 조사 필요)'
        print(f'  판정: {verdict}')
    print(f'{"=" * 55}')


if __name__ == '__main__':
    main()
