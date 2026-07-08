"""
실시간 CSI 위치·행동 분류 파이프라인 — 메인 엔트리.

  시리얼/리플레이 → 파싱 → seq_id 동기화 → 링버퍼(200) →
  [10프레임마다] 윈도우 전처리 → ZoneMLP → 평균 제거 → ZoneExpertCNN → 콘솔

실행 예:
  라이브   : python Realtime/main.py --mode live --ports COM3 COM4 COM5 COM6
  리플레이 : python Realtime/main.py --mode replay --subject kms --action walk --position 5
  전속력   : ... --speed 0   (검증용, 페이싱 없음)
"""
import argparse
import json
import queue
import time

import numpy as np

import paths
from engine import InferenceEngine
from export_scaler import load_scaler
from postprocess import ConsoleSink, MajorityVoteSmoother
from ring_buffer import FrameRingBuffer
from sources import ReplaySource, SerialSource
from synchronizer import FrameSynchronizer
from window_prep import preprocess_window

WINDOW_SIZE = 200
STRIDE = 10


def run_pipeline(source, sink, engine=None, smoother=None,
                 stride: int = STRIDE, device: str = 'cpu',
                 stats_interval: float = 5.0, verbose: bool = True):
    """소스 → 예측 루프. 소스 종료(None 센티널) 또는 Ctrl+C까지 실행."""
    scaler_mean, scaler_scale = load_scaler()
    if engine is None:
        engine = InferenceEngine(device=device)
    if smoother is None:
        smoother = MajorityVoteSmoother(k=5)

    pkt_q = queue.Queue(maxsize=8192)
    sync = FrameSynchronizer()
    ring = FrameRingBuffer(WINDOW_SIZE)

    n_frames = 0
    n_preds = 0
    last_stats = time.monotonic()

    def handle_frames(frames):
        nonlocal n_frames, n_preds
        for fr in frames:
            ring.push(fr.data)
            n_frames += 1
            # 윈도우 시작이 stride 배수가 되는 시점에 예측
            # (오프라인 슬라이딩 윈도우 start=0,10,20,...과 정렬)
            if ring.full() and (n_frames - WINDOW_SIZE) % stride == 0:
                raw_win = ring.snapshot()
                nan_ratio = float(np.isnan(raw_win).mean())
                win = preprocess_window(raw_win, scaler_mean, scaler_scale)
                pred = engine.predict(win)
                pred.window_start = n_frames - WINDOW_SIZE
                pred.nan_ratio = nan_ratio
                smoothed = smoother.update(pred)
                sink.emit(pred, smoothed)
                n_preds += 1

    source.start(pkt_q)
    if verbose:
        print('[pipeline] 시작 - 워밍업(50 seq) + 버퍼(200프레임) 후 첫 예측')

    try:
        while True:
            # 패킷 수신 여부와 무관하게 주기적으로 상태 출력
            # (수신 0인 상태를 조용히 지나치지 않도록)
            now = time.monotonic()
            if verbose and now - last_stats > stats_interval:
                last_stats = now
                health = source.health() if hasattr(source, 'health') else ''
                print(f'[stats] frames={n_frames} preds={n_preds} '
                      f'{sync.stats()} {health}')

            try:
                pkt = pkt_q.get(timeout=0.5)
            except queue.Empty:
                handle_frames(sync.poll())     # 타임아웃 슬롯 방출
                if getattr(source, 'finished', None) and source.finished.is_set():
                    break
                continue

            if pkt is None:                    # 리플레이 종료 센티널
                handle_frames(sync.flush())
                break

            handle_frames(sync.push(pkt.rx, pkt.seq_id, pkt.amp, pkt.t_recv))
    except KeyboardInterrupt:
        print('\n[pipeline] 중단됨 (Ctrl+C)')
    finally:
        source.stop()

    if verbose:
        print(f'[pipeline] 종료 - frames={n_frames} preds={n_preds} | {sync.stats()}')
    return n_frames, n_preds


def build_source(args):
    if args.mode == 'live':
        ports = args.ports
        baud = args.baud
        if args.config:
            with open(args.config, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
            ports = cfg.get('ports', ports)
            baud = cfg.get('baud', baud)
        if not ports or not (1 <= len(ports) <= 4):
            raise SystemExit('--ports COM7 COM10 COM11 COM9 형식 (1-RX 테스트는 '
                             '--ports COM7 --mirror)')
        return SerialSource(ports, baud=baud, mirror=args.mirror)

    return ReplaySource.from_sample(
        paths.RAW_DATA_DIR, args.subject, args.action, args.position,
        speed=args.speed, mirror=args.mirror,
    )


def main():
    ap = argparse.ArgumentParser(description='실시간 CSI zone+action 분류')
    ap.add_argument('--mode', choices=['live', 'replay'], default='replay')
    ap.add_argument('--ports', nargs='+', default=None,
                    help='라이브: COM 포트 4개 (rx1~rx4 순서)')
    ap.add_argument('--baud', type=int, default=2_000_000,
                    help='수집 스크립트(multi_rx_collect_csi_v4)와 동일: 2000000')
    ap.add_argument('--config', default=None, help='JSON 설정 파일 (ports/baud)')
    ap.add_argument('--subject', default='kms')
    ap.add_argument('--action', default='walk')
    ap.add_argument('--position', type=int, default=5)
    ap.add_argument('--speed', type=float, default=1.0,
                    help='리플레이 배속 (0 = 페이싱 없이 전속력)')
    ap.add_argument('--mirror', action='store_true',
                    help='1-RX 동작 테스트: RX 1개의 패킷을 4채널에 복제 '
                         '(예측값은 의미 없음, 배관 검증 전용)')
    ap.add_argument('--smooth-k', type=int, default=5)
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()

    source = build_source(args)
    run_pipeline(
        source,
        sink=ConsoleSink(),
        smoother=MajorityVoteSmoother(k=args.smooth_k),
        device=args.device,
    )


if __name__ == '__main__':
    main()
