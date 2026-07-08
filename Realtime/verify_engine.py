"""
[검증 A] 추론 엔진 정확성.

kms(테스트 피험자) NPZ를 저장된 스케일러로 정규화 → 오프라인과 동일한
슬라이딩 윈도우(200, stride=10) 생성 후,

  1) inference.py와 동일한 배치 추론 (기준)
  2) InferenceEngine 윈도우 단건 추론 (실시간 경로)

두 경로의 zone/action 예측이 100% 일치해야 한다.

실행: python Realtime/verify_engine.py [--max-samples N] [--device cpu]
"""
import argparse

import numpy as np
import torch

import paths
from dataset import CSIRawDataset, FEATURE_DIM, TEST_SUBJECT  # Zone_Expert
from engine import InferenceEngine
from export_scaler import load_scaler

WINDOW_SIZE = 200
STRIDE = 10


def make_windows(seq: np.ndarray):
    """(800, 664) → (n_win, 200, 664), 시작 0,10,...,600 (inference.py와 동일)"""
    wins = []
    start = 0
    while start + WINDOW_SIZE <= seq.shape[0]:
        wins.append(seq[start:start + WINDOW_SIZE])
        start += STRIDE
    return np.stack(wins).astype(np.float32)


@torch.no_grad()
def offline_predict_windows(engine: InferenceEngine, windows: np.ndarray):
    """inference.py 루프와 동일한 배치 추론 (기준 경로).

    반환: (zone_preds(n,), action_preds(n,))
    """
    device = engine.device
    x = torch.from_numpy(windows).to(device)              # (n, 200, 664)
    zone_preds = engine.zone_mlp(x).argmax(1)             # (n,)
    x_mr = x - x.mean(dim=1, keepdim=True)
    action_preds = torch.zeros(x.size(0), dtype=torch.long, device=device)
    for z in range(4):
        mask = (zone_preds == z)
        if mask.sum() == 0:
            continue
        action_preds[mask] = engine.experts[z](x_mr[mask]).argmax(1)
    return zone_preds.cpu().numpy(), action_preds.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-samples', type=int, default=0,
                    help='0 = kms 전체 샘플')
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()

    scaler_mean, scaler_scale = load_scaler()
    engine = InferenceEngine(device=args.device)

    test = CSIRawDataset([TEST_SUBJECT])
    n_samples = len(test)
    if args.max_samples > 0:
        n_samples = min(n_samples, args.max_samples)

    total = 0
    zone_match = 0
    action_match = 0
    zone_correct = 0
    action_correct = 0
    latencies = []

    for i in range(n_samples):
        seq = (test.data[i] - scaler_mean) / scaler_scale   # 저장 스케일러 적용
        seq = seq.astype(np.float32)
        wins = make_windows(seq)

        z_ref, a_ref = offline_predict_windows(engine, wins)

        for w in range(wins.shape[0]):
            pred = engine.predict(wins[w])
            latencies.append(pred.latency_ms)
            total += 1
            zone_match += int(pred.zone == z_ref[w])
            action_match += int(pred.action == a_ref[w])
            zone_correct += int(pred.zone == test.zone_labels[i])
            action_correct += int(pred.action == test.action_labels[i])

        if (i + 1) % 16 == 0:
            print(f'  {i + 1}/{n_samples} samples ...')

    print(f'\n{"=" * 55}')
    print(f'[검증 A] 엔진 vs 오프라인 배치 - {n_samples} samples, {total} windows')
    print(f'{"=" * 55}')
    print(f'  zone   일치율 : {zone_match / total:.4f}  (기대: 1.0000)')
    print(f'  action 일치율 : {action_match / total:.4f}  (기대: 1.0000)')
    print(f'  (참고) 정답 대비 zone acc   : {zone_correct / total:.4f}')
    print(f'  (참고) 정답 대비 action acc : {action_correct / total:.4f}')
    print(f'  단건 추론 시간: median {np.median(latencies):.1f}ms / '
          f'p95 {np.percentile(latencies, 95):.1f}ms '
          f'(예산 333ms)')

    if zone_match == total and action_match == total:
        print('\n[검증 A] PASS - 실시간 엔진과 오프라인 경로 완전 일치')
    else:
        print('\n[검증 A] FAIL - 불일치 발생, 엔진/모델 로딩 점검 필요')
        raise SystemExit(1)


if __name__ == '__main__':
    main()
