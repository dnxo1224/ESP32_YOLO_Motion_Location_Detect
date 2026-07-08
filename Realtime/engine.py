"""
추론 엔진: 전처리 완료된 (200, 664) 윈도우 → zone + action 예측.

오프라인 inference.py의 루프와 동일한 순서:
  1. ZoneMLP(원본 윈도우, 평균 제거 없음) → zone
  2. 윈도우 평균 제거 (x - x.mean(dim=1))
  3. ZoneExpertCNN[zone] → action
"""
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from models import load_models


@dataclass
class Prediction:
    zone: int
    zone_conf: float
    action: int
    action_conf: float
    zone_probs: np.ndarray = field(repr=False)
    action_probs: np.ndarray = field(repr=False)
    window_start: int = -1        # 윈도우 시작 프레임(슬롯) 인덱스
    nan_ratio: float = 0.0        # 전처리 전 윈도우의 NaN 비율
    latency_ms: float = 0.0       # 추론 소요 시간
    t_wall: float = 0.0           # 예측 시각 (time.monotonic)


class InferenceEngine:
    def __init__(self, device: str = 'cpu'):
        self.device = torch.device(device)
        self.zone_mlp, self.experts = load_models(self.device)

    @torch.no_grad()
    def predict(self, window: np.ndarray) -> Prediction:
        """window: (200, 664) float32, 정규화 완료 상태."""
        t0 = time.perf_counter()

        x = torch.from_numpy(np.ascontiguousarray(window)) \
                 .float().unsqueeze(0).to(self.device)      # (1, 200, 664)

        zone_logits = self.zone_mlp(x)
        zone_probs = torch.softmax(zone_logits, dim=1)[0]
        zone = int(zone_probs.argmax())

        x_mr = x - x.mean(dim=1, keepdim=True)              # 윈도우 평균 제거
        action_logits = self.experts[zone](x_mr)
        action_probs = torch.softmax(action_logits, dim=1)[0]
        action = int(action_probs.argmax())

        latency = (time.perf_counter() - t0) * 1000

        return Prediction(
            zone=zone,
            zone_conf=float(zone_probs[zone]),
            action=action,
            action_conf=float(action_probs[action]),
            zone_probs=zone_probs.cpu().numpy(),
            action_probs=action_probs.cpu().numpy(),
            latency_ms=latency,
            t_wall=time.monotonic(),
        )
