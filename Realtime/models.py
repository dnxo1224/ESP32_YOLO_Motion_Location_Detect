"""
실시간 추론용 모델 로더.

- ZoneMLP  : Zone_Expert/inference.py:82-103에서 복사 (inference.py는
             import 시 matplotlib/seaborn을 끌고 오므로 클래스만 가져옴).
             load_state_dict(strict=True)가 구조 일치를 보증한다.
- ZoneExpertCNN : Zone_Expert/model_cnn.py에서 import (cnn_norx_noos,
             use_rx_weight=False), zone별 4개.
"""
import torch
import torch.nn as nn

import paths
from dataset import FEATURE_DIM          # Zone_Expert
from model_cnn import ZoneExpertCNN      # Zone_Expert

NUM_ZONES    = 4
ACTION_NAMES = ['handsup', 'sit', 'stand', 'walk']
ZONE_NAMES   = [f'Zone {i}' for i in range(NUM_ZONES)]


class ZoneMLP(nn.Module):
    """Zone_Expert/inference.py의 ZoneMLP (mean_only)와 동일 구조."""

    def __init__(self, input_dim: int = FEATURE_DIM, num_classes: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 664) → 시간축 평균 → (B, 664) → MLP
        return self.net(x.mean(dim=1))


def load_models(device: torch.device):
    """(zone_mlp, {zone: expert}) — 모두 eval 모드."""
    zone_mlp = ZoneMLP().to(device)
    zone_mlp.load_state_dict(
        torch.load(paths.ZONE_MLP_WEIGHTS, map_location=device)
    )
    zone_mlp.eval()

    experts = {}
    for z in range(NUM_ZONES):
        m = ZoneExpertCNN(zone_id=z, use_rx_weight=False).to(device)
        ckpt = f'{paths.EXPERT_WEIGHTS_DIR}/zone_expert_action_{z}_best.pt'
        m.load_state_dict(torch.load(ckpt, map_location=device))
        m.eval()
        experts[z] = m

    return zone_mlp, experts
