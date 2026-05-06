"""
Zone Expert Action Classifier — 모델 모듈

ZoneExpertLSTM:
  - Zone에 물리적으로 가까운 RX 채널을 입력 단계에서 강조 (정적 채널 가중치)
  - 단방향 LSTM 2층 → FC 분류기
"""
import torch
import torch.nn as nn

from dataset import FEATURE_DIM

# ─── Zone → RX 매핑 (0-indexed) ───────────────────────────────────────────────
# 사용자 확인: Zone 0→RX1(코드RX0), Zone 1→RX2(코드RX1),
#              Zone 2→RX3(코드RX2), Zone 3→RX4(코드RX3)
ZONE_RX_MAP = {0: 0, 1: 1, 2: 2, 3: 3}

# RX 채널 가중치 기본값
RX_HIGH_W = 1.0   # 해당 Zone RX 강조
RX_LOW_W  = 1.0   # 나머지 RX 약화


class ZoneExpertLSTM(nn.Module):
    """
    Zone 특화 단방향 LSTM 행동 분류기.

    Args:
        zone_id   : 전문가가 담당하는 Zone 번호 (0~3)
        input_dim : 입력 특징 차원 (기본 664)
        hidden_dim: LSTM hidden 크기 (기본 128)
        num_layers: LSTM 레이어 수 (기본 2)
        num_classes: 행동 클래스 수 (기본 4)
        dropout   : Dropout 비율 (기본 0.3)
        rx_high_w : 해당 Zone RX 가중치 (기본 2.0)
        rx_low_w  : 나머지 RX 가중치 (기본 0.5)
    """

    def __init__(self,
                 zone_id:       int,
                 input_dim:     int   = FEATURE_DIM,
                 hidden_dim:    int   = 128,
                 num_layers:    int   = 2,
                 num_classes:   int   = 4,
                 dropout:       float = 0.3,
                 use_rx_weight: bool  = True,
                 rx_high_w:     float = RX_HIGH_W,
                 rx_low_w:      float = RX_LOW_W):
        super().__init__()

        self.zone_id       = zone_id
        self.use_rx_weight = use_rx_weight

        # ── 고정 RX 채널 가중치 벡터 ──────────────────────────────────────────
        if use_rx_weight:
            rx_code_idx = ZONE_RX_MAP[zone_id]
            w = torch.full((input_dim,), rx_low_w)
            feature_idx = torch.arange(rx_code_idx, input_dim, 4)
            w[feature_idx] = rx_high_w
            self.register_buffer('rx_weight', w)  # (664,) — 학습 파라미터 아님

        # ── 단방향 LSTM ───────────────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size  = input_dim,
            hidden_size = hidden_dim,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
            bidirectional = False,
        )

        # ── 분류기 ────────────────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor,
                zone_labels: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x           : (B, T, 664) — 슬라이딩 윈도우 CSI 시퀀스
            zone_labels : (B,) long   — 학습 시 전달, 해당 Zone 샘플만 rx_weight 적용
                          None        — 추론 시 (Zone ZONE_ID 데이터만 들어오므로 항상 적용)
        Returns:
            logits: (B, num_classes)
        """
        if self.use_rx_weight:
            if zone_labels is not None:
                mask = (zone_labels == self.zone_id).float()             # (B,)
                w = mask.view(-1, 1, 1) * (self.rx_weight - 1.0) + 1.0  # (B, 1, 664)
                x = x * w
            else:
                x = x * self.rx_weight                                   # (B, T, 664)
        # use_rx_weight=False → x 그대로 (모든 RX 채널 동등)

        _, (hn, _) = self.lstm(x)
        out = hn[-1]                                       # (B, hidden_dim)
        return self.classifier(out)                        # (B, num_classes)
