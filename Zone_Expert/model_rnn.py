"""
Zone Expert Action Classifier — Vanilla RNN 모델 모듈

ZoneExpertRNN:
  - Zone에 물리적으로 가까운 RX 채널을 입력 단계에서 강조 (정적 채널 가중치)
  - 단방향 Vanilla RNN 2층 → FC 분류기
  - hidden_dim=300: vanilla RNN은 LSTM 대비 파라미터 효율이 1/4이므로
    hidden size를 늘려 LSTM(hidden=128, ~548K)과 유사한 파라미터 수 유지 (~490K)
"""
import torch
import torch.nn as nn

from dataset import FEATURE_DIM

# ─── Zone → RX 매핑 (0-indexed) ───────────────────────────────────────────────
ZONE_RX_MAP = {0: 0, 1: 1, 2: 2, 3: 3}

RX_HIGH_W = 1.0
RX_LOW_W  = 1.0


class ZoneExpertRNN(nn.Module):
    """
    Zone 특화 단방향 Vanilla RNN 행동 분류기.

    Args:
        zone_id    : 전문가가 담당하는 Zone 번호 (0~3)
        input_dim  : 입력 특징 차원 (기본 664)
        hidden_dim : RNN hidden 크기 (기본 300)
        num_layers : RNN 레이어 수 (기본 2)
        num_classes: 행동 클래스 수 (기본 4)
        dropout    : Dropout 비율 (기본 0.3)
        rx_high_w  : 해당 Zone RX 가중치
        rx_low_w   : 나머지 RX 가중치
    """

    def __init__(self,
                 zone_id:     int,
                 input_dim:   int   = FEATURE_DIM,
                 hidden_dim:  int   = 300,
                 num_layers:  int   = 2,
                 num_classes: int   = 4,
                 dropout:     float = 0.3,
                 rx_high_w:   float = RX_HIGH_W,
                 rx_low_w:    float = RX_LOW_W):
        super().__init__()

        self.zone_id = zone_id

        # ── 고정 RX 채널 가중치 벡터 ──────────────────────────────────────────
        rx_code_idx = ZONE_RX_MAP[zone_id]
        w = torch.full((input_dim,), rx_low_w)
        feature_idx = torch.arange(rx_code_idx, input_dim, 4)
        w[feature_idx] = rx_high_w
        self.register_buffer('rx_weight', w)  # (664,)

        # ── 단방향 Vanilla RNN ────────────────────────────────────────────────
        # nn.RNN은 (output, hn) 반환 — LSTM의 (output, (hn, cn))과 달리 tuple unpack 불필요
        self.rnn = nn.RNN(
            input_size   = input_dim,
            hidden_size  = hidden_dim,
            num_layers   = num_layers,
            batch_first  = True,
            dropout      = dropout if num_layers > 1 else 0.0,
            nonlinearity = 'tanh',
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
                          None        — 추론 시 항상 적용
        Returns:
            logits: (B, num_classes)
        """
        if zone_labels is not None:
            mask = (zone_labels == self.zone_id).float()   # (B,)
            w = mask.view(-1, 1, 1) * (self.rx_weight - 1.0) + 1.0  # (B, 1, 664)
            x = x * w
        else:
            x = x * self.rx_weight                         # (B, T, 664)

        _, hn = self.rnn(x)       # hn: (num_layers, B, hidden_dim)
        out = hn[-1]              # (B, hidden_dim) — 마지막 레이어의 최종 hidden state
        return self.classifier(out)
