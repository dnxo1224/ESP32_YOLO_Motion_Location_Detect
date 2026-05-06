"""
Zone Expert Action Classifier — Transformer 모델 모듈

ZoneExpertTransformer:
  - Zone에 물리적으로 가까운 RX 채널을 입력 단계에서 강조 (정적 채널 가중치)
  - Linear Projection (664 → 128) + 학습 가능한 Positional Encoding
  - TransformerEncoder (3층, d_model=128, nhead=4, dim_ff=256) → 시간축 Mean Pooling
  - FC 분류기 (128 → 64 → 4)

파라미터 수: ~517K  (LSTM ~548K 대비 유사한 규모)
"""
import math
import torch
import torch.nn as nn

from dataset import FEATURE_DIM

# ─── Zone → RX 매핑 (0-indexed) ───────────────────────────────────────────────
ZONE_RX_MAP = {0: 0, 1: 1, 2: 2, 3: 3}

RX_HIGH_W = 1.0
RX_LOW_W  = 1.0


class ZoneExpertTransformer(nn.Module):
    """
    Zone 특화 Transformer Encoder 행동 분류기.

    Args:
        zone_id      : 전문가가 담당하는 Zone 번호 (0~3)
        input_dim    : 입력 특징 차원 (기본 664)
        d_model      : Transformer 내부 임베딩 차원 (기본 128)
        nhead        : Multi-head Attention 헤드 수 (기본 4)
        num_layers   : TransformerEncoder 층 수 (기본 3)
        dim_feedforward : FFN 내부 차원 (기본 256)
        dropout      : Dropout 비율 (기본 0.1)
        num_classes  : 행동 클래스 수 (기본 4)
        max_seq_len  : 최대 시퀀스 길이 — Positional Encoding 크기 (기본 200)
        rx_high_w    : 해당 Zone RX 가중치
        rx_low_w     : 나머지 RX 가중치
    """

    def __init__(self,
                 zone_id:         int,
                 input_dim:       int   = FEATURE_DIM,
                 d_model:         int   = 128,
                 nhead:           int   = 4,
                 num_layers:      int   = 3,
                 dim_feedforward: int   = 256,
                 dropout:         float = 0.1,
                 num_classes:     int   = 4,
                 max_seq_len:     int   = 200,
                 use_rx_weight:   bool  = True,
                 rx_high_w:       float = RX_HIGH_W,
                 rx_low_w:        float = RX_LOW_W):
        super().__init__()

        self.zone_id       = zone_id
        self.d_model       = d_model
        self.use_rx_weight = use_rx_weight

        # ── 고정 RX 채널 가중치 벡터 ──────────────────────────────────────────
        if use_rx_weight:
            rx_code_idx = ZONE_RX_MAP[zone_id]
            w = torch.full((input_dim,), rx_low_w)
            feature_idx = torch.arange(rx_code_idx, input_dim, 4)
            w[feature_idx] = rx_high_w
            self.register_buffer('rx_weight', w)  # (664,)

        # ── 입력 선형 투영 (664 → d_model) ───────────────────────────────────
        self.input_proj = nn.Linear(input_dim, d_model)

        # ── 학습 가능한 Positional Encoding ──────────────────────────────────
        # 고정 sinusoidal 대신 학습 가능한 임베딩 사용 (고정 길이 시퀀스에 유리)
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, max_seq_len, d_model)
        )
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        # ── Transformer Encoder ───────────────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = nhead,
            dim_feedforward = dim_feedforward,
            dropout         = dropout,
            batch_first     = True,   # (B, T, d_model) 형식
            norm_first      = True,   # Pre-LN: 학습 안정성 향상 (GPT-2 스타일)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers = num_layers,
        )

        # ── 분류기 ────────────────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            nn.GELU(),
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
        # ── RX 채널 가중치 ────────────────────────────────────────────────────
        if self.use_rx_weight:
            if zone_labels is not None:
                mask = (zone_labels == self.zone_id).float()             # (B,)
                w = mask.view(-1, 1, 1) * (self.rx_weight - 1.0) + 1.0  # (B, 1, 664)
                x = x * w
            else:
                x = x * self.rx_weight                                   # (B, T, 664)
        # use_rx_weight=False → x 그대로 (모든 RX 채널 동등)

        # ── 입력 투영 + Positional Encoding ──────────────────────────────────
        x = self.input_proj(x)                                # (B, T, d_model)
        x = x + self.pos_embedding[:, :x.size(1), :]         # (B, T, d_model)

        # ── Transformer Encoder ───────────────────────────────────────────────
        x = self.transformer(x)                               # (B, T, d_model)

        # ── 시간축 Mean Pooling → 분류 ────────────────────────────────────────
        x = x.mean(dim=1)                                     # (B, d_model)
        return self.classifier(x)                             # (B, num_classes)
