"""
Zone Expert Action Classifier — CNN 모델 모듈

ZoneExpertCNN:
  - Zone에 물리적으로 가까운 RX 채널을 입력 단계에서 강조 (정적 채널 가중치) - 현재는 X
  - 1D CNN (시간축 방향) → AdaptiveAvgPool → FC 분류기
"""
import torch
import torch.nn as nn

from dataset import FEATURE_DIM

# ─── Zone → RX 매핑 (0-indexed) ───────────────────────────────────────────────
ZONE_RX_MAP = {0: 0, 1: 1, 2: 2, 3: 3}

RX_HIGH_W = 1.0
RX_LOW_W  = 1.0


class ZoneExpertCNN(nn.Module):
    """
    Zone 특화 1D CNN 행동 분류기.

    Args:
        zone_id       : 전문가가 담당하는 Zone 번호 (0~3)
        input_dim     : 입력 특징 차원 (기본 664)
        num_classes   : 행동 클래스 수 (기본 4)
        dropout       : Dropout 비율 (기본 0.3)
        use_rx_weight : True  → Zone별 RX 채널 가중치 적용
                        False → 가중치 없이 모든 RX 채널 동등하게 처리
        rx_high_w     : 해당 Zone RX 가중치 (use_rx_weight=True 일 때만 유효)
        rx_low_w      : 나머지 RX 가중치   (use_rx_weight=True 일 때만 유효)
    """

    def __init__(self,
                 zone_id:        int,
                 input_dim:      int   = FEATURE_DIM,
                 num_classes:    int   = 4,
                 dropout:        float = 0.3,
                 use_rx_weight:  bool  = True,
                 rx_high_w:      float = RX_HIGH_W,
                 rx_low_w:       float = RX_LOW_W):
        super().__init__()

        self.zone_id       = zone_id
        self.use_rx_weight = use_rx_weight

        # ── 고정 RX 채널 가중치 벡터 ──────────────────────────────────────────
        if use_rx_weight:
            rx_code_idx = ZONE_RX_MAP[zone_id]
            w = torch.full((input_dim,), rx_low_w)
            feature_idx = torch.arange(rx_code_idx, input_dim, 4)
            w[feature_idx] = rx_high_w
            self.register_buffer('rx_weight', w)  # (664,)
        # use_rx_weight=False 일 때는 rx_weight 버퍼 자체를 생성하지 않음

        # ── 1D CNN (Conv1d는 (B, C, L) 입력 기대) ────────────────────────────
        # Block 1~2: 664 채널 → 128 채널
        self.conv_blocks = nn.Sequential(
            # Block 1
            nn.Conv1d(input_dim, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            # Block 2
            nn.Conv1d(128, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            # Pooling: 200 → 50
            nn.MaxPool1d(kernel_size=4, stride=4),
            # Block 3
            nn.Conv1d(128, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            # Block 4
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            # 전체 시간축 풀링: 50 → 1
            nn.AdaptiveAvgPool1d(1),
        )

        # ── 분류기 ────────────────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, 64),
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
                          (use_rx_weight=False 이면 무시)
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

        # Conv1d 입력 형식: (B, C, L) → permute
        x = x.permute(0, 2, 1)                             # (B, 664, T)
        x = self.conv_blocks(x)                            # (B, 128, 1)
        x = x.squeeze(-1)                                  # (B, 128)
        return self.classifier(x)                          # (B, num_classes)
