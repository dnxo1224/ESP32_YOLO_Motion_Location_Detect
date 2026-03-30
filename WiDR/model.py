"""
WiDR 모델 — Wi-Fi Dual-task Recognition

논문: "Wi-Fi CSI-Based Human Activity Recognition and Indoor Localization
       with Sampling Irregularity Mitigation" (Lee & Toh, IEEE IoT Journal 2025)

구현 범위:
  - Dual-Stream Cross-Attention (Section III-B, Eq.4, 12-15)
  - Parameter-Shared Dual-Task Learning (Section III-C, Eq.6-8)
  - Neural CDE (Section III-A)는 생략 (데이터 이미 보간 완료)
"""
import torch
import torch.nn as nn


class DualStreamCrossAttention(nn.Module):
    """
    논문 Section III-B + III-C의 단일 태스크 브랜치

    Temporal stream (Q) × Channel stream (K, V) → Cross-Attention → MHA → FFN → GAP

    Args:
        input_dim: 입력 feature 차원 (456 = 114 subcarriers × 4 RX)
        seq_len: 시퀀스 길이 (872)
        d_model: 내부 임베딩 차원 (128)
        num_heads: Multi-Head Attention head 수 (4, 논문 최적)
        d_ffn: Feed-Forward Network 은닉 차원 (256, 논문 최적)
        dropout: 드롭아웃 비율
    """
    def __init__(self, input_dim=456, seq_len=872, d_model=128, num_heads=4, d_ffn=256, dropout=0.1):
        super().__init__()

        # Temporal Stream: (B, L, d) → (B, L, d_model)
        # 논문 Eq.12: Q^CA = X̂ W_q^CA
        self.temporal_proj = nn.Linear(input_dim, d_model)

        # Channel Stream: (B, d, L) → (B, d, d_model)
        # 논문 Eq.12: K^CA = X̂^T W_k^CA, V^CA = X̂^T W_v^CA
        self.channel_proj = nn.Linear(seq_len, d_model)

        # Cross-Attention (논문 Eq.4, 12)
        # Q from temporal stream, K/V from channel stream
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm_ca = nn.LayerNorm(d_model)

        # Multi-Head Self-Attention (논문 Eq.13-14)
        # Z^CA를 입력으로 self-attention 수행
        self.self_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm_mha = nn.LayerNorm(d_model)

        # Feed-Forward Network (논문 Eq.15)
        # Z^FFN = GELU(Z^MHA W1 + b1) W2 + b2
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, d_model),
            nn.Dropout(dropout),
        )
        self.norm_ffn = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        Args:
            x: (B, 872, 456) — 입력 CSI 시퀀스
        Returns:
            (B, d_model) — GAP 후 특징 벡터
        """
        # --- Temporal Stream ---
        # (B, 872, 456) → (B, 872, 128)
        temporal_q = self.temporal_proj(x)

        # --- Channel Stream ---
        # (B, 872, 456) → transpose → (B, 456, 872) → (B, 456, 128)
        x_channel = x.transpose(1, 2)
        channel_kv = self.channel_proj(x_channel)

        # --- Cross-Attention (논문 Eq.4, 12) ---
        # Q = temporal, K = V = channel
        ca_out, _ = self.cross_attention(
            query=temporal_q,
            key=channel_kv,
            value=channel_kv
        )
        # Residual + LayerNorm
        ca_out = self.norm_ca(temporal_q + ca_out)  # (B, 872, 128)

        # --- Multi-Head Self-Attention (논문 Eq.13-14) ---
        mha_out, _ = self.self_attention(
            query=ca_out,
            key=ca_out,
            value=ca_out
        )
        # Residual + LayerNorm
        mha_out = self.norm_mha(ca_out + mha_out)  # (B, 872, 128)

        # --- FFN (논문 Eq.15) ---
        ffn_out = self.ffn(mha_out)
        # Residual + LayerNorm
        ffn_out = self.norm_ffn(mha_out + ffn_out)  # (B, 872, 128)

        # --- Global Average Pooling ---
        out = ffn_out.mean(dim=1)  # (B, 128)

        return out


class WiDRNet(nn.Module):
    """
    WiDR 전체 모델 — Dual-Task Recognition

    논문 Section III-C (Fig. 1):
    두 개의 동일 구조 브랜치가 각자의 파라미터(Θ1, Θ2)를 학습하고,
    Parameter Sharing Regularization (Eq.6)으로 연결됨.

    Args:
        input_dim: 입력 feature 차원 (456)
        seq_len: 시퀀스 길이 (872)
        d_model: 내부 임베딩 차원 (128)
        num_heads: MHA head 수 (4)
        d_ffn: FFN 은닉 차원 (256)
        num_actions: 행동 클래스 수 (4)
        num_zones: 구역 클래스 수 (4)
        fc_hidden: 분류 FC 은닉 차원 (논문: 200, 여기서는 128 사용)
        dropout: 드롭아웃 비율
    """
    def __init__(self, input_dim=456, seq_len=872, d_model=128, num_heads=4, d_ffn=256,
                 num_actions=4, num_zones=4, fc_hidden=128, dropout=0.1):
        super().__init__()

        # Branch 1: Action Recognition (Task 1)
        # 논문 Eq.9-11: CA → MHA → FFN with parameter set Θ1
        self.encoder_action = DualStreamCrossAttention(
            input_dim=input_dim, seq_len=seq_len,
            d_model=d_model, num_heads=num_heads, d_ffn=d_ffn, dropout=dropout
        )
        # 논문 Eq.16: Ŷ1 = softmax(Z^FFN_(1) W^FC_(1) + b^FC_(1))
        self.classifier_action = nn.Sequential(
            nn.Linear(d_model, fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, num_actions),
        )

        # Branch 2: Zone Classification (Task 2)
        # 동일 구조, 별도 파라미터 Θ2
        self.encoder_zone = DualStreamCrossAttention(
            input_dim=input_dim, seq_len=seq_len,
            d_model=d_model, num_heads=num_heads, d_ffn=d_ffn, dropout=dropout
        )
        # 논문 Eq.16: Ŷ2 = softmax(Z^FFN_(2) W^FC_(2) + b^FC_(2))
        self.classifier_zone = nn.Sequential(
            nn.Linear(d_model, fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, num_zones),
        )

    def forward(self, x):
        """
        Args:
            x: (B, 872, 456)
        Returns:
            action_logits: (B, num_actions)
            zone_logits: (B, num_zones)
        """
        # Branch 1
        feat_action = self.encoder_action(x)       # (B, d_model)
        action_logits = self.classifier_action(feat_action)  # (B, 4)

        # Branch 2
        feat_zone = self.encoder_zone(x)            # (B, d_model)
        zone_logits = self.classifier_zone(feat_zone)        # (B, 4)

        return action_logits, zone_logits

    def param_reg_loss(self):
        """
        Parameter Sharing Regularization (논문 Eq.6)

        L_reg = ||Θ1 - Θ2||²

        Θ1 = encoder_action + classifier_action의 파라미터
        Θ2 = encoder_zone + classifier_zone의 파라미터
        """
        loss = torch.tensor(0.0, device=next(self.parameters()).device)

        # Encoder 파라미터 비교 (CA + MHA + FFN)
        for p1, p2 in zip(self.encoder_action.parameters(), self.encoder_zone.parameters()):
            loss = loss + torch.sum((p1 - p2) ** 2)

        # Classifier 파라미터 비교 (FC layers)
        for p1, p2 in zip(self.classifier_action.parameters(), self.classifier_zone.parameters()):
            loss = loss + torch.sum((p1 - p2) ** 2)

        return loss


# --- 모델 동작 테스트 ---
if __name__ == "__main__":
    device = torch.device('cpu')
    model = WiDRNet().to(device)

    # 파라미터 수 출력
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Dummy forward pass
    x = torch.randn(2, 872, 456).to(device)
    action_logits, zone_logits = model(x)
    print(f"Input: {x.shape}")
    print(f"Action logits: {action_logits.shape}")
    print(f"Zone logits: {zone_logits.shape}")

    # Regularization loss
    reg_loss = model.param_reg_loss()
    print(f"Param reg loss: {reg_loss.item():.4f}")
