"""
WiDR 모델 — Wi-Fi Dual-task Recognition

논문: "Wi-Fi CSI-Based Human Activity Recognition and Indoor Localization
       with Sampling Irregularity Mitigation" (Lee & Toh, IEEE IoT Journal 2025)

구현 범위:
  - Neural CDE (Section III-A, Eq.1-3)
  - Dual-Stream Cross-Attention (Section III-B, Eq.4, 12-15)
  - Parameter-Shared Dual-Task Learning (Section III-C, Eq.6-8)
"""
import torch
import torch.nn as nn

try:
    import torchcde
    HAS_TORCHCDE = True
except ImportError:
    HAS_TORCHCDE = False


# ============================================================
# Component 1: Neural CDE (논문 Section III-A, Eq.1-3)
# ============================================================

class CDEFunc(nn.Module):
    """
    CDE 벡터장 fθ: R^w → R^(w × (d+1))   (논문 Eq.2)

    dz/dt = fθ(z_t) · dX/dt
    torchcde.cdeint 는 forward(t, z) 시그니처를 요구함.

    Args:
        hidden_dim: 은닉 노드 수 w (논문 최적: 400)
        input_dim:  진폭 채널 수 d (114, 시간 채널 제외)
    """
    def __init__(self, hidden_dim: int, input_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        # 2-layer MLP with tanh (GUIDE Section 3)
        self.linear1 = nn.Linear(hidden_dim, 128)
        self.linear2 = nn.Linear(128, hidden_dim * (input_dim + 1))

    def forward(self, t, z):
        """
        Args:
            t: 시간 스칼라 (torchcde 인터페이스, 사용 안 함)
            z: (..., w) 은닉 상태
        Returns:
            (..., w, d+1) 벡터장 행렬
        """
        z = self.linear1(z).tanh()
        z = self.linear2(z)                                           # (..., w*(d+1))
        return z.view(*z.shape[:-1], self.hidden_dim, self.input_dim + 1)


class WiCDE(nn.Module):
    """
    Neural CDE 모듈 (논문 Section III-A)

    불규칙 샘플링된 CSI 시퀀스를 연속 경로로 변환하여
    균일 그리드 출력 X̂를 생성.

    Input:  (B, L, d+1) — 진폭[0:d] + 정규화된 시간 채널[d]
            NaN는 누락 패킷을 표시 (torchcde가 보간 처리)
    Output: (B, L, d)   — 복원된 진폭 시퀀스 X̂

    Args:
        input_dim:  진폭 채널 d = 114
        hidden_dim: CDE 은닉 노드 w (논문 최적: 400)
        l_fixed:    출력 시퀀스 길이 (= 입력 L = 872)
    """
    def __init__(self, input_dim: int = 114, hidden_dim: int = 400, l_fixed: int = 872):
        super().__init__()
        if not HAS_TORCHCDE:
            raise ImportError("torchcde가 설치되어 있지 않습니다.\n"
                              "실행: pip install torchcde")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.l_fixed = l_fixed

        # fθ: 벡터장 (Eq.2)
        self.cde_func = CDEFunc(hidden_dim=hidden_dim, input_dim=input_dim)

        # ζθ: 초기 은닉 상태 매핑  R^(d+1) → R^w  (Eq.1 초기값)
        self.initial_state = nn.Linear(input_dim + 1, hidden_dim)

        # ℓθ: 출력 복원  R^w → R^d  (Eq.3)
        self.output_map = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, d+1) — 마지막 채널이 상대 시간 (i/30.0)
        Returns:
            x_hat: (B, L, d) — CDE 복원 진폭
        """
        # 시간 그리드 추출 (배치 내 동일하므로 첫 번째 샘플 사용)
        t = x[0, :, self.input_dim]  # (L,)

        # Natural Cubic Spline 계수 계산 (NaN → 자동 보간)
        coeffs = torchcde.natural_cubic_spline_coeffs(x, t)
        X = torchcde.CubicSpline(coeffs)

        # 초기 은닉 상태 ζθ(t0, x0)
        z0 = self.initial_state(X.evaluate(X.interval[0]))  # (B, w)

        # CDE 적분: z_t = z_t0 + ∫ fθ(z_s) dX(s)  (Eq.1)
        z_T = torchcde.cdeint(X=X, func=self.cde_func, z0=z0, t=t)  # (B, L, w)

        # 출력 복원 x̂_t = ℓθ(z_t)  (Eq.3)
        return self.output_map(z_T)  # (B, L, d)


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
        fc_hidden:  분류 FC 은닉 차원 (논문: 200)
        dropout:    드롭아웃 비율
        use_cde:    True → WiCDE 사용 (RawCSIDataset 입력 필요)
        cde_hidden: CDE 은닉 노드 w (논문 최적: 400)
    """
    def __init__(self, input_dim=456, seq_len=872, d_model=128, num_heads=4, d_ffn=256,
                 num_actions=4, num_zones=4, fc_hidden=200, dropout=0.1,
                 use_cde=False, cde_hidden=400):
        super().__init__()
        self.use_cde = use_cde

        # Neural CDE (논문 Section III-A) — 두 브랜치 공유, 1개만 생성
        if use_cde:
            # input_dim=114: 단일 RX 서브캐리어 수 (4 RX concat은 CDE 통과 후 수행)
            self.cde = WiCDE(input_dim=114, hidden_dim=cde_hidden, l_fixed=seq_len)

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
            x: (B, 872, 456)          — use_cde=False
               (B, 4, 872, 115)       — use_cde=True  (4 RX, 진폭+시간채널)
        Returns:
            action_logits: (B, num_actions)
            zone_logits:   (B, num_zones)
        """
        if self.use_cde:
            # x: (B, 4, 872, 115)
            B = x.shape[0]
            # 4 RX를 배치 차원으로 묶어 WiCDE 병렬 처리
            x_rx = x.view(B * 4, x.shape[2], x.shape[3])   # (B*4, 872, 115)
            x_hat_rx = self.cde(x_rx)                        # (B*4, 872, 114)
            # RX 차원 복원 후 feature 차원으로 concat
            x_hat = x_hat_rx.view(B, 4, x_hat_rx.shape[1], x_hat_rx.shape[2])
            x = x_hat.permute(0, 2, 1, 3).reshape(B, x_hat_rx.shape[1], 456)
            # x: (B, 872, 456)

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
