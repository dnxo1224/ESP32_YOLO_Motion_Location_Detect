import torch
import torch.nn as nn
import torch.nn.functional as F

class DualStreamTransformer(nn.Module):
    def __init__(self, num_classes=5, num_zones=4, d_model=128, num_heads=4, dim_feedforward=512, dropout=0.1):
        """
        Args:
            num_classes (int): 행동 분류를 위한 최종 클래스 개수 (기본: 5)
            num_zones (int): 위치(구역) 분류를 위한 최종 클래스 개수 (기본: 4)
            d_model (int): Query, Key, Value를 투영할 임베딩 차원 크기
            num_heads (int): Multi-Head Attention의 Head 개수 (논문 기준: 4)
            dim_feedforward (int): FFN(Feed Forward Network)의 은닉층 차원 크기
            dropout (float): 드롭아웃 비율
        """
        super(DualStreamTransformer, self).__init__()
        
        # 1. Temporal Stream (L=800, d=664)를 d_model로 선형 변환
        self.temporal_proj = nn.Linear(664, d_model)
        
        # 2. Channel Stream (d=664, L=800)을 d_model로 선형 변환
        self.channel_proj = nn.Linear(800, d_model)
        
        # 3. Cross-Attention 계층
        # batch_first=True 적용시 파라미터는 (Batch, Seq_len, d_model) 형태를 취함
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 모델의 안정적인 학습을 위한 Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # 4. FFN (Feed Forward Network)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),  # 트랜스포머 구조에서 주로 쓰이는 0이하 값에 스무딩을 주는 활성화 함수
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        
        # 5. Activity Classification (행동 분류) Fully Connected Layer
        self.fc_action = nn.Linear(d_model, num_classes)
        
        # 6. Zone Classification (구역 분류) Fully Connected Layer
        self.fc_zone = nn.Linear(d_model, num_zones)
        
    def forward(self, x):
        """
        Args:
            x (Tensor): [Batch, 800, 664] 형태의 입력 데이터 (Batch, Length, Dimension)
        """
        
        # ========================================
        # 1. Query, Key, Value 생성
        # ========================================
        # [Batch, 800, 664] -> (선형 변환) -> [Batch, 800, d_model]
        temporal_q = self.temporal_proj(x)
        
        # x를 전치(Transpose)하여 Channel Stream 획득: [Batch, 664, 800]
        x_channel = x.transpose(1, 2)
        # [Batch, 664, 800] -> (선형 변환) -> [Batch, 664, d_model]
        channel_kv = self.channel_proj(x_channel)
        
        # ========================================
        # 2. Cross-Attention 메커니즘 연산
        # ========================================
        # Q = temporal_q (시간 관점), K = channel_kv, V = channel_kv (공간/주파수 관점)
        # 출력: attn_output은 (Batch, 800, d_model), attn_weights는 상관관계 맵
        attn_output, attn_weights = self.cross_attention(
            query=temporal_q,
            key=channel_kv,
            value=channel_kv
        )
        
        # Residual Connection (잔차 연결) + Layer Normalization
        x_attn = self.norm1(temporal_q + attn_output)
        
        # ========================================
        # 3. FFN (Feed Forward Network) 연산
        # ========================================
        ffn_output = self.ffn(x_attn)
        
        # Residual Connection + Layer Normalization
        x_out = self.norm2(x_attn + ffn_output)
        
        # ========================================
        # 4. Global Average Pooling (GAP)
        # ========================================
        # 현재 x_out 크기: [Batch, 800, d_model]
        # 시간축(Sequence=800)인 dim=1에 대해 평균 풀링(Average Pooling) 수행
        x_gap = torch.mean(x_out, dim=1)  # [Batch, d_model]
        
        # ========================================
        # 5. 분류 (출력 계층)
        # ========================================
        # 최종 [Batch, 5] 행동 분류
        logits_action = self.fc_action(x_gap)
        
        # 최종 [Batch, 4] 구역 분류
        logits_zone = self.fc_zone(x_gap)
        
        return logits_action, logits_zone, attn_weights
        
        # ========================================
        # 5. 분류 (출력 계층)
        # ========================================
        # 최종 [Batch, 5] 행동 분류
        logits_action = self.fc_action(x_gap)
        
        # 최종 [Batch, 4] 구역 분류
        logits_zone = self.fc_zone(x_gap)
        
        return logits_action, logits_zone, attn_weights

# 모델 동작 테스트
if __name__ == "__main__":
    batch_size = 32
    time_steps = 800
    channels = 664
    num_classes = 5
    num_zones = 4
    
    # 더미 데이터 생성 [Batch, 800, 664]
    mock_x = torch.randn(batch_size, time_steps, channels)
    
    # 모델 인스턴스화 (임베딩 128차원으로 세팅)
    model = DualStreamTransformer(num_classes=num_classes, num_zones=num_zones, d_model=128, num_heads=4)
    
    logits_act, logits_zone, attn_weights = model(mock_x)
    
    print("입력 텐서 형태:", mock_x.shape)
    print("행동 분류 출력 텐서 형태:", logits_act.shape)
    print("구역 분류 출력 텐서 형태:", logits_zone.shape)
    print("Cross-Attention 가중치 텐서 형태:", attn_weights.shape)
