import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ==========================================
# 1. 1D Transformer 기반 Masked Autoencoder (MAE) 모델
# ==========================================
class TransformerMAE(nn.Module):
    def __init__(self, seq_len=800, embed_dim=128, num_heads=4, num_layers=3, dropout=0.1):
        super(TransformerMAE, self).__init__()
        self.seq_len = seq_len
        
        # 입력 차원 매핑 (1 -> embed_dim)
        self.input_proj = nn.Linear(1, embed_dim)
        
        # Positional Encoding (학습 가능한 파라미터로 처리)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, embed_dim))
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 출력 차원 매핑 (embed_dim -> 1)
        self.output_proj = nn.Linear(embed_dim, 1)

    def forward(self, x):
        # x shape: (batch, seq_len, 1)
        x = self.input_proj(x)
        x = x + self.pos_embed # 위치 정보 더하기
        
        encoded = self.encoder(x)
        decoded = self.output_proj(encoded)
        
        # decoded shape: (batch, seq_len, 1)
        return decoded

# ==========================================
# 2. 커스텀 Dataset (서브캐리어 하나가 1개의 샘플)
# ==========================================
class CSIDataset(Dataset):
    def __init__(self, data_matrix):
        """
        data_matrix: (num_samples, seq_len) 형태의 numpy array
        (예: num_samples = 총 서브캐리어 개수, seq_len = 800)
        """
        self.data = torch.FloatTensor(data_matrix)
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        # Transformer는 (seq_len, feature_dim) 입력을 받으므로 차원 추가
        return self.data[idx].unsqueeze(-1)

# ==========================================
# 3. 보간 및 학습 파이프라인
# ==========================================
def train_target_model(df_rx, epochs=200, lr=1e-3, mask_ratio=0.15):
    """
    하나의 CSV 파일 (1개의 RX 데이터) 내에 있는 166개의 서브캐리어를
    배치(Batch)로 삼아 이 파일만의 특징을 학습하고 보간합니다.
    """
    seq_len = len(df_rx)
    sub_cols = [c for c in df_rx.columns if '_sub_' in c]
    
    # 모델 입력용으로 데이터 정규화 및 전처리
    raw_array = df_rx[sub_cols].values.T  # (166, 800)
    
    # NaN이 존재하는 위치 기록 (최종 복구용)
    nan_mask = np.isnan(raw_array)
    
    # 딥러닝 텐서 주입 전, 임시로 선형 보간하여 구멍 메우기 (모델 초기 상태 안정화)
    temp_df = pd.DataFrame(raw_array).interpolate(axis=1, limit_direction='both')
    imputed_array = temp_df.values
    
    # Min-Max Scaling (샘플 별 독립적)
    mins = imputed_array.min(axis=1, keepdims=True)
    maxs = imputed_array.max(axis=1, keepdims=True)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0 # 0나누기 방지
    scaled_array = (imputed_array - mins) / ranges
    
    # Dataloader 생성
    dataset = CSIDataset(scaled_array)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 디바이스 및 모델 세팅
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransformerMAE(seq_len=seq_len, embed_dim=64, num_heads=4, num_layers=2).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    model.train()
    
    # --- Training Loop ---
    for epoch in range(epochs):
        for batch_data in dataloader:
            batch_data = batch_data.to(device)  # (batch, seq_len, 1)
            batch_size = batch_data.size(0)
            
            optimizer.zero_grad()
            
            # --- Masking 전략 ---
            # 훈련할 때, 임의로 데이터를 가리고(Masking -> 0) 원본을 맞추게 함.
            # (BERT의 MLM 방식과 유사)
            mask = torch.ones_like(batch_data)
            num_mask = int(seq_len * mask_ratio)
            
            for i in range(batch_size):
                # 각 샘플별로 랜덤하게 마스킹 인덱스 선정
                mask_idx = np.random.choice(seq_len, num_mask, replace=False)
                mask[i, mask_idx, 0] = 0
                
            masked_input = batch_data * mask
            
            # 예측 및 역전파
            output = model(masked_input)
            loss = criterion(output, batch_data)
            
            loss.backward()
            optimizer.step()
            
    # --- 임퍼테이션 (추론) ---
    model.eval()
    all_data_tensor = torch.FloatTensor(scaled_array).unsqueeze(-1).to(device)
    
    with torch.no_grad():
        reconstructed = model(all_data_tensor).squeeze(-1).cpu().numpy()
        
    # 복원값 스케일 되돌리기
    reconstructed = (reconstructed * ranges) + mins
    
    # 원본 배열 복사 후, 처음에 진짜 NaN이었던 부분만 Transformer의 예측값으로 덮어씀
    final_array = raw_array.copy()
    final_array[nan_mask] = reconstructed[nan_mask]
    
    # DataFrame으로 돌려놓기
    result_df = df_rx.copy()
    result_df[sub_cols] = final_array.T
    
    return result_df


# ==========================================
# 4. 전체 파일 대상 배치 프로세싱
# ==========================================
def process_all_mae_transformer():
    input_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_aligned_800'))
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_interpolated_tf_mae_800'))
    os.makedirs(output_dir, exist_ok=True)
    
    csv_files = glob.glob(os.path.join(input_dir, "*_800.csv"))
    total_files = len(csv_files)
    
    print(f"총 {total_files}개의 RX 데이터에 대해 [Transformer 기반 Masked Autoencoder] 보간을 시작합니다.")
    print("GPU 사용 유무:", torch.cuda.is_available())
    print("=" * 60)
    
    success_cnt = 0
    for idx, file_path in enumerate(csv_files):
        filename = os.path.basename(file_path)
        out_path = os.path.join(output_dir, filename)
        
        # 이미 처리된 파일이면 건너뛰기 (이어하기 지원)
        if os.path.exists(out_path):
            success_cnt += 1
            continue
            
        print(f"[{idx+1}/{total_files}] 처리중: {filename} ...", end="", flush=True)
        try:
            df = pd.read_csv(file_path, index_col='seq_id')
            
            # 결측치가 하나라도 존재하는 파일만 딥러닝 모델 학습 및 보간 수행
            if df.isna().sum().sum() > 0:
                df = train_target_model(df, epochs=150, lr=0.002, mask_ratio=0.15)
            
            df.to_csv(out_path)
            print(" 완료")
            success_cnt += 1
            
        except Exception as e:
            print(f" 오류 발생! ({e})")
            
    print("=" * 60)
    print(f"모든 AI 보간 작업 완료! (성공: {success_cnt}/{total_files})")
    print(f"결과 위치: {output_dir}")

if __name__ == "__main__":
    process_all_mae_transformer()
