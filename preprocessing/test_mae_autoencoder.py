import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# --- 1. 데이터 파싱 함수 (이전과 동일) ---
def extract_csi_amplitude(row_data):
    try:
        data_list = list(row_data)
        if isinstance(data_list[0], str):
            clean = data_list[0].replace('"""[', '').replace('"[', '').strip()
            data_list[0] = int(clean) if clean else 0
        if isinstance(data_list[-1], str):
            clean = data_list[-1].replace(']"""', '').replace(']"', '').strip()
            data_list[-1] = int(clean) if clean else 0
            
        cleaned = [float(x) for x in data_list if str(x).replace('-','').replace('.','').isdigit()]
        arr = np.array(cleaned)
        if len(arr) % 2 != 0: arr = arr[:-1]
        
        real, imag = arr[0::2], arr[1::2]
        return list(np.sqrt(real**2 + imag**2))
    except:
        return np.nan

def remove_null_subcarriers(amplitude_array):
    if not isinstance(amplitude_array, list): return np.nan
    arr = np.array(amplitude_array)
    if len(arr) == 192:
        null_subcarriers = list(range(0, 6)) + [32] + list(range(59, 66)) + list(range(123, 134)) + [191]
        valid_idx = [i for i in range(192) if i not in null_subcarriers]
        return arr[valid_idx]
    return np.nan

# --- 2. MAE (Masked Autoencoder) 모델 정의 ---
class MaskedAutoencoder1D(nn.Module):
    def __init__(self, seq_len=800, hidden_dim=256):
        super(MaskedAutoencoder1D, self).__init__()
        
        # 간단한 MLP 기반 Encoder-Decoder 구조
        self.encoder = nn.Sequential(
            nn.Linear(seq_len, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, seq_len)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def train_mae_imputer(series, epochs=2000, lr=0.005, mask_ratio=0.15):
    """
    1D 시계열 전용 Masked Autoencoder 학습 및 추론
    """
    # 원본 길이 기록
    orig_index = series.index
    seq_len = len(series)
    
    # 훈련을 위해 현재의 결측치는 선형보간 및 앞/뒤 복사로 완전히 평탄화(초기화)
    # 모델은 이 초기 평탄화된 데이터를 토대로 전체적인 패턴(주기성, 흐름)을 학습하게 됨
    filled_array = series.interpolate(method='linear').bfill().ffill().values
    
    # Min-Max Scaling (NN 학습 속도/안정성 확보)
    min_val, max_val = filled_array.min(), filled_array.max()
    if max_val - min_val > 0:
        scaled_array = (filled_array - min_val) / (max_val - min_val)
    else:
        scaled_array = filled_array
        
    tensor_data = torch.FloatTensor(scaled_array).unsqueeze(0) # Shape: (1, seq_len)
    
    model = MaskedAutoencoder1D(seq_len=seq_len, hidden_dim=256)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    model.train()
    # 실제 수신된(결측아닌) 인덱스들
    valid_mask = ~series.isna().values
    valid_indices = np.where(valid_mask)[0]
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # --- Artificial Masking (학습용) ---
        # 실제 데이터가 있는 구간 중 랜덤하게 일부를 가림(Mask)
        mask = torch.ones_like(tensor_data)
        
        # valid_indices 중에서 mask_ratio 만큼 무작위 선택하여 0으로 만듦
        if len(valid_indices) > 0:
            num_mask = int(len(valid_indices) * mask_ratio)
            masked_idx = np.random.choice(valid_indices, size=num_mask, replace=False)
            mask[0, masked_idx] = 0
        
        # 마스킹 적용 (가려진 곳은 0, 엄밀히는 패치 단위 마스킹이 좋으나 여기선 포인트 레벨 마스킹)
        masked_input = tensor_data * mask

        # 모델 예측
        reconstructed = model(masked_input)
        
        # Loss는 원래 '유효했던 데이터' 위치에서만 계산 (실제 진짜 NaN이었던 구간은 Loss에서 제외)
        loss = criterion(reconstructed[0, valid_indices], tensor_data[0, valid_indices])
        
        loss.backward()
        optimizer.step()
        
    # --- 추론 (Imputation) ---
    model.eval()
    with torch.no_grad():
        # 처음 NaN이었던 부분들을 선형보간한 입력값을 바탕으로 전체 복원 시도
        reconstructed_output = model(tensor_data).squeeze(0).numpy()
        
    # 역정규화 (Inverse Transform)
    if max_val - min_val > 0:
        reconstructed_rescaled = reconstructed_output * (max_val - min_val) + min_val
    else:
        reconstructed_rescaled = reconstructed_output
        
    # 최종 결측치만 모델의 예측값으로 덮어씌움
    final_output = series.copy()
    nan_mask = series.isna()
    final_output[nan_mask] = reconstructed_rescaled[nan_mask]
    
    return final_output


def test_mae_autoencoder_vs_spline():
    base_dir = r"c:\Users\User\ESP32_YOLO_Motion_Location_Detect\data_aligned_800"
    
    # 패킷 로스가 많았던 데이터 파일
    file_name = "kmh_benddown_4_rx2_800.csv"
    file_path = os.path.join(base_dir, file_name)
    
    print(f"Loading {file_path}...")
    df = pd.read_csv(file_path, index_col='seq_id')
    
    # 특정 서브캐리어 시계열 추출 (800프레임 중 300프레임 확인)
    target_col = 'rx2_sub_10'
    ts_df = pd.DataFrame({'Damaged': df[target_col]}).iloc[300:600].copy()
    
    print("Applying Spline Interpolation...")
    ts_df['Spline'] = ts_df['Damaged'].copy()
    ts_df['Spline'] = ts_df['Spline'].interpolate(method='spline', order=3).bfill().ffill()
    
    print("Applying Masked Autoencoder (MAE) Neural Net Interpolation...")
    ts_df['MAE_Autoencoder'] = train_mae_imputer(ts_df['Damaged'].copy(), epochs=800, lr=0.01)
    
    # --- 시각화 ---
    plt.figure(figsize=(14, 8))
    
    missing_mask = ts_df['Damaged'].isna()
    
    # 1. Spline 결과 비교
    plt.subplot(2, 1, 1)
    plt.plot(ts_df.index, ts_df['Damaged'], 'ko', alpha=0.5, label='Actual Received Packets (True Signal)')
    plt.plot(ts_df.index, ts_df['Spline'], 'g--', label='Spline Imputed Data', linewidth=2)
    # 실제 결측 부위 표시
    for idx in ts_df.index[missing_mask]:
        plt.axvline(idx, color='red', alpha=0.1, linewidth=2)
    plt.title("Spline Interpolation on Heavy Missing Drops")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. MAE Autoencoder 결과 비교
    plt.subplot(2, 1, 2)
    plt.plot(ts_df.index, ts_df['Damaged'], 'ko', alpha=0.5, label='Actual Received Packets (True Signal)')
    plt.plot(ts_df.index, ts_df['MAE_Autoencoder'], 'b--', label='Masked Autoencoder Imputation', linewidth=2)
    # 실제 결측 부위 표시
    for idx in ts_df.index[missing_mask]:
        plt.axvline(idx, color='blue', alpha=0.1, linewidth=2)
    plt.title("Masked Autoencoder (Neural Pattern Reconstructor) on Heavy Missing Drops")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = r'c:\Users\User\ESP32_YOLO_Motion_Location_Detect\preprocessing\test_mae_autoencoder_comparison.png'
    plt.savefig(save_path)
    print(f"Plot successfully saved to {save_path}")

if __name__ == "__main__":
    test_mae_autoencoder_vs_spline()
