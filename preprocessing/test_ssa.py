
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# --- 1. 데이터 파싱 함수 (기존과 동일) ---
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

# --- 2. MAE 보간 알고리즘 (Moving Average) ---
def mae_impute(series, window=10):
    """
    이동 평균(Moving Average) 기반 결측치 보간
    (결측치를 중심으로 양방향 이동평균을 구하여 채워 넣음)
    """
    # 1. Forward 이동 평균
    forward_ma = series.rolling(window=window, min_periods=1).mean()
    # 2. Backward 이동 평균
    backward_ma = series[::-1].rolling(window=window, min_periods=1).mean()[::-1]
    
    # 두 방향의 평균을 결합 (NaN을 채우는데 보다 안정적)
    combined_ma = (forward_ma + backward_ma) / 2
    
    # 그래도 비어 있는 처음과 끝단은 선형 보간/bfill/ffill로 방어
    combined_ma = combined_ma.interpolate(method='linear').bfill().ffill()
    
    # 기존 데이터에서 결측치 부분만 이동 평균 값으로 대치
    filled_series = series.copy()
    mask = filled_series.isna()
    filled_series[mask] = combined_ma[mask]
    
    return filled_series

def test_ssa_vs_spline():
    base_dir = r"c:\Users\User\ESP32_YOLO_Motion_Location_Detect\data_aligned_800"
    
    # 여러 파일 중 결측치가 어느 정도 존재하는 파일을 선택합니다.
    # 예: swt_benddown_1_rx2_800.csv 등 (결측이 있을만한 데이터셋 임의 지정)
    file_name = "kmh_benddown_4_rx2_800.csv"
    file_path = os.path.join(base_dir, file_name)
    
    print(f"Loading {file_path}...")
    df = pd.read_csv(file_path, index_col='seq_id')
    
    # 특정 서브캐리어(예: rx2_sub_10) 시계열 추출 (800프레임 중 300프레임만 확인)
    target_col = 'rx2_sub_10'
    
    # 800 시퀀스 중 변동을 잘 볼 수 있는 구간 선택
    # 주의: 이 구간 안에 실제 결측치(NaN)가 하나라도 존재해야 보간법 비교가 가능합니다!
    ts_df = pd.DataFrame({'Damaged': df[target_col]}).iloc[300:600].copy()
    
    print("Applying Spline Interpolation...")
    ts_df['Spline'] = ts_df['Damaged'].copy()
    # 양 끝이 NaN이면 Spline이 처리를 못할 수 있으므로, 최소한의 선형/bfill/ffill 보정
    ts_df['Spline'] = ts_df['Spline'].interpolate(method='spline', order=3).bfill().ffill()
    
    print("Applying MAE Interpolation...")
    ts_df['MAE'] = mae_impute(ts_df['Damaged'].copy(), window=20)
    
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
    plt.title("Spline Interpolation (Local Curve Fitting) on Real Missing Packets")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. MAE 결과 비교
    plt.subplot(2, 1, 2)
    plt.plot(ts_df.index, ts_df['Damaged'], 'ko', alpha=0.5, label='Actual Received Packets (True Signal)')
    plt.plot(ts_df.index, ts_df['MAE'], 'b--', label='MAE Imputed Data (Moving Average)', linewidth=2)
    # 실제 결측 부위 표시
    for idx in ts_df.index[missing_mask]:
        plt.axvline(idx, color='blue', alpha=0.1, linewidth=2)
    plt.title("MAE Interpolation (Moving Average Smoothing) on Real Missing Packets")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = r'c:\Users\User\ESP32_YOLO_Motion_Location_Detect\preprocessing\test_ssa_comparison.png'
    plt.savefig(save_path)
    print(f"Plot successfully saved to {save_path}")

if __name__ == "__main__":
    test_ssa_vs_spline()