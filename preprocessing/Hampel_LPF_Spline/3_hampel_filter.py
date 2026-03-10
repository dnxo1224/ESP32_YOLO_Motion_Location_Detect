import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# --- 재사용 컴포넌트 간략화 ---
def extract_csi_amplitude(row_data):
    try:
        data_list = list(row_data)
        if isinstance(data_list[0], str):
            clean_str = data_list[0].replace('"""[', '').replace('"[', '').strip()
            data_list[0] = int(clean_str) if clean_str else 0
        if isinstance(data_list[-1], str):
            clean_str = data_list[-1].replace(']"""', '').replace(']"', '').strip()
            data_list[-1] = int(clean_str) if clean_str else 0
            
        cleaned_data = [float(x) for x in data_list if str(x).replace('-','').replace('.','').isdigit()]
        data_array = np.array(cleaned_data)
        if len(data_array) % 2 != 0: data_array = data_array[:-1]
            
        real_parts = data_array[0::2]
        imag_parts = data_array[1::2]
        return list(np.sqrt(real_parts**2 + imag_parts**2))
    except: return np.nan

def remove_null_subcarriers(amplitude_array):
    if not isinstance(amplitude_array, list): return np.nan
    arr = np.array(amplitude_array)
    num_subcarriers = len(arr)
    if num_subcarriers == 192:
        null_subcarriers = list(range(0, 6)) + [32] + list(range(59, 66)) + list(range(123, 134)) + [191]
        valid_idx = [i for i in range(192) if i not in null_subcarriers]
        return arr[valid_idx]
    return np.nan

def hampel_filter(data_series, window_size=5, n_sigmas=3):
    """
    Hampel Filter (이상치 탐지 및 제거)
    - data_series: 1차원 시계열 데이터
    - window_size: 양방향 윈도우 크기 (5면 앞뒤 5개씩, 총 11개)
    - n_sigmas: 몇 배의 절대편차를 벗어나면 이상치로 간주할지 설정 (보통 3)
    반환: 튀는 값이 제거(NaN) 혹은 주변 중앙값(Median)으로 치환된 시계열
    """
    new_series = data_series.copy()
    
    # pandas Rolling Median 활용
    # center=True 로 설정하여 앞뒤 데이터를 균형있게 참조
    rolling_median = data_series.rolling(window=2*window_size+1, center=True).median()
    
    # MAD (Median Absolute Deviation) 계수 (보통 1.4826을 곱하여 표준편차에 근사)
    k = 1.4826 
    
    rolling_mad = k * data_series.rolling(window=2*window_size+1, center=True).apply(lambda x: np.median(np.abs(x - np.median(x))))
    
    # 경계값(Threshold) 계산
    outlier_idx = np.abs(data_series - rolling_median) > (n_sigmas * rolling_mad)
    
    # 이상치를 발견하면 해당 윈도우의 중앙값(Median)으로 치환
    new_series[outlier_idx] = rolling_median[outlier_idx]
    
    return new_series, outlier_idx


def test_hampel_filter():
    base_dir = "/Users/seolwootae/ESP32_YOLO/data"
    subject = "jhj"
    action = "benddown"
    sample_num = "1"
    
    # Rx1 파일 로드
    file_path = os.path.join(base_dir, subject, f"{subject}_{action}_{sample_num}_rx1.csv")
    print(f"\nProcessing: Rx1")
    df = pd.read_csv(file_path, header=None, encoding='utf-8', on_bad_lines='skip', engine='python')
    df = df.dropna(subset=[2])
    df[2] = df[2].astype(int)
    
    start_col = 25
    for col in df.columns[:50]:
        if df[col].dtype == object and isinstance(df[col].iloc[0], str) and '[' in df[col].iloc[0]:
            start_col = col; break
            
    end_col = df.columns[-2]
    csi_df = df.loc[:, start_col:end_col]
    
    cleaned_amps = []
    for _, row in csi_df.iterrows():
        amp = extract_csi_amplitude(row)
        cleaned_amps.append(remove_null_subcarriers(amp))
        
    temp_df = pd.DataFrame({'seq_id': df[2], 'rx1_amps': cleaned_amps})
    temp_df = temp_df.drop_duplicates(subset=['seq_id'])
    temp_df.set_index('seq_id', inplace=True)
    
    # 서브캐리어 하나 추출 (테스트용 인덱스 10)
    target_idx = 10
    time_series_raw = [val[target_idx] if isinstance(val, np.ndarray) else np.nan for val in temp_df['rx1_amps']]
    ts_df = pd.DataFrame({'raw': time_series_raw}, index=temp_df.index)
    
    # 일부러 튀는 이상치(Spike) 데이터를 생성 (테스트용)
    ts_df['raw'].iloc[150] = ts_df['raw'].iloc[150] + 500
    ts_df['raw'].iloc[250] = ts_df['raw'].iloc[250] - 400
    ts_df['raw'].iloc[350] = ts_df['raw'].iloc[350] + 600
    
    # 앞선 과정대로 보간으로 NaN 제거 (Hampel은 NaN이 있으면 에러나므로)
    ts_df['interpolated'] = ts_df['raw'].interpolate(method='spline', order=3).bfill().ffill()
    
    # Hampel Filter 적용 (Window=5, Sigma=3)
    filtered_series, outliers = hampel_filter(ts_df['interpolated'], window_size=5, n_sigmas=3)
    ts_df['hampel'] = filtered_series
    
    outlier_count = outliers.sum()
    print(f"Hampel filter detected {outlier_count} outliers (Spikes/Noise).")
    
    # 시각화 비교
    plt.figure(figsize=(14, 6))
    
    plt.plot(ts_df.index, ts_df['interpolated'], 'k-', alpha=0.4, label='Original with Spikes')
    plt.plot(ts_df.index[outliers], ts_df['interpolated'][outliers], 'ro', markersize=6, label='Detected Outliers')
    plt.plot(ts_df.index, ts_df['hampel'], 'b-', linewidth=2, label='After Hampel Filter')
    
    plt.title(f"Hampel Filter Outlier Removal (Rx1, Subcarrier #{target_idx})")
    plt.xlabel("Sequence ID (Time)")
    plt.ylabel("CSI Amplitude")
    plt.legend()
    plt.grid(True, alpha=0.4)
    
    save_path = '/Users/seolwootae/ESP32_YOLO/preprocessing/Hampel_LPF_Spline/test_hampel.png'
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    test_hampel_filter()
