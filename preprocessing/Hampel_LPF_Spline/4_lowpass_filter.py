import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# --- 이전에 작성한 함수들 (1, 2, 3 단계 분) 간략화 포함 ---
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
    if num_subcarriers > 0:
        edge_cut = int(num_subcarriers * 0.1)
        valid_idx = list(range(edge_cut, num_subcarriers - edge_cut))
        return arr[valid_idx]
    return np.nan

def butter_lowpass_filter(data, cutoff, fs, order=3):
    """
    Scipy의 Butterworth Low-pass Filter 적용
    - data: 1차원 시계열 데이터 배열
    - cutoff: 차단 주파수 (Hz) - 인간의 움직임은 보통 1~5Hz 내외
    - fs: 샘플링 주파수 (Hz) - 여기서는 30Hz
    - order: 필터 차수 (클수록 더 예리한 컷오프)
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    # Apply filter forward and backward to avoid phase shift (filtfilt)
    y = filtfilt(b, a, data)
    return y

def test_lowpass_filter():
    base_dir = "/Users/seolwootae/ESP32_YOLO/data"
    subject = "jhj"
    action = "benddown"
    sample_num = "1"
    
    # Rx1 파일 1개만 샘플 테스트 (빠른 테스트)
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
    
    # 1 & 2. 진폭 추출 및 널 캐리어 제거
    cleaned_amps = []
    for _, row in csi_df.iterrows():
        amp = extract_csi_amplitude(row)
        cleaned_amps.append(remove_null_subcarriers(amp))
        
    temp_df = pd.DataFrame({'seq_id': df[2], 'rx1_amps': cleaned_amps})
    temp_df = temp_df.drop_duplicates(subset=['seq_id'])
    temp_df.set_index('seq_id', inplace=True)
    
    # 하나의 서브캐리어(예: 인덱스 10) 시계열을 가져와 테스트
    target_idx = 10
    time_series_raw = [val[target_idx] if isinstance(val, np.ndarray) else np.nan for val in temp_df['rx1_amps']]
    ts_df = pd.DataFrame({'raw': time_series_raw}, index=temp_df.index)
    
    # 3. 보간 (필터링 전 비어있는 값이 있으면 filtfilt에서 에러 발생)
    ts_df['spline'] = ts_df['raw'].interpolate(method='spline', order=3).bfill().ffill()
    
    # 4. Low-pass Filter 적용 (Cutoff 3Hz, fs 30Hz)
    fs = 30.0       # 측정 샘플링 레이트
    cutoff = 3.0    # 컷오프 주파수 설정 (보통 제스처/행동은 3~5Hz)
    
    filtered_data = butter_lowpass_filter(ts_df['spline'].values, cutoff, fs, order=4)
    ts_df['filtered'] = filtered_data
    
    # 시각화 비교
    plt.figure(figsize=(14, 6))
    
    plt.plot(ts_df.index, ts_df['spline'], 'k-', alpha=0.3, label='Before LPF (Spline only)')
    plt.plot(ts_df.index, ts_df['filtered'], 'r-', linewidth=2, label=f'After LPF (Cutoff={cutoff}Hz)')
    
    plt.title(f"Low-pass Filter Smoothing (Rx1, Subcarrier #{target_idx})")
    plt.xlabel("Sequence ID (Time)")
    plt.ylabel("CSI Amplitude")
    plt.legend()
    plt.grid(True, alpha=0.4)
    
    save_path = '/Users/seolwootae/ESP32_YOLO/preprocessing/Hampel_LPF_Spline/test_lpf.png'
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    test_lowpass_filter()
