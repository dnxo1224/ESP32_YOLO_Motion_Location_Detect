import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# 재사용 가능한 파싱 함수
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
        
        if len(data_array) % 2 != 0:
            data_array = data_array[:-1]
            
        real_parts = data_array[0::2]
        imag_parts = data_array[1::2]
        
        amplitude = np.sqrt(real_parts**2 + imag_parts**2)
        return list(amplitude)
    except:
        return np.nan

def remove_null_subcarriers(amplitude_array):
    """
    ESP32 CSI 64/192 서브캐리어 포맷에 따른 Null Subcarrier 제거.
    - 64개: (이전 로직)
    - 192개: 사용자 제공 Null 서브캐리어 인덱스 제거
      제거 대상 서브캐리어 번호: 0~5, 32, 59~65, 123~133, 191
    """
    if not isinstance(amplitude_array, list):
        return np.nan
        
    arr = np.array(amplitude_array)
    num_subcarriers = len(arr)
    
    if num_subcarriers == 64:
        valid_idx = list(range(6, 32)) + list(range(33, 59))
        return arr[valid_idx]
    
    elif num_subcarriers == 192:
        # 제거(Null)할 서브캐리어 번호 목록
        null_subcarriers = list(range(0, 6)) + [32] + list(range(59, 66)) + list(range(123, 134)) + [191]
        
        # 192 인덱스(0~191) 중에서 Null이 아닌 인덱스만 필터링
        valid_idx = [i for i in range(192) if i not in null_subcarriers]
        
        return arr[valid_idx]
        
    elif num_subcarriers > 0:
        return arr
    
    return np.nan


def test_null_removal_and_spline():
    base_dir = "/Users/seolwootae/ESP32_YOLO/data"
    subject = "jhj"
    action = "benddown"
    sample_num = "1"
    
    # Rx1 ~ Rx4 병합을 위해 4개 파일 로드 준비
    rx_paths = [os.path.join(base_dir, subject, f"{subject}_{action}_{sample_num}_rx{i}.csv") for i in range(1, 5)]
    
    extracted_dfs = []
    
    for i, file_path in enumerate(rx_paths):
        print(f"\nProcessing: Rx{i+1}")
        df = pd.read_csv(file_path, header=None, encoding='utf-8', on_bad_lines='skip', engine='python')
        
        # 시퀀스 아이디(Col 2)
        df = df.dropna(subset=[2])
        df[2] = df[2].astype(int)
        
        # CSI 시작 컬럼 탐색
        start_col = 25
        for col in df.columns[:50]:
            if df[col].dtype == object and isinstance(df[col].iloc[0], str) and '[' in df[col].iloc[0]:
                start_col = col; break
                
        end_col = df.columns[-2]
        csi_df = df.loc[:, start_col:end_col]
        
        # 1. 진폭 추출
        print("  - Extracting Amplitudes...")
        amplitudes = []
        for _, row in csi_df.iterrows():
            amplitudes.append(extract_csi_amplitude(row))
            
        # 2. 널 서브캐리어 제거
        print("  - Removing Null Subcarriers...")
        cleaned_amps = [remove_null_subcarriers(a) for a in amplitudes]
        
        # 임시 DataFrame
        temp_df = pd.DataFrame({
            'seq_id': df[2],
            f'rx{i+1}_amps': cleaned_amps
        })
        temp_df = temp_df.drop_duplicates(subset=['seq_id'])
        temp_df.set_index('seq_id', inplace=True)
        extracted_dfs.append(temp_df)

    # 3. 데이터 병합 (Outer Join) - Rx1~Rx4 타임라인 동기화
    print("\nMerging and Interpolating...")
    merged_df = extracted_dfs[0].join(extracted_dfs[1:], how='outer')
    merged_df = merged_df.sort_index()
    
    # 결측치 확인
    print("Missing packets before interpolation:")
    print(merged_df.isna().sum())
    
    # 4. Spline Interpolation을 위해 각각의 서브캐리어를 독립된 컬럼으로 분해 후 보간
    # 배열 형태의 열은 보간이 바로 안 되므로 전부 펼침(Flatten)
    rx1_valid_len = len([x for x in merged_df['rx1_amps'] if isinstance(x, np.ndarray)][0])
    print(f"Subcarriers per packet after Null removal: {rx1_valid_len}")
    
    plt.figure(figsize=(12, 6))
    
    # 예시: Rx1의 중간 채널(예: 서브캐리어 인덱스 10) 시계열을 추출하여 Spline 전/후 비교
    target_rx = 'rx1_amps'
    target_subcarrier_idx = 10
    
    # 시계열 벡터 하나 만들기
    time_series_raw = []
    for val in merged_df[target_rx]:
        if isinstance(val, np.ndarray):
            time_series_raw.append(val[target_subcarrier_idx])
        else:
            time_series_raw.append(np.nan)
            
    ts_df = pd.DataFrame({'raw': time_series_raw}, index=merged_df.index)
    
    # Spline 보간
    ts_df['spline'] = ts_df['raw'].interpolate(method='spline', order=3).bfill().ffill()
    
    # 시각화
    plt.plot(ts_df.index, ts_df['spline'], 'g--', label='Spline Imputed Path', alpha=0.7)
    
    # 원래 포인트
    missing_mask = ts_df['raw'].isna()
    plt.plot(ts_df.index[~missing_mask], ts_df['raw'][~missing_mask], 'ko', markersize=3, label='Observed Data')
    
    # 보간된 포인트 X 표기
    plt.plot(ts_df.index[missing_mask], ts_df['spline'][missing_mask], 'rX', markersize=6, label='Imputed (Spline)')
    
    plt.title(f"Null Subcarriers Removed & Spline Interpolation (Rx1, Subcarrier #{target_subcarrier_idx})")
    plt.xlabel("Sequence ID")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = '/Users/seolwootae/ESP32_YOLO/preprocessing/Hampel_LPF_Spline/test_spline_real.png'
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    test_null_removal_and_spline()
