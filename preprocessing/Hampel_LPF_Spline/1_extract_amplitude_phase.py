import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

def extract_csi_amplitude_phase(row_data):
    """
    esp32 csi 데이터를 복소수(Real, Imaginary)로 파싱하여 
    Amplitude(진폭)와 Phase(위상) 배열로 반환합니다.
    esp32 서브캐리어는 64개이며 (Real, Imag) 쌍으로 총 128개의 정수값이 들어있습니다.
    (간혹 더 긴 MAC 헤더 포함 구조가 있으나 실질적인 CSI는 128 혹은 그 이상의 짝수배 배열입니다)
    """
    try:
        # 추출된 데이터 시리즈를 리스트로 변환
        data_list = list(row_data)
        
        # 첫 번째 원소에 혹시 남아있는 """[ 문자가 있다면 숫자화
        if isinstance(data_list[0], str):
            clean_str = data_list[0].replace('"""[', '').replace('"[', '').strip()
            data_list[0] = int(clean_str) if clean_str else 0
            
        # 마지막 원소에 혹시 남아있는 ]""" 문자가 있다면 숫자화
        if isinstance(data_list[-1], str):
            clean_str = data_list[-1].replace(']"""', '').replace(']"', '').strip()
            data_list[-1] = int(clean_str) if clean_str else 0
            
        # 모든 값을 float로 캐스팅 (혹시나 하는 에러 대비)
        cleaned_data = [float(x) for x in data_list if str(x).replace('-','').replace('.','').isdigit()]
        data_array = np.array(cleaned_data)
        
        # 짝수개가 아니면 마지막 혹은 앞을 버림
        if len(data_array) % 2 != 0:
            data_array = data_array[:-1]
            
        real_parts = data_array[0::2]
        imag_parts = data_array[1::2]
        
        # 진폭(Amplitude) 및 위상(Phase) 계산
        amplitude = np.sqrt(real_parts**2 + imag_parts**2)
        phase = np.arctan2(imag_parts, real_parts)
        
        return amplitude, phase

    except Exception as e:
        print(f"Parse error: {e}")
        return np.nan, np.nan


def process_sample_file(file_path):
    """
    단일 CSV 파일을 읽어서 Amplitude와 Phase를 추출하는 테스트 함수
    """
    print(f"Processing: {os.path.basename(file_path)}")
    # CSV 파일을 읽어올 때 줄바꿈 등으로 인한 에러 무시
    df = pd.read_csv(file_path, header=None, encoding='utf-8', on_bad_lines='skip', engine='python')
    
    # 3번째 열 (Index 2) 가 시퀀스 ID
    df = df.dropna(subset=[2])
    
    # 데이터 구조: 0~24번 인덱스 근처까지 메타데이터
    # 25번(대략) 컬럼부터 문자열 """[0 등으로 시작함
    # 뒤에서 두 번째 컬럼(Timestamp) 직전까지가 CSI 밸류
    
    start_col = -1
    for col in df.columns[:50]:
        if df[col].dtype == object and isinstance(df[col].iloc[0], str) and '[' in df[col].iloc[0]:
            start_col = col
            break
            
    if start_col == -1:
        # 만약 '[ ]' 가 없다면 보통 25번째 열부터 CSI 시작
        start_col = 25
        
    print(f"  - Assuming CSI data starts at column index: {start_col}")
    
    # 마지막 열은 Timestamp
    end_col = df.columns[-2]
    
    csi_df = df.loc[:, start_col:end_col]
    
    # 각 행(row)을 numpy array 추출기에 던짐
    amplitudes = []
    phases = []
    
    for _, row in csi_df.iterrows():
        amp, phs = extract_csi_amplitude_phase(row)
        amplitudes.append(amp)
        phases.append(phs)
        
    df['amplitude'] = amplitudes
    df['phase'] = phases
    
    # 첫 번째 유효한 패킷 시각화
    for i, amp in enumerate(amplitudes):
        if amp is not np.nan and len(amp) > 10:
            plt.figure(figsize=(10, 4))
            plt.plot(amp, marker='o', linestyle='-', color='b', markersize=3)
            plt.title(f"Extracted Amplitude for Packet (Subcarriers: {len(amp)})")
            plt.xlabel("Subcarrier Index")
            plt.ylabel("Amplitude")
            plt.grid(True, alpha=0.3)
            
            save_path = '/Users/seolwootae/ESP32_YOLO/preprocessing/Hampel_LPF_Spline/test_amplitude.png'
            plt.savefig(save_path)
            print(f"  - Amplitude plot saved to: {save_path}")
            break
        
    return df

if __name__ == "__main__":
    test_file = "/Users/seolwootae/ESP32_YOLO/data/jhj/jhj_benddown_1_rx1.csv"
    if os.path.exists(test_file):
        df_processed = process_sample_file(test_file)
        print("\nSuccessfully extracted Amplitude & Phase!")
        print(f"Dataframe Rows: {len(df_processed)}")
        if len(df_processed) > 0:
            # 첫번째 유효 샘플의 진폭 길이 출력
            valid_amps = [a for a in df_processed['amplitude'] if not isinstance(a, float)]
            if valid_amps:
                print(f"Amplitude Array Length (per packet): {len(valid_amps[0])}")
    else:
        print(f"Test file not found: {test_file}")
