import os
import glob
import numpy as np
import pandas as pd

# --- 파이프라인 상수 ---
NULL_SUBCARRIERS_192 = list(range(0, 6)) + [32] + list(range(59, 66)) + list(range(123, 134)) + [191]
VALID_IDX_192 = [i for i in range(192) if i not in NULL_SUBCARRIERS_192]
NUM_SUBCARRIERS = 166

# --- 유틸리티 함수 ---
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
        return np.sqrt(real**2 + imag**2)
    except:
        return None

def preprocess_single_rx(df_rx):
    start_col = 25
    for col in df_rx.columns[:50]:
        if df_rx[col].dtype == object and isinstance(df_rx[col].iloc[0], str) and '[' in df_rx[col].iloc[0]:
            start_col = col; break
    csi_df = df_rx.loc[:, start_col:df_rx.columns[-2]]
    amps = csi_df.apply(extract_csi_amplitude, axis=1).tolist()
    cleaned = []
    for a in amps:
        if a is None: cleaned.append(None)
        elif len(a) == 192: cleaned.append(a[VALID_IDX_192])
        elif len(a) == 64: cleaned.append(a[list(range(6, 32)) + list(range(33, 59))])
        else: cleaned.append(a)
    return cleaned

def check_missing_single_experiment(base_dir, subj, action, sample_num):
    rx_paths = [os.path.join(base_dir, subj, f"{subj}_{action}_{sample_num}_rx{i}.csv") for i in range(1, 5)]
    if not all(os.path.exists(p) for p in rx_paths):
        return False

    extracted_dfs = []
    for i, file_path in enumerate(rx_paths):
        df = pd.read_csv(file_path, header=None, encoding='utf-8', on_bad_lines='skip', engine='python')
        if df.empty: return False
        df = df.dropna(subset=[2])
        df[2] = df[2].astype(int)
        amps = preprocess_single_rx(df)
        temp_df = pd.DataFrame({'seq_id': df[2], f'rx{i+1}_amps': amps}).drop_duplicates(subset=['seq_id']).set_index('seq_id')
        extracted_dfs.append(temp_df)

    merged_df = extracted_dfs[0].join(extracted_dfs[1:], how='outer').sort_index()
    if len(merged_df) == 0: return False

    start_seq = merged_df.index.min()
    end_seq = start_seq + 899  # 900 패킷 (30초)

    print(f"\n[INFO] 데이터 분석: {subj} - {action} - {sample_num}")
    print(f" -> 패킷 시퀀스 범위 (Raw): {merged_df.index.min()} ~ {merged_df.index.max()} (수신된 고유 시퀀스 개수: {len(merged_df)}개)")

    # 시퀀스 정렬 및 누락 확인
    merged_df = merged_df.reindex(range(start_seq, end_seq + 1))

    for i in range(1, 5):
        col_name = f'rx{i}_amps'
        received_count = merged_df[col_name].notna().sum()
        missing_count = 900 - received_count
        print(f" -> RX{i} 누락 패킷: {missing_count}개 (정상 수신: {received_count}/900)")
        
    return True

def analyze_all_missing_packets():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    
    subjects = ["jhj", "kjh", "kmh", "swt"]
    all_experiments = []
    for subj in subjects:
        subj_dir = os.path.join(base_dir, subj)
        if not os.path.exists(subj_dir): continue
        rx1_files = glob.glob(os.path.join(subj_dir, f"{subj}_*_rx1.csv"))
        for f in rx1_files:
            parts = os.path.basename(f).split('_')
            if len(parts) >= 4:
                all_experiments.append((subj, parts[1], parts[2]))

    # 정렬하여 출력 순서 보기 좋게
    all_experiments.sort()
    
    total = len(all_experiments)
    print(f"총 {total}개의 실험 데이터셋을 발견했습니다. 결측 패킷 분석을 시작합니다...\n")
    print("=" * 60)

    for idx, (subj, action, sample_num) in enumerate(all_experiments):
        check_missing_single_experiment(base_dir, subj, action, sample_num)
        print("-" * 60)

if __name__ == "__main__":
    analyze_all_missing_packets()
