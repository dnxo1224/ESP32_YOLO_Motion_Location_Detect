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

def generate_800_csv_none_empty(base_dir, sample_num, subj_name, output_dir):
    """
    none_empty_X 데이터를 불러와 800 패킷 단위(청크)로 계속 쪼개며,
    각 청크를 해당 피실험자 이름(subj_name)의 'empty' 액션으로 매핑하여 저장
    예: none_empty_1 -> swt_empty_1_rx1_800.csv, swt_empty_2_rx1_800.csv ...
    """
    rx_paths = [os.path.join(base_dir, "none_empty", f"none_empty_{sample_num}_rx{i}.csv") for i in range(1, 5)]
    if not all(os.path.exists(p) for p in rx_paths):
        return False

    extracted_dfs = []
    for i, file_path in enumerate(rx_paths):
        df = pd.read_csv(file_path, header=None, encoding='utf-8', on_bad_lines='skip', engine='python')
        if df.empty: return False
        df = df.dropna(subset=[2])
        df[2] = df[2].astype(int)
        
        # CSI 진폭 추출
        amps = preprocess_single_rx(df)
        
        data_rows = []
        for seq, amp in zip(df[2], amps):
            row_dict = {'seq_id': seq}
            if amp is not None and len(amp) == NUM_SUBCARRIERS:
                for j in range(NUM_SUBCARRIERS):
                    row_dict[f'rx{i+1}_sub_{j}'] = amp[j]
            else:
                for j in range(NUM_SUBCARRIERS):
                    row_dict[f'rx{i+1}_sub_{j}'] = np.nan
            data_rows.append(row_dict)
            
        temp_df = pd.DataFrame(data_rows).drop_duplicates(subset=['seq_id']).set_index('seq_id')
        extracted_dfs.append(temp_df)

    # 4개 RX 데이터 병합
    merged_df = extracted_dfs[0].join(extracted_dfs[1:], how='outer').sort_index()
    if len(merged_df) == 0: return False

    # 최대 최소 끄트머리 측정
    start_seq = merged_df.index.min()
    end_seq = merged_df.index.max()

    chunk_idx = 1
    current_start = start_seq
    saved_chunks = 0
    
    # 800 패킷 단위로 최대 16개까지만 스라이싱 반복
    while current_start + 799 <= end_seq and chunk_idx <= 16:
        current_end = current_start + 799
        chunk_df = merged_df.reindex(range(current_start, current_end + 1))
        
        # 만약 이 구간(800개) 전체가 결측치(NaN)라면 쓸모없는 구간이므로 스킵
        if chunk_df.isna().all().all():
            current_start += 800
            continue
            
        # 단일 RX CSV 저장
        for i in range(1, 5):
            rx_cols = [c for c in chunk_df.columns if c.startswith(f'rx{i}_sub_')]
            rx_df_subset = chunk_df[rx_cols]
            
            # 피실험자 이름과 현재 청크 번호 할당
            out_name = f"{subj_name}_empty_{chunk_idx}_rx{i}_800.csv"
            out_path = os.path.join(output_dir, out_name)
            rx_df_subset.to_csv(out_path)

        chunk_idx += 1
        current_start += 800
        saved_chunks += 1

    print(f" -> Saved {saved_chunks} chunks: {subj_name}_empty_X_rxX_800.csv (Shape: 800x166)")
    return True

def process_none_empty():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'data_aligned_800')
    os.makedirs(output_dir, exist_ok=True)
    
    subj_dir = os.path.join(base_dir, "none_empty")
    if not os.path.exists(subj_dir):
        print("none_empty 데이터 폴더를 찾을 수 없습니다.")
        return
        
    # sample 번호 목록 스캔
    rx1_files = glob.glob(os.path.join(subj_dir, "*_rx1.csv"))
    sample_nums = []
    for f in rx1_files:
        parts = os.path.basename(f).split('_')
        if len(parts) >= 4:
            sample_nums.append(parts[2])  # none_empty_1_rx1.csv 에서 '1' 추출
            
    sample_nums.sort()
    total = len(sample_nums)
    
    # 데이터 매핑 (사용자 지정)
    SUBJECT_MAPPING = {
        '1': 'swt',
        '2': 'kmh',
        '3': 'kjh',
        '4': 'jhj'
    }
    
    print(f"총 {total}개의 none_empty 실험을 800-패킷 시퀀스 블럭으로 쪼개서 피실험자에게 할당합니다...\n")
    print("=" * 60)

    success_cnt = 0
    for idx, sample_num in enumerate(sample_nums):
        subj_name = SUBJECT_MAPPING.get(str(sample_num), f"unknown_{sample_num}")
        print(f"[{idx+1}/{total}] none_empty_{sample_num} 처리중... (할당 타겟: {subj_name}_empty)")
        
        if generate_800_csv_none_empty(base_dir, sample_num, subj_name, output_dir):
            success_cnt += 1
            
    print("=" * 60)
    print(f"작업 완료! (총 {success_cnt}/{total} 원본 데이터셋 기반 청크 정렬 성공)")

if __name__ == "__main__":
    process_none_empty()
