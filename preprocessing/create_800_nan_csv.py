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

def generate_from_paths(rx_paths, subj, action, sample_num, output_dir):
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
        
        # 각 패킷(166개 서브캐리어)을 개별 행열로 펼쳐 저장 (다루기 쉬운 구조로 변경)
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

    # 4개 RX 데이터를 seq_id 기준으로 Outer Join (동기화)
    merged_df = extracted_dfs[0].join(extracted_dfs[1:], how='outer').sort_index()
    if len(merged_df) == 0: return False

    # 800 패킷 강제 추출 및 정렬 (비어있는 시퀀스는 모든 RX가 NaN으로 들어감)
    start_seq = merged_df.index.min()
    end_seq = start_seq + 799  # 딱 800개 (0~799)
    merged_df = merged_df.reindex(range(start_seq, end_seq + 1))

    # 각 RX 별로 컬럼을 쪼개서 개별 CSV 파일로 저장
    for i in range(1, 5):
        # rx{i}_sub_0 ~ rx{i}_sub_165 형태의 컬럼명만 필터링
        rx_cols = [c for c in merged_df.columns if c.startswith(f'rx{i}_sub_')]
        rx_df_subset = merged_df[rx_cols]
        
        # 저장 형태: jhj_walk_1_rx1_800.csv 또는 none_empty_1_rx1_800.csv
        out_name = f"{subj}_{action}_{sample_num}_rx{i}_800.csv" if subj != "none_empty" else f"none_empty_{sample_num}_rx{i}_800.csv"
        out_path = os.path.join(output_dir, out_name)
        rx_df_subset.to_csv(out_path)

    print(f" -> Saved 4 RX files: {out_name[:-10]}X_800.csv (Shape: 800x166)")
    
    return True

def process_all_data():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'data_aligned_800')
    os.makedirs(output_dir, exist_ok=True)
    
    # none_empty 추가
    subjects = ["jhj", "kjh", "kmh", "swt", "none"]
    all_experiments = []
    
    for subj in subjects:
        if subj == "none":
            # none_empty 폴더 내부 파일 파싱
            subj_dir = os.path.join(base_dir, "none_empty")
            if not os.path.exists(subj_dir): continue
            rx1_files = glob.glob(os.path.join(subj_dir, "*_rx1.csv"))
            for f in rx1_files:
                parts = os.path.basename(f).split('_')
                # none_empty_1_rx1.csv -> parts: ['none', 'empty', '1', 'rx1.csv']
                if len(parts) >= 4:
                    # subj: "none_empty", action: parts[1] ('empty'), sample_num: parts[2] ('1')
                    all_experiments.append(("none_empty", parts[1], parts[2]))
        else:
            subj_dir = os.path.join(base_dir, subj)
            if not os.path.exists(subj_dir): continue
            rx1_files = glob.glob(os.path.join(subj_dir, f"{subj}_*_rx1.csv"))
            for f in rx1_files:
                parts = os.path.basename(f).split('_')
                if len(parts) >= 4:
                    all_experiments.append((subj, parts[1], parts[2]))

    all_experiments.sort()
    total = len(all_experiments)
    
    print(f"총 {total}개의 실험 데이터를 800-패킷 시퀀스로 정렬하여 {output_dir} 에 저장합니다...\n")
    print("=" * 60)

    success_cnt = 0
    for idx, (subj, action, sample_num) in enumerate(all_experiments):
        print(f"[{idx+1}/{total}] {subj}_{action}_{sample_num} 처리중...")
        if subj == "none_empty":
            # none_empty의 경우 base_dir/none_empty 폴더 경로에 접근하게 해야 하므로,
            # generate_800_csv_single_experiment 내에서 path 구조 분기를 위해 base_dir 활용
            
            # 여기서 편의상 base_dir 밑에 바로 none_empty 폴더가 있고 
            # 파일이 none_empty_1_rx{i}.csv 임을 고려하여 rx_paths를 강제로 생성해 넘겨주는 방식보다는
            # 함수를 약간 리팩토링합니다.
            rx_paths = [os.path.join(base_dir, "none_empty", f"none_empty_{sample_num}_rx{i}.csv") for i in range(1, 5)]
        else:
            rx_paths = [os.path.join(base_dir, subj, f"{subj}_{action}_{sample_num}_rx{i}.csv") for i in range(1, 5)]
            
        if generate_from_paths(rx_paths, subj, action, sample_num, output_dir):
            success_cnt += 1
        
    print("=" * 60)
    print(f"작업 완료! (총 {success_cnt}/{total} 실험. 산출물 {success_cnt*4} 파일 저장 성공)")

if __name__ == "__main__":
    process_all_data()
