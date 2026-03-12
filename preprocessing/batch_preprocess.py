import os
import glob
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# --- 파이프라인 상수 ---
NULL_SUBCARRIERS_192 = list(range(0, 6)) + [32] + list(range(59, 66)) + list(range(123, 134)) + [191]
VALID_IDX_192 = [i for i in range(192) if i not in NULL_SUBCARRIERS_192]
NUM_SUBCARRIERS = 166  # 192 - 26(Null)

# --- 파이프라인 함수들 ---
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

def hampel_filter_2d(data_2d, window_size=5, n_sigmas=3):
    df = pd.DataFrame(data_2d)
    rolling_median = df.rolling(window=2*window_size+1, center=True, min_periods=1).median()
    k = 1.4826
    mad = k * df.rolling(window=2*window_size+1, center=True, min_periods=1).apply(
        lambda x: np.median(np.abs(x - np.median(x))), raw=True)
    diff = np.abs(df - rolling_median)
    outlier_mask = diff > (n_sigmas * mad)
    df[outlier_mask] = rolling_median[outlier_mask]
    return df.values

def butter_lowpass_filter(data, cutoff=3.0, fs=30.0, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    if data.shape[0] <= max(len(a), len(b)) * 3:
        return data
    return filtfilt(b, a, data, axis=0)

def min_max_scale(tensor_3d):
    min_val = np.nanmin(tensor_3d)
    max_val = np.nanmax(tensor_3d)
    if max_val == min_val: return np.zeros_like(tensor_3d)
    return (tensor_3d - min_val) / (max_val - min_val)

def process_single_experiment(base_dir, subj, action, sample_num):
    rx_paths = [os.path.join(base_dir, subj, f"{subj}_{action}_{sample_num}_rx{i}.csv") for i in range(1, 5)]
    if not all(os.path.exists(p) for p in rx_paths):
        return None

    extracted_dfs = []
    for i, file_path in enumerate(rx_paths):
        df = pd.read_csv(file_path, header=None, encoding='utf-8', on_bad_lines='skip', engine='python')
        if df.empty: return None
        df = df.dropna(subset=[2])
        df[2] = df[2].astype(int)
        amps = preprocess_single_rx(df)
        temp_df = pd.DataFrame({'seq_id': df[2], f'rx{i+1}_amps': amps}).drop_duplicates(subset=['seq_id']).set_index('seq_id')
        extracted_dfs.append(temp_df)

    merged_df = extracted_dfs[0].join(extracted_dfs[1:], how='outer').sort_index()
    if len(merged_df) == 0: return None

    # ==========================================
    # [정렬 및 결측 판단] 시퀀스 순서대로 4개의 RX 데이터를 정렬 및 패킷 유실 알림
    # ==========================================
    start_seq = merged_df.index.min()
    end_seq = start_seq + 899  # 900 패킷 (30초)

    print(f"\n  [INFO] 정렬 및 결측 분석 ({subj} - {action} - {sample_num})")
    print(f"  -> 패킷 시퀀스 범위 (Raw): {merged_df.index.min()} ~ {merged_df.index.max()} (고유 seq: {len(merged_df)}개)")

    # 시퀀스 순서대로 900개의 시퀀스를 보장하여 재정렬. 이 과정에서 누락된 seq_id는 전체 NaN 행으로 추가되어 정렬됨.
    merged_df = merged_df.reindex(range(start_seq, end_seq + 1))

    # 보간 이전에 각 RX별 결측 패킷(Missing packets) 개수 판단 및 출력
    for i in range(1, 5):
        col_name = f'rx{i}_amps'
        received_count = merged_df[col_name].notna().sum()
        missing_count = 900 - received_count
        print(f"  -> RX{i} 보간 전 누락 패킷: {missing_count}개 (수신: {received_count}/900)")

    rx_tensors = []
    for i in range(1, 5):
        col_name = f'rx{i}_amps'
        rx_matrix = []
        for val in merged_df[col_name]:
            if isinstance(val, np.ndarray) and len(val) == NUM_SUBCARRIERS:
                rx_matrix.append(val)
            else:
                rx_matrix.append(np.full(NUM_SUBCARRIERS, np.nan))

        rx_df = pd.DataFrame(rx_matrix)

        # ==========================================
        # [Step 1] 시간에 따른 패킷 드롭(전체 행 결측) 보간
        # ==========================================
        # 전체가 NaN인 행(Packet Drop)에 대해서만 위아래 시간 흐름에 따라 선형/Spline 보간을 진행합니다.
        # 부분적으로 NaN이 있는 경우는 MICE가 해결할 수 있도록 내버려둡니다.
        full_nan_mask = rx_df.isna().all(axis=1)
        if full_nan_mask.any():
            # 임시로 전체 NaN인 곳만 보간하기 편하게 제한적 처리를 구사할 수도 있지만, 
            # pandas 특성상 전체에 적용 후, 부분 결측치는 다시 NaN으로 돌려놓는 트릭을 사용합니다.
            temp_df = rx_df.copy()
            temp_df = temp_df.interpolate(method='spline', order=3).bfill().ffill()
            rx_df[full_nan_mask] = temp_df[full_nan_mask]

        # ==========================================
        # [Step 2] 서브캐리어 간 연관성을 이용한 MICE 예측 (부분 결측)
        # ==========================================
        # 만약 일부 서브캐리어 데이터만 유실된 '부분 결측치'가 존재한다면,
        # 머신러닝(MICE)을 활용하여 같은 시간대의 다른 서브캐리어 정보를 보고 예측해 채워넣습니다.
        if rx_df.isna().sum().sum() > 0:
            imputer = IterativeImputer(max_iter=10, random_state=42)
            rx_imputed = imputer.fit_transform(rx_df.values)
            rx_df = pd.DataFrame(rx_imputed)

        # 방어 코드 (여전히 남은 결측치가 있다면 0으로 초기화)
        rx_df = rx_df.fillna(0)

        # ==========================================
        # [Step 3] 이상치 및 노이즈 제거 (Hampel & Low-Pass)
        # ==========================================
        rx_vals = hampel_filter_2d(rx_df.values, window_size=5, n_sigmas=3)
        rx_vals = butter_lowpass_filter(rx_vals, cutoff=3.0, fs=30.0)
        rx_tensors.append(rx_vals)

    tensor_3d = np.stack(rx_tensors, axis=-1)
    tensor_3d = min_max_scale(tensor_3d)
    return tensor_3d

def process_all_data():
    # 현재 스크립트 위치 기준으로 상위 폴더의 data 디렉토리 참조
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    output_dir = os.path.join(os.path.dirname(__file__), 'processed_tensors')
    os.makedirs(output_dir, exist_ok=True)

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

    total = len(all_experiments)
    print(f"Total experiments found: {total}")

    success = 0
    for idx, (subj, action, sample_num) in enumerate(all_experiments):
        out_name = f"{subj}_{action}_{sample_num}.npy"
        out_path = os.path.join(output_dir, out_name)
        if os.path.exists(out_path):
            print(f"[{idx+1}/{total}] {out_name} already exists, skipping.")
            success += 1
            continue

        print(f"[{idx+1}/{total}] Processing {subj}_{action}_{sample_num} ...", end="", flush=True)
        try:
            tensor_3d = process_single_experiment(base_dir, subj, action, sample_num)
            if tensor_3d is not None:
                np.save(out_path, tensor_3d)
                print(f" OK. Shape: {tensor_3d.shape}")
                success += 1
            else:
                print(" Skipped (missing files).")
        except Exception as e:
            print(f" Error: {e}")

    print(f"\nDone! {success}/{total} experiments saved to {output_dir}")

if __name__ == "__main__":
    process_all_data()
