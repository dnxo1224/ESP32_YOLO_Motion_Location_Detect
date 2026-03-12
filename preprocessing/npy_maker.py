import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm

def convert_csv_to_npy():
    # 경로 설정
    input_dir = "../data_interpolated_spline_800"
    output_dir = "./processed_tensors_800_664"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 파일 탐색 조건 수정 (*_rx1_800.csv)
    rx1_files = glob.glob(os.path.join(input_dir, "*_rx1_800.csv"))
    print(f"총 {len(rx1_files)} 개의 실험 데이터를 병합 및 변환합니다.")
    
    success_count = 0
    
    for rx1_path in tqdm(rx1_files):
        basename = os.path.basename(rx1_path)
        
        # 2. prefix 추출 수정 (예: jhj_benddown_1_rx1_800.csv -> jhj_benddown_1)
        prefix = basename.replace('_rx1_800.csv', '')
        
        rx_data_list = []
        is_valid = True
        
        for i in range(1, 5):
            # 3. Rx2, Rx3, Rx4 파일명 재조립 수정
            rx_path = os.path.join(input_dir, f"{prefix}_rx{i}_800.csv")
            
            if not os.path.exists(rx_path):
                print(f"\n[경고] 파일 누락: {rx_path}")
                is_valid = False
                break
                
            # CSV 로드
            df = pd.read_csv(rx_path)
            
            # 뒤에서부터 166개의 컬럼(서브캐리어 데이터)만 확실하게 추출
            if df.shape[1] >= 166:
                data = df.iloc[:, -166:].values
            else:
                print(f"\n[경고] 데이터 형태 이상 ({df.shape}): {rx_path}")
                is_valid = False
                break
            
            # 프레임 수가 800개가 맞는지 확인 (부족하면 패딩, 남으면 자르기)
            if data.shape[0] > 800:
                data = data[:800, :]
            elif data.shape[0] < 800:
                padding = np.zeros((800 - data.shape[0], 166))
                data = np.vstack([data, padding])
                
            rx_data_list.append(data)
            
        if is_valid:
            # 4개의 배열 (800, 166) 을 가로축(axis=1)으로 병합 -> (800, 664)
            merged_tensor = np.concatenate(rx_data_list, axis=1)
            
            # .npy 포맷으로 저장
            save_path = os.path.join(output_dir, f"{prefix}.npy")
            np.save(save_path, merged_tensor)
            success_count += 1
            
    print(f"\n✅ 변환 완료! 총 {success_count} 개의 파일이 '{output_dir}' 에 저장되었습니다.")

if __name__ == "__main__":
    convert_csv_to_npy()