import os
import glob
import pandas as pd
import numpy as np

def interpolate_all_files():
    input_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_aligned_800'))
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_interpolated_spline_800'))
    os.makedirs(output_dir, exist_ok=True)
    
    csv_files = glob.glob(os.path.join(input_dir, "*_800.csv"))
    total_files = len(csv_files)
    
    print(f"총 {total_files}개의 RX 데이터에 대해 [Spline] 보간을 실행합니다.\n")
    print("=" * 60)
    
    success_cnt = 0
    for idx, file_path in enumerate(csv_files):
        filename = os.path.basename(file_path)
        out_path = os.path.join(output_dir, filename)
        
        try:
            # seq_id 를 index로 불러오기
            df = pd.read_csv(file_path, index_col='seq_id')
            
            # 결측치가 하나라도 있으면 보간 수행
            if df.isna().sum().sum() > 0:
                # Spline (order=3) 사용 시 값이 극단적으로 뭉치거나 간격이 불규칙하면
                # 행렬 연산 수렴 실패 (maxit warning)가 발생할 수 있습니다.
                # 이를 방지하기 위해 단조성(Monotonicity)을 보장하여 오버슈팅이 없는 pchip 보간을 사용합니다.
                df = df.interpolate(method='pchip').bfill().ffill()
                
            # 보간 완료된 데이터 저장
            df.to_csv(out_path)
            
            if (idx + 1) % 50 == 0 or (idx + 1) == total_files:
                print(f"[{idx+1}/{total_files}] 처리 및 저장 완료: {output_dir}")
                
            success_cnt += 1
            
        except Exception as e:
            print(f"[{idx+1}/{total_files}] 오류 발생 ({filename}): {e}")
            
    print("=" * 60)
    print(f"모든 Spline 보간 작업 완료! (성공: {success_cnt}/{total_files})")
    print(f"보간된 결과물은 다음 폴더에 저장되었습니다: \n{output_dir}")

if __name__ == "__main__":
    interpolate_all_files()
