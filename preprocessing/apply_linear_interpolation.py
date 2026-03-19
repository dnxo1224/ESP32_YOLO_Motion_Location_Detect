import os
import glob
import pandas as pd
import numpy as np

def interpolate_all_files():
    input_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_aligned_872'))
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_interpolated_linear_872'))
    os.makedirs(output_dir, exist_ok=True)

    csv_files = sorted(glob.glob(os.path.join(input_dir, "*_872.csv")))
    total_files = len(csv_files)

    print(f"총 {total_files}개의 RX 데이터에 대해 [Linear] 보간을 실행합니다.\n")
    print("=" * 60)

    success_cnt = 0
    nan_file_cnt = 0

    for idx, file_path in enumerate(csv_files):
        filename = os.path.basename(file_path)
        out_path = os.path.join(output_dir, filename)

        try:
            # seq_id 를 index로 불러오기
            df = pd.read_csv(file_path, index_col='seq_id')

            nan_count = df.isna().sum().sum()

            # 결측치가 하나라도 있으면 선형 보간 수행
            if nan_count > 0:
                nan_file_cnt += 1
                # 선형 보간 수행 (앞뒤 값을 이용한 직선 보간)
                # 양 끝단 결측치는 bfill/ffill로 처리
                df = df.interpolate(method='linear').bfill().ffill()

            # 보간 완료된 데이터 저장
            df.to_csv(out_path)

            if (idx + 1) % 50 == 0 or (idx + 1) == total_files:
                print(f"[{idx+1}/{total_files}] 처리 및 저장 완료")

            success_cnt += 1

        except Exception as e:
            print(f"[{idx+1}/{total_files}] 오류 발생 ({filename}): {e}")

    print("=" * 60)
    print(f"모든 Linear 보간 작업 완료! (성공: {success_cnt}/{total_files})")
    print(f"결측치가 있었던 파일 수: {nan_file_cnt}/{total_files}")
    print(f"보간된 결과물은 다음 폴더에 저장되었습니다: \n{output_dir}")

if __name__ == "__main__":
    interpolate_all_files()
