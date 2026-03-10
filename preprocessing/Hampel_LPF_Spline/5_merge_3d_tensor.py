import pandas as pd
import numpy as np
import os

# (생략: 이전 1, 2, 3, 4단계 함수들은 동일하게 재사용 가정, 여기서는 병합 형태만 테스트)
def merge_to_3d_tensor(rx_data_dict):
    """
    rx_data_dict: {'rx1': np.array(Time, Subcarriers), 'rx2': np.array(...), ...}
    반환: np.array(Time, Subcarriers, 4)의 3D 텐서
    """
    # 딕셔너리에 들어온 배열들을 채널 축(axis=2)을 기준으로 적층(Stack)
    rx_keys = sorted(rx_data_dict.keys()) # rx1, rx2, rx3, rx4 순서 보장
    stacked = np.stack([rx_data_dict[k] for k in rx_keys], axis=-1)
    return stacked

def test_3d_merge():
    # 가상의 보간/필터링이 끝난 데이터 생성
    # Time = 900 프레임, Subcarriers = 154개, Rx = 4개
    time_steps = 900
    subcarriers = 154
    
    mock_rx_data = {
        'rx1': np.random.rand(time_steps, subcarriers),
        'rx2': np.random.rand(time_steps, subcarriers),
        'rx3': np.random.rand(time_steps, subcarriers),
        'rx4': np.random.rand(time_steps, subcarriers)
    }
    
    # 병합
    tensor_3d = merge_to_3d_tensor(mock_rx_data)
    
    print("--- 3D Tensor Merge Test ---")
    print(f"Input arrays shape: {mock_rx_data['rx1'].shape}")
    print(f"Final 3D Tensor shape: {tensor_3d.shape}  -> (Time, Subcarriers, Rx_Channels)")
    
if __name__ == "__main__":
    test_3d_merge()
