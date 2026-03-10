import numpy as np
import matplotlib.pyplot as plt
import os

def min_max_scale_3d_tensor(tensor_3d):
    """
    YOLO 입력(0~1 또는 0~255)을 위한 정규화 함수
    - 3D 텐서 전체 최소/최대값을 찾아 0~1 사이로 스케일링
    """
    # 결측치가 남아있을 혹시 모를 경우를 대비하여 nan 함수 사용
    min_val = np.nanmin(tensor_3d)
    max_val = np.nanmax(tensor_3d)
    
    if max_val == min_val:
        # 분모가 0이 되는 것을 방지
        return np.zeros_like(tensor_3d)
        
    scaled_tensor = (tensor_3d - min_val) / (max_val - min_val)
    return scaled_tensor

def standardization_3d_tensor(tensor_3d):
    """
    Z-score 정규화 (표준화)
    평균 0, 분산 1로 변환
    """
    mean_val = np.nanmean(tensor_3d)
    std_val = np.nanstd(tensor_3d)
    
    if std_val == 0:
        return np.zeros_like(tensor_3d)
        
    scaled_tensor = (tensor_3d - mean_val) / std_val
    return scaled_tensor

def test_normalization():
    # 원본(Mock) 파형: 시계열 900 프레임, 서브캐리어 100, Rx 4
    np.random.seed(42)
    time_steps, subcarriers, rx_channels = 900, 100, 4
    
    # 일부러 큰 값 (노이즈 섞인 형태)
    mock_tensor = np.random.randn(time_steps, subcarriers, rx_channels) * 50 + 100
    
    print(f"Original Tensor Shape: {mock_tensor.shape}")
    print(f"Original Range Min: {np.nanmin(mock_tensor):.2f}, Max: {np.nanmax(mock_tensor):.2f}")
    
    scaled_minmax = min_max_scale_3d_tensor(mock_tensor)
    print(f"\nMin-Max Scaled Range Min: {np.nanmin(scaled_minmax):.2f}, Max: {np.nanmax(scaled_minmax):.2f}")
    
    scaled_zscore = standardization_3d_tensor(mock_tensor)
    print(f"Z-score Scaled Range Mean: {np.nanmean(scaled_zscore):.2f}, Std: {np.nanstd(scaled_zscore):.2f}")
    
    # YOLO 이미지 형태로 저장하기 테스트 (채널 1개(Rx1) 시각화)
    plt.figure(figsize=(10, 4))
    
    # x축: 시간, y축: 서브캐리어
    plt.imshow(scaled_minmax[:, :, 0].T, aspect='auto', cmap='jet', origin='lower')
    plt.colorbar(label='Normalized Amplitude (0~1)')
    plt.title("2D Spectrogram-like Image for YOLO Input (Rx1 Channel)")
    plt.xlabel("Time Sequence (Frames)")
    plt.ylabel("Subcarriers")
    
    save_path = '/Users/seolwootae/ESP32_YOLO/preprocessing/Hampel_LPF_Spline/test_normalization_image.png'
    plt.savefig(save_path)
    print(f"\nDummy image saved to {save_path}")

if __name__ == "__main__":
    test_normalization()
