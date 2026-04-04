import os
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# --- 설정 및 하이퍼파라미터 ---
WINDOW_SIZE = 90
STRIDE = 10
# 데이터셋 규격: 800 (Packet) x 166 (Channel) x 4 (Subcarrier/Dimension)
PACKET_LEN = 800
CHANNEL_DIM = 166
SUB_DIM = 4
INPUT_DIM = CHANNEL_DIM * SUB_DIM  # 664

# 데이터 경로 설정 (LightGBM/processed 폴더 사용)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), 'LightGBM', 'processed')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_and_preprocess_data(data_dir, test_subject='kjh'):
    """LightGBM/processed 폴더의 모든 npz 파일에서 데이터를 로드하고 Train/Test로 분리"""
    print(f"Loading data from {data_dir}...")
    
    npz_files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {data_dir}")
    
    x_train_list, y_train_list = [], []
    x_test_list, y_test_list = [], []
    
    for fname in tqdm(npz_files, desc="Reading NPZ files"):
        file_path = os.path.join(data_dir, fname)
        data = np.load(file_path, allow_pickle=True)
        
        # grids shape: (4, 800, 166) -> (Time, Channel, RX) 형태로 변경 필요
        # (4, 800, 166) -> (800, 166, 4)
        x = data['grids'].transpose(1, 2, 0)
        y = int(data['action'])
        subject = str(data['subject'])
        
        if subject == test_subject:
            x_test_list.append(x)
            y_test_list.append(y)
        else:
            x_train_list.append(x)
            y_train_list.append(y)
            
    x_train_raw = np.array(x_train_list)
    y_train_raw = np.array(y_train_list)
    x_test_raw = np.array(x_test_list)
    y_test_raw = np.array(y_test_list)
    
    print(f"Loaded: Train Samples={len(x_train_raw)}, Test Samples={len(x_test_raw)}")
    return x_train_raw, y_train_raw, x_test_raw, y_test_raw

def create_sliding_windows(x_data, y_data):
    """3차원 시계열 데이터에서 슬라이딩 윈도우 추출 및 Flatten"""
    window_features = []
    window_labels = []
    sample_ids = []
    
    for i in range(len(x_data)):
        sample = x_data[i] # (800, 166, 4)
        label = y_data[i]
        
        start = 0
        while start + WINDOW_SIZE <= PACKET_LEN:
            window = sample[start:start + WINDOW_SIZE] # (90, 166, 4)
            # SVM 입력을 위해 1차원으로 Flatten (90 * 166 * 4 = 59,760 차원)
            # 주의: 데이터가 많을 경우 메모리 부족이 발생할 수 있음
            window_features.append(window.flatten())
            window_labels.append(label)
            sample_ids.append(i) # Voting을 위한 원본 샘플 ID
            start += STRIDE
            
    return np.array(window_features), np.array(window_labels), sample_ids

def run_svm_analysis():
    # 1. 데이터 로드 및 분리
    x_train_raw, y_train_raw, x_test_raw, y_test_raw = load_and_preprocess_data(DATA_DIR)

    # 2. 슬라이딩 윈도우 생성
    print("Creating sliding windows...")
    x_train, y_train, _ = create_sliding_windows(x_train_raw, y_train_raw)
    x_test, y_test, test_sample_ids = create_sliding_windows(x_test_raw, y_test_raw)
    
    print(f"Windowed Train: {x_train.shape}, Windowed Test: {x_test.shape}")

    # 3. 데이터 정규화 (SVM에서 매우 중요)
    print("Scaling features...")
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # 4. SVM 모델 설정 및 학습
    print(f"Training SVM with C=100, gamma=1e-4...")
    # 메모리 및 속도 문제 방지를 위해 max_iter 제한을 고려할 수 있음
    svm = SVC(
        kernel='rbf',
        C=100,
        gamma=1e-4,
        class_weight='balanced',
        probability=True
    )
    svm.fit(x_train, y_train)

    # 5. 추론 및 샘플 단위 Majority Voting
    print("Evaluating with Majority Voting...")
    win_preds = svm.predict(x_test)
    
    # 샘플별 예측값 모으기
    sample_preds = {}
    for i, pred in enumerate(win_preds):
        sid = test_sample_ids[i]
        if sid not in sample_preds:
            sample_preds[sid] = []
        sample_preds[sid].append(pred)
    
    # Voting 결과 도출
    final_preds = []
    for sid in sorted(sample_preds.keys()):
        voted = Counter(sample_preds[sid]).most_common(1)[0][0]
        final_preds.append(voted)
    
    # 6. 결과 출력
    acc = np.mean(final_preds == y_test_raw)
    print(f"\n✨ Final Test Accuracy (Voted): {acc:.4f}")
    print("\n[Classification Report]")
    print(classification_report(y_test_raw, final_preds))

    # Confusion Matrix 시각화
    cm = confusion_matrix(y_test_raw, final_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='PuBu')
    plt.title(f'SVM Action Classification (Voted)\nAcc: {acc:.4f}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(RESULTS_DIR, 'action_svm_voted_cm.png'))
    print(f"Confusion Matrix saved to {RESULTS_DIR}")

if __name__ == "__main__":
    run_svm_analysis()