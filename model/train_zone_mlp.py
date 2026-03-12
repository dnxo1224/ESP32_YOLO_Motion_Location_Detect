import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

# --- 1. Zone 라벨 변환 함수 ---
def get_zone_label(position_idx):
    """ 위치 번호(1~16)를 Zone(0~3)으로 매핑 """
    if position_idx in [1, 2, 5, 6]: return 0      # Zone 1
    elif position_idx in [3, 4, 7, 8]: return 1    # Zone 2
    elif position_idx in [9, 10, 13, 14]: return 2 # Zone 3
    elif position_idx in [11, 12, 15, 16]: return 3 # Zone 4
    else: raise ValueError(f"Invalid position: {position_idx}")

# --- 2. 커스텀 Dataset 클래스 (CSV에서 직접 로드) ---
class CSIDataset(Dataset):
    def __init__(self, csv_dir):
        """
        data_interpolated_spline_800 폴더에서 직접 CSV를 읽어
        rx1~rx4를 병합하여 (800, 664) 텐서를 생성
        """
        # rx1 파일 목록으로 실험 단위 파악
        self.csv_dir = csv_dir
        rx1_files = glob.glob(os.path.join(csv_dir, "*_rx1_800.csv"))
        
        # prefix 목록 생성 (예: jhj_benddown_1)
        self.prefixes = []
        for f in rx1_files:
            basename = os.path.basename(f)
            prefix = basename.replace('_rx1_800.csv', '')
            self.prefixes.append(prefix)
        
        print(f"총 {len(self.prefixes)}개의 실험 데이터를 발견했습니다.")
        
    def __len__(self):
        return len(self.prefixes)
    
    def __getitem__(self, idx):
        prefix = self.prefixes[idx]
        
        # 파일명 분석: 예) jhj_walk_1 -> position=1
        parts = prefix.split('_')
        position_idx = int(parts[2])
        label = get_zone_label(position_idx)
        
        # rx1 ~ rx4 CSV 로드 후 병합 -> (800, 664)
        rx_data_list = []
        for i in range(1, 5):
            rx_path = os.path.join(self.csv_dir, f"{prefix}_rx{i}_800.csv")
            df = pd.read_csv(rx_path)
            # 뒤에서 166개 컬럼(서브캐리어 데이터)만 추출
            data = df.iloc[:, -166:].values
            
            # 프레임 수 800으로 맞추기
            if data.shape[0] > 800:
                data = data[:800, :]
            elif data.shape[0] < 800:
                padding = np.zeros((800 - data.shape[0], 166))
                data = np.vstack([data, padding])
            
            rx_data_list.append(data)
        
        # 4개 rx 병합: (800, 166) x 4 -> (800, 664)
        merged = np.concatenate(rx_data_list, axis=1)
        
        x_tensor = torch.tensor(merged, dtype=torch.float32)
        y_tensor = torch.tensor(label, dtype=torch.long)
        
        return x_tensor, y_tensor

# --- 3. MLP 모델 정의 ---
class ZoneMLP(nn.Module):
    def __init__(self, num_classes=4):
        super(ZoneMLP, self).__init__()
        self.fc1 = nn.Linear(664, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # x 형태: (Batch, 800, 664)
        x = x.mean(dim=1)  # 시간축 평균 -> (Batch, 664)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)  # (Batch, 4)
        return x

# --- 4. 메인 학습 루프 ---
def train_model():
    # 설정값
    DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data_interpolated_spline_800")
    BATCH_SIZE = 16
    EPOCHS = 30
    LEARNING_RATE = 0.001
    
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 데이터셋 로드 및 Train/Val 분할 (8:2)
    dataset = CSIDataset(DATA_DIR)
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Total data: {total_size} | Train: {train_size} | Val: {val_size}")
    
    # 모델, 손실 함수, 옵티마이저 초기화
    model = ZoneMLP(num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 학습 시작
    for epoch in range(EPOCHS):
        model.train()
        train_loss, train_correct = 0, 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += torch.sum(preds == labels.data)
            
        epoch_train_loss = train_loss / train_size
        epoch_train_acc = train_correct.double() / train_size
        
        # 검증 루프
        model.eval()
        val_loss, val_correct = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels.data)
                
        epoch_val_loss = val_loss / val_size
        epoch_val_acc = val_correct.double() / val_size
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] "
              f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}")

    print("✅ MLP Zone Classification Training Complete!")

if __name__ == "__main__":
    train_model()
