import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# 앞에서 작성한 모델 임포트
from dual_stream_transformer import DualStreamTransformer

# ==========================================
# 1. Config & Label Definitions
# ==========================================
BATCH_SIZE = 16  # 트랜스포머는 메모리를 많이 사용하므로 배치 사이즈를 줄임
EPOCHS = 30
LEARNING_RATE = 1e-4

# Subjects
TRAIN_SUBJECTS = ['swt', 'kjh', 'kmh']
TEST_SUBJECTS = ['jhj']

# Actions (5 classes)
ACTION_MAP = {
    'benddown': 0,
    'handsup': 1,
    'stand': 2,
    'walk': 3,
    'empty': 4
}

# Zones (4 classes mapped from 16 collection locations)
ZONE_MAP = {
    1: 0, 2: 0, 5: 0, 6: 0,       # Zone 1
    3: 1, 4: 1, 7: 1, 8: 1,       # Zone 2
    9: 2, 10: 2, 13: 2, 14: 2,    # Zone 3
    11: 3, 12: 3, 15: 3, 16: 3    # Zone 4
}

# ==========================================
# 2. Dataset Definition (For [Batch, 800, 664])
# ==========================================
class CSIDualStreamDataset(Dataset):
    def __init__(self, base_dir, subjects):
        """
        Loads 4 RX files per experiment, merges them into a single (800, 664) feature matrix.
        LSTM처럼 슬라이딩 윈도우(Sliding Window)로 자르지 않고 논문대로 800개의 시간 프레임을 통째로 사용합니다.
        """
        self.samples = []       # List of (800, 664) arrays
        self.action_labels = [] # List of int (0 to 4)
        self.zone_labels = []   # List of int (0 to 3)

        if not os.path.exists(base_dir):
            print(f"Warning: Directory {base_dir} not found.")
            return

        all_rx1_files = glob.glob(os.path.join(base_dir, "*_rx1_800.csv"))
        
        valid_experiments = []
        for f in all_rx1_files:
            basename = os.path.basename(f)
            parts = basename.split('_')
            subj = parts[0]
            if subj in subjects:
                action = parts[1]
                sample_num = int(parts[2])
                valid_experiments.append((subj, action, sample_num))
        
        print(f"Loading data for subjects {subjects}...")
        
        for idx, (subj, action, sample_num) in enumerate(valid_experiments):
            rx_dfs = []
            valid_read = True
            for rx_idx in range(1, 5):
                file_path = os.path.join(base_dir, f"{subj}_{action}_{sample_num}_rx{rx_idx}_800.csv")
                if not os.path.exists(file_path):
                    valid_read = False
                    break
                df = pd.read_csv(file_path, index_col=0)
                features = df.values # (800, 166)
                if features.shape != (800, 166):
                    valid_read = False
                    break
                rx_dfs.append(features)
            
            if not valid_read:
                continue
                
            # Stack all 4 RXs horizontally -> Shape: (800, 664)
            merged_features = np.hstack(rx_dfs)
            
            a_label = ACTION_MAP.get(action, -1)
            z_label = ZONE_MAP.get(sample_num, -1)
            
            if a_label == -1 or z_label == -1:
                continue

            # NA/NaN 체크
            if np.isnan(merged_features).any():
                continue
                
            self.samples.append(merged_features)
            self.action_labels.append(a_label)
            self.zone_labels.append(z_label)
                
        self.samples = np.array(self.samples, dtype=np.float32)
        self.action_labels = np.array(self.action_labels, dtype=np.int64)
        self.zone_labels = np.array(self.zone_labels, dtype=np.int64)
        
        print(f"Loaded {len(self.samples)} full samples (800x664) for subjects {subjects}.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.samples[idx]), 
            torch.tensor(self.action_labels[idx]),
            torch.tensor(self.zone_labels[idx])
        )

# ==========================================
# 3. Training Pipeline
# ==========================================
def train_dual_stream_model():
    # 데이터 경로 설정 (현재 코드 위치 기준)
    base_dir = os.path.join(os.path.dirname(__file__), '..', 'data_interpolated_spline_800')
    
    if not os.path.exists(base_dir):
        print(f"ERROR: Cannot find data directory at {base_dir}")
        return

    # 1. Prepare Datasets and DataLoaders
    print("Initializing Training Dataset...")
    train_dataset = CSIDualStreamDataset(base_dir, subjects=TRAIN_SUBJECTS)
    
    print("\nInitializing Testing Dataset...")
    test_dataset = CSIDualStreamDataset(base_dir, subjects=TEST_SUBJECTS)
    
    if len(train_dataset) == 0 or len(test_dataset) == 0:
        print("Dataset is empty. Exiting.")
        return

    # ----- Apply StandardScaler (시간 x 공간 전체에 대한 정규화) -----
    print("Applying StandardScaler...")
    scaler = StandardScaler()
    N_tr, T, F = train_dataset.samples.shape
    train_2d = train_dataset.samples.reshape(-1, F)
    train_dataset.samples = scaler.fit_transform(train_2d).reshape(N_tr, T, F)
    
    N_te, _, _ = test_dataset.samples.shape
    test_2d = test_dataset.samples.reshape(-1, F)
    test_dataset.samples = scaler.transform(test_2d).reshape(N_te, T, F)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 2. Initialize Model, Loss, Optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nTraining on Device: {device}")
    
    # DualStreamTransformer 초기화 (클래스 5개, 구역 4개, 임베딩 차원 128)
    model = DualStreamTransformer(num_classes=5, num_zones=4, d_model=128, num_heads=4).to(device)
    
    criterion_action = nn.CrossEntropyLoss()
    criterion_zone = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 3. Training Loop
    all_train_losses = []
    all_val_losses = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        
        for batch_x, batch_y_act, batch_y_zone in train_loader:
            batch_x = batch_x.to(device)
            batch_y_act = batch_y_act.to(device)
            batch_y_zone = batch_y_zone.to(device)
            
            optimizer.zero_grad()
            
            # Forward Pass 
            # 모델은 행동 분류 예측값, 구역 분류 예측값, 어텐션 맵 반환
            preds_act, preds_zone, _ = model(batch_x)
            
            # Loss 계산
            loss_act = criterion_action(preds_act, batch_y_act)
            loss_zone = criterion_zone(preds_zone, batch_y_zone)
            loss = loss_act + loss_zone
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * batch_x.size(0)
            
        epoch_loss = running_loss / len(train_dataset)
        all_train_losses.append(epoch_loss)
        
        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        
        all_act_preds, all_act_trues = [], []
        all_zone_preds, all_zone_trues = [], []
        
        with torch.no_grad():
            for batch_x, batch_y_act, batch_y_zone in test_loader:
                batch_x = batch_x.to(device)
                batch_y_act = batch_y_act.to(device)
                batch_y_zone = batch_y_zone.to(device)
                
                preds_act, preds_zone, _ = model(batch_x)
                
                loss_act = criterion_action(preds_act, batch_y_act)
                loss_zone = criterion_zone(preds_zone, batch_y_zone)
                loss = loss_act + loss_zone
                val_loss += loss.item() * batch_x.size(0)
                
                # Accuracies
                all_act_preds.extend(torch.argmax(preds_act, dim=1).cpu().numpy())
                all_act_trues.extend(batch_y_act.cpu().numpy())
                all_zone_preds.extend(torch.argmax(preds_zone, dim=1).cpu().numpy())
                all_zone_trues.extend(batch_y_zone.cpu().numpy())
                
        val_loss = val_loss / len(test_dataset)
        all_val_losses.append(val_loss)
        
        act_acc = accuracy_score(all_act_trues, all_act_preds)
        zone_acc = accuracy_score(all_zone_trues, all_zone_preds)
        
        print(f"Epoch [{epoch}/{EPOCHS}] Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} "
              f"| Val Action Acc: {act_acc:.4f} | Val Zone Acc: {zone_acc:.4f}")

    # 4. Final Evaluation Report
    print("\n================== FINAL TEST EVALUATION ==================")
    print("\n[Action Classification Report]")
    target_names_act = ['benddown', 'handsup', 'stand', 'walk', 'empty']
    print(classification_report(all_act_trues, all_act_preds, target_names=target_names_act))
    
    print("\n[Zone Classification Report]")
    target_names_zone = ['Zone 1', 'Zone 2', 'Zone 3', 'Zone 4']
    print(classification_report(all_zone_trues, all_zone_preds, target_names=target_names_zone))
    
    # Save Model Weights
    os.makedirs('weights', exist_ok=True)
    torch.save(model.state_dict(), 'weights/dual_stream_transformer.pth')
    print("Model saved to weights/dual_stream_transformer.pth")

    # ----- Output Confusion Matrices -----
    print("\nGenerating Confusion Matrix Plot...")
    
    cm_act = confusion_matrix(all_act_trues, all_act_preds)
    cm_zone = confusion_matrix(all_zone_trues, all_zone_preds)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    sns.heatmap(cm_act, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names_act, yticklabels=target_names_act, ax=axes[0])
    axes[0].set_title(f'Dual Stream Action Classification\nAcc: {act_acc * 100:.2f}%', pad=20)
    axes[0].set_ylabel('Actual (True)')
    axes[0].set_xlabel('Predicted')
    axes[0].xaxis.tick_top()
    axes[0].xaxis.set_label_position('top')
    
    sns.heatmap(cm_zone, annot=True, fmt='d', cmap='Oranges', 
                xticklabels=target_names_zone, yticklabels=target_names_zone, ax=axes[1])
    axes[1].set_title(f'Dual Stream Zone Classification\nAcc: {zone_acc * 100:.2f}%', pad=20)
    axes[1].set_ylabel('Actual (True)')
    axes[1].set_xlabel('Predicted')
    axes[1].xaxis.tick_top()
    axes[1].xaxis.set_label_position('top')
    
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/dual_stream_confusion_matrix.png')
    print("Confusion matrix plots saved to 'results/dual_stream_confusion_matrix.png'.")
    plt.show()

if __name__ == "__main__":
    train_dual_stream_model()
