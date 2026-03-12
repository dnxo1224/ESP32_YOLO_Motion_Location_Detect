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
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. Config & Label Definitions
# ==========================================
WINDOW_SIZE = 40
STRIDE = 3
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 0.001

# Subjects
ALL_SUBJECTS = ['jhj', 'kjh', 'kmh', 'swt']
# User requested 3:1 ratio (Train 3 subjects, Test 1 subject)
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
# 2. Dataset Definition
# ==========================================
class CSIMultiTaskDataset(Dataset):
    def __init__(self, base_dir, subjects, window_size=40, stride=4):
        """
        Loads 4 RX files per experiment, merges them into (800, 664) feature matrix,
        and slices them into (40, 664) windows with the specified stride.
        """
        self.windows = []       # List of (40, 664) arrays
        self.action_labels = [] # List of int (0 to 4)
        self.zone_labels = []   # List of int (0 to 3)

        if not os.path.exists(base_dir):
            print(f"Warning: Directory {base_dir} not found. Please run preprocessing first.")
            return

        # Find all rx1 files to identify unique experiments
        # Filename expected format: {subj}_{action}_{sample_num}_rx1_800.csv
        all_rx1_files = glob.glob(os.path.join(base_dir, "*_rx1_800.csv"))
        
        valid_experiments = []
        for f in all_rx1_files:
            basename = os.path.basename(f)
            parts = basename.split('_')
            # Example: 'swt_empty_1_rx1_800.csv' or 'jhj_walk_12_rx1_800.csv'
            subj = parts[0]
            if subj in subjects:
                action = parts[1]
                sample_num = int(parts[2])
                valid_experiments.append((subj, action, sample_num))
        
        print(f"Loading data for subjects {subjects}...")
        
        for idx, (subj, action, sample_num) in enumerate(valid_experiments):
            # Load RX1 to RX4
            rx_dfs = []
            valid_read = True
            for rx_idx in range(1, 5):
                file_path = os.path.join(base_dir, f"{subj}_{action}_{sample_num}_rx{rx_idx}_800.csv")
                if not os.path.exists(file_path):
                    valid_read = False
                    break
                df = pd.read_csv(file_path, index_col=0)
                # Ensure 166 columns (rx{idx}_sub_0 ... rx{idx}_sub_165)
                # Drop seq_id if it came as a column, otherwise it's index
                features = df.values # Shape should be (800, 166)
                if features.shape != (800, 166):
                    valid_read = False
                    break
                rx_dfs.append(features)
            
            if not valid_read:
                continue
                
            # Stack all 4 RXs horizontally -> Shape: (800, 664)
            merged_features = np.hstack(rx_dfs)
            
            # Determine labels
            a_label = ACTION_MAP.get(action, -1)
            # Both regular actions and empty actions have a sample_num from 1 to 16
            z_label = ZONE_MAP.get(sample_num, -1)
            
            if a_label == -1 or z_label == -1:
                continue

            # Sliding window slicing
            num_windows = (merged_features.shape[0] - window_size) // stride + 1
            for w in range(num_windows):
                start_idx = w * stride
                end_idx = start_idx + window_size
                window_data = merged_features[start_idx:end_idx, :]
                
                # Check for strictly invalid numeric data
                if np.isnan(window_data).any():
                    continue
                    
                self.windows.append(window_data)
                self.action_labels.append(a_label)
                self.zone_labels.append(z_label)
                
        self.windows = np.array(self.windows, dtype=np.float32)
        self.action_labels = np.array(self.action_labels, dtype=np.int64)
        self.zone_labels = np.array(self.zone_labels, dtype=np.int64)
        
        print(f"Loaded {len(self.windows)} windows for subjects {subjects}.")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.windows[idx]), 
            torch.tensor(self.action_labels[idx]), 
            torch.tensor(self.zone_labels[idx])
        )


# ==========================================
# 3. Model Definition (Multi-Task LSTM)
# ==========================================
class CSIMultiTaskLSTM(nn.Module):
    def __init__(self, input_dim=664, hidden_dim=128, num_layers=2, num_actions=5, num_zones=4):
        super(CSIMultiTaskLSTM, self).__init__()
        
        # Shared LSTM feature extractor
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3
        )
        
        # Dual Classification Heads
        self.fc_action = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_actions)
        )
        
        self.fc_zone = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_zones)
        )

    def forward(self, x):
        # x shape: (Batch, Seq_Length, Input_Dim) -> (B, 40, 664)
        lstm_out, (hn, cn) = self.lstm(x)
        
        # Take the hidden state of the final layer from the last time step
        # hn shape: (num_layers, batch, hidden_size)
        last_hidden = hn[-1, :, :] # Shape: (B, 128)
        
        # Pass features to both specialized heads
        out_action = self.fc_action(last_hidden)
        out_zone = self.fc_zone(last_hidden)
        
        return out_action, out_zone


# ==========================================
# 4. Training Pipeline
# ==========================================
def train_multi_task_model():
    base_dir = os.path.join(os.path.dirname(__file__), '..', 'data_interpolated_spline_800')
    
    # Check if dataset exists before starting
    if not os.path.exists(base_dir):
        print("ERROR: Run preprocessing scripts to generate 'data_interpolated_spline_800' first!")
        return

    # 1. Prepare Datasets and DataLoaders
    print("Initializing Training Dataset...")
    train_dataset = CSIMultiTaskDataset(base_dir, subjects=TRAIN_SUBJECTS, window_size=WINDOW_SIZE, stride=STRIDE)
    
    print("\nInitializing Testing Dataset...")
    test_dataset = CSIMultiTaskDataset(base_dir, subjects=TEST_SUBJECTS, window_size=WINDOW_SIZE, stride=STRIDE)
    
    if len(train_dataset) == 0 or len(test_dataset) == 0:
        print("Dataset is empty. Exiting.")
        return

    # ----- Apply StandardScaler (Fit on Train, Transform Test) -----
    print("Applying StandardScaler...")
    scaler = StandardScaler()
    N_tr, T, F = train_dataset.windows.shape
    train_2d = train_dataset.windows.reshape(-1, F)
    train_dataset.windows = scaler.fit_transform(train_2d).reshape(N_tr, T, F)
    
    N_te, _, _ = test_dataset.windows.shape
    test_2d = test_dataset.windows.reshape(-1, F)
    test_dataset.windows = scaler.transform(test_2d).reshape(N_te, T, F)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 2. Initialize Model, Loss, Optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nTraining on Device: {device}")
    
    model = CSIMultiTaskLSTM(input_dim=664, hidden_dim=128, num_layers=2, num_actions=5, num_zones=4).to(device)
    
    criterion_action = nn.CrossEntropyLoss()
    criterion_zone = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 3. Training Loop
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        
        for batch_x, batch_y_act, batch_y_zone in train_loader:
            batch_x = batch_x.to(device)
            batch_y_act = batch_y_act.to(device)
            batch_y_zone = batch_y_zone.to(device)
            
            optimizer.zero_grad()
            
            # Forward Pass (Predict both)
            preds_act, preds_zone = model(batch_x)
            
            # Combine Losses (Equal weighting for now)
            loss_act = criterion_action(preds_act, batch_y_act)
            loss_zone = criterion_zone(preds_zone, batch_y_zone)
            loss = loss_act + loss_zone
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * batch_x.size(0)
            
        epoch_loss = running_loss / len(train_dataset)
        
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
                
                preds_act, preds_zone = model(batch_x)
                
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
        act_acc = accuracy_score(all_act_trues, all_act_preds)
        zone_acc = accuracy_score(all_zone_trues, all_zone_preds)
        
        print(f"Epoch [{epoch}/{EPOCHS}] Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} "
              f"| Val Action Acc: {act_acc:.4f} | Val Zone Acc: {zone_acc:.4f}")

    # 4. Final Evaluation Report
    print("\n================== FINAL TEST EVALUATION ==================")
    print("\n[Action Classification Report]")
    print(classification_report(all_act_trues, all_act_preds, target_names=['benddown', 'handsup', 'stand', 'walk', 'empty']))
    
    print("\n[Zone Classification Report]")
    print(classification_report(all_zone_trues, all_zone_preds, target_names=['Zone 1', 'Zone 2', 'Zone 3', 'Zone 4']))

    # Save Model Weights
    os.makedirs('weights', exist_ok=True)
    torch.save(model.state_dict(), 'weights/multi_task_lstm.pth')
    print("Model saved to weights/multi_task_lstm.pth")

    # ----- Output Confusion Matrices -----
    print("\nGenerating Confusion Matrix Plots...")
    action_names = ['benddown', 'handsup', 'stand', 'walk', 'empty']
    zone_names = ['Zone 1', 'Zone 2', 'Zone 3', 'Zone 4']
    
    cm_act = confusion_matrix(all_act_trues, all_act_preds)
    cm_zone = confusion_matrix(all_zone_trues, all_zone_preds)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    sns.heatmap(cm_act, annot=True, fmt='d', cmap='Blues', 
                xticklabels=action_names, yticklabels=action_names, ax=axes[0])
    axes[0].set_title(f'Action Classification\nAcc: {act_acc * 100:.2f}%', pad=20)
    axes[0].set_ylabel('Actual (True)')
    axes[0].set_xlabel('Predicted')
    axes[0].xaxis.tick_top()
    axes[0].xaxis.set_label_position('top')
    
    sns.heatmap(cm_zone, annot=True, fmt='d', cmap='Oranges', 
                xticklabels=zone_names, yticklabels=zone_names, ax=axes[1])
    axes[1].set_title(f'Zone Classification\nAcc: {zone_acc * 100:.2f}%', pad=20)
    axes[1].set_ylabel('Actual (True)')
    axes[1].set_xlabel('Predicted')
    axes[1].xaxis.tick_top()
    axes[1].xaxis.set_label_position('top')

    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/multi_task_confusion_matrix.png')
    print("Confusion matrix plots saved to 'results/multi_task_confusion_matrix.png'.")
    plt.show()

if __name__ == "__main__":
    train_multi_task_model()
