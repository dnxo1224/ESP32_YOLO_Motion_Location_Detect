"""
Zone 가중치 기반 행동 분류기 (Phase 3) - Transformer 적용 (Window/Packet 단위 평가)
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from dataset_872 import (
    CSIDataset872, normalize_data, get_device,
    TRAIN_SUBJECTS, TEST_SUBJECT, FEATURE_DIM,
    ACTION_MAP, ZONE_MAP
)
from train_zone_lstm_872 import (
    ZoneLSTM, SlidingWindowDataset,
    WINDOW_SIZE, STRIDE
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), 'weights')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)

ACTION_NAMES = ['handsup', 'sit', 'stand', 'walk']
ZONE_HIGH_W = 0.7   
ZONE_LOW_W  = 0.1   


# ─── 모델 정의 ────────────────────────────────────────────────────────────────

class ActionDualStreamTransformer(nn.Module):
    """
    Dual Stream Transformer Action 분류기 
    입력: (Batch, 90, 456)
    """
    def __init__(self, input_dim=FEATURE_DIM, seq_len=WINDOW_SIZE, num_classes=4, 
                 d_model=128, num_heads=4, dim_feedforward=512, dropout=0.3):
        super().__init__()
        
        self.temporal_proj = nn.Linear(input_dim, d_model)
        self.channel_proj = nn.Linear(seq_len, d_model)
        
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        
        self.fc_action = nn.Linear(d_model, num_classes)

    def forward(self, x):
        temporal_q = self.temporal_proj(x)
        
        x_channel = x.transpose(1, 2)
        channel_kv = self.channel_proj(x_channel)
        
        attn_output, _ = self.cross_attention(
            query=temporal_q,
            key=channel_kv,
            value=channel_kv
        )
        
        x_attn = self.norm1(temporal_q + attn_output)
        ffn_output = self.ffn(x_attn)
        x_out = self.norm2(x_attn + ffn_output)
        
        x_gap = torch.mean(x_out, dim=1)
        logits_action = self.fc_action(x_gap)
        return logits_action

# ─── 슬라이딩 윈도우 ─────────────────────────────────────────────────────────

class ActionSlidingWindowDataset(SlidingWindowDataset):
    def __init__(self, base_dataset, window_size=WINDOW_SIZE, stride=STRIDE):
        self.windows = []
        self.action_labels = []
        self.zone_labels = []

        for sample_idx in range(len(base_dataset)):
            x, action_y, zone_y = base_dataset[sample_idx]
            x_np = x.numpy()
            seq_len = x_np.shape[0]

            start = 0
            while start + window_size <= seq_len:
                self.windows.append(x_np[start:start + window_size])
                self.action_labels.append(action_y.item())
                self.zone_labels.append(zone_y.item())
                start += stride

        print(f"  ActionSlidingWindow: {len(base_dataset)} samples → {len(self.windows)} windows/packets")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        x = torch.tensor(self.windows[idx], dtype=torch.float32)
        y = torch.tensor(self.action_labels[idx], dtype=torch.long)
        return x, y

    def get_zone_weights(self, target_zone):
        return [
            ZONE_HIGH_W if z == target_zone else ZONE_LOW_W
            for z in self.zone_labels
        ]


def evaluate_action_model(model, win_dataset, device):
    """개별 윈도우(패킷) 단위로 정확도 평가"""
    model.eval()
    all_preds, all_labels = [], []

    loader = DataLoader(win_dataset, batch_size=256, shuffle=False)
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(1).cpu().numpy().tolist()
            
            all_preds.extend(preds)
            all_labels.extend(labels.numpy().tolist())

    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    return correct / len(all_labels), all_preds, all_labels


# ─── 학습 함수 ────────────────────────────────────────────────────────────────

def train_action_model(model, train_win_dataset, test_win_dataset, device, model_name,
                       target_zone=None, epochs=50):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4) 
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    if target_zone is not None:
        weights = train_win_dataset.get_zone_weights(target_zone)
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True
        )
        train_loader = DataLoader(train_win_dataset, batch_size=64, sampler=sampler)
    else:
        train_loader = DataLoader(train_win_dataset, batch_size=64, shuffle=True)

    best_acc = 0
    best_epoch = 0

    for epoch in tqdm(range(epochs), desc=f"Action({model_name})"):
        model.train()
        train_loss, train_correct = 0, 0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()

        scheduler.step()
        train_acc = train_correct / len(train_win_dataset)

        win_acc, _, _ = evaluate_action_model(model, test_win_dataset, device)

        if win_acc > best_acc:
            best_acc = win_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(),
                       os.path.join(WEIGHTS_DIR, f'{model_name}.pt'))

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  [{epoch+1:>2}/{epochs}] "
                  f"train_win: {train_acc:.4f} | test_win: {win_acc:.4f}")

    print(f"  ✅ Best: {best_acc:.4f} (epoch {best_epoch})")
    return best_acc


# ─── 메인 ────────────────────────────────────────────────────────────────────

def main():
    device = get_device()
    print(f"Device: {device}")

    print("\n[1] Loading data...")
    train_raw = CSIDataset872(TRAIN_SUBJECTS, task='both')
    test_raw  = CSIDataset872([TEST_SUBJECT],  task='both')
    normalize_data(train_raw, test_raw)

    print(f"Train: {len(train_raw)} samples | Test: {len(test_raw)} samples")

    print("\n[2] Loading pre-trained zone model...")
    zone_model = ZoneLSTM(input_dim=FEATURE_DIM, num_classes=4).to(device)
    zone_ckpt = os.path.join(WEIGHTS_DIR, 'zone_lstm_best.pt')
    zone_model.load_state_dict(torch.load(zone_ckpt, map_location=device))
    zone_model.eval()
    print(f"  Loaded: {zone_ckpt}")

    print("\n[3] Creating sliding window/packet datasets...")
    test_raw_zone = CSIDataset872([TEST_SUBJECT], task='zone')
    test_raw_zone.data = test_raw.data  

    test_zone_win = SlidingWindowDataset(test_raw_zone)

    train_action_win = ActionSlidingWindowDataset(train_raw)
    test_action_win  = ActionSlidingWindowDataset(test_raw)

    print("\n[4] Evaluating zones for test packets...")
    zone_model.eval()
    loader = DataLoader(test_action_win, batch_size=256, shuffle=False) # zone label from action dataset
    
    zone_window_preds = []
    zone_window_labels = []
    
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            outputs = zone_model(inputs)
            preds = outputs.argmax(1).cpu().numpy().tolist()
            zone_window_preds.extend(preds)

    zone_window_labels = test_action_win.zone_labels
    zone_acc = sum(p == l for p, l in zip(zone_window_preds, zone_window_labels)) / len(zone_window_labels)
    print(f"  Zone Acc (per packet): {zone_acc:.4f}")

    print("\n[5] Training baseline transformer action model (no zone weighting)...")
    baseline_model = ActionDualStreamTransformer().to(device)
    baseline_acc = train_action_model(
        baseline_model, train_action_win, test_action_win,
        device, model_name='action_transformer_baseline',
        target_zone=None, epochs=50
    )

    print("\n[6] Training zone-weighted transformer action models...")
    zone_model_accs = {}
    for zone_i in range(4):
        print(f"\n  --- Zone {zone_i} Transformer Action Model "
              f"(high_w={ZONE_HIGH_W}, others={ZONE_LOW_W}) ---")
        model_i = ActionDualStreamTransformer().to(device)
        acc_i = train_action_model(
            model_i, train_action_win, test_action_win,
            device,
            model_name=f'action_transformer_zone_{zone_i}',
            target_zone=zone_i, epochs=50
        )
        zone_model_accs[zone_i] = acc_i

    print("\n[7] Zone-weighted pipeline inference (kjh test set, per packet)...")
    action_models = {}
    for zone_i in range(4):
        m = ActionDualStreamTransformer().to(device)
        m.load_state_dict(torch.load(
            os.path.join(WEIGHTS_DIR, f'action_transformer_zone_{zone_i}.pt'),
            map_location=device))
        m.eval()
        action_models[zone_i] = m

    pipeline_preds = []
    true_actions   = []

    loader = DataLoader(test_action_win, batch_size=256, shuffle=False)
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            # 1. predict zone
            zone_outputs = zone_model(inputs)
            batch_zone_preds = zone_outputs.argmax(1) # (batch_size,)
            
            # 2. for each packet in batch, use corresponding action_model
            batch_action_preds = torch.zeros_like(batch_zone_preds)
            for z in range(4):
                mask = (batch_zone_preds == z)
                if mask.any():
                    action_outputs = action_models[z](inputs[mask])
                    batch_action_preds[mask] = action_outputs.argmax(1)
            
            pipeline_preds.extend(batch_action_preds.cpu().numpy().tolist())
            true_actions.extend(labels.cpu().numpy().tolist())

    pipeline_acc = sum(p == t for p, t in zip(pipeline_preds, true_actions)) / len(true_actions)

    print("\n" + "=" * 60)
    print(f"ACTION CLASSIFICATION RESULTS (Test: kjh, {len(true_actions)} packets)")
    print("=" * 60)
    print(f"  Baseline Transformer (no weight) : {baseline_acc:.4f}")
    print(f"  Zone-Weighted Pipeline (Trans)   : {pipeline_acc:.4f}")
    print(f"    (Zone prediction acc           : {zone_acc:.4f})")
    print("-" * 60)
    print("  Zone-specific model best accs:")
    for z, acc in zone_model_accs.items():
        n_samples = sum(1 for z_ in zone_window_preds if z_ == z)
        print(f"    Zone {z} model: {acc:.4f}  (predicted {n_samples} test packets → this zone)")
    print("=" * 60)

    cm = confusion_matrix(true_actions, pipeline_preds)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=ACTION_NAMES, yticklabels=ACTION_NAMES, ax=axes[0])
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    axes[0].set_title(f'Zone-Weighted Pipeline (Transformer)\nAcc: {pipeline_acc:.4f} (Packets)')

    baseline_model.load_state_dict(torch.load(
        os.path.join(WEIGHTS_DIR, 'action_transformer_baseline.pt'), map_location=device))
    _, base_preds, base_labels = evaluate_action_model(
        baseline_model, test_action_win, device)
    cm_base = confusion_matrix(base_labels, base_preds)
    sns.heatmap(cm_base, annot=True, fmt='d', cmap='Oranges',
                xticklabels=ACTION_NAMES, yticklabels=ACTION_NAMES, ax=axes[1])
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')
    axes[1].set_title(f'Baseline Transformer (No Weighting)\nAcc: {baseline_acc:.4f} (Packets)')

    plt.suptitle(f'Transformer Action Classification Comparison per Packet ({len(true_actions)} packets)', fontsize=13)
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, 'action_zone_weighted_transformer_cm.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\nConfusion matrix saved: {save_path}")

    print("\nClassification Report (Zone-Weighted Pipeline):")
    print(classification_report(true_actions, pipeline_preds, target_names=ACTION_NAMES))

    print("\nClassification Report (Baseline):")
    print(classification_report(base_labels, base_preds, target_names=ACTION_NAMES))


if __name__ == '__main__':
    main()
