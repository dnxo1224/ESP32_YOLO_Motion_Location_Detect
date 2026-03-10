import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from dataset import create_dataloaders, ACTION_NAMES
from model import CSIClassifier


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    
    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * batch_x.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(batch_y).sum().item()
        total += batch_y.size(0)
    
    return total_loss / total, 100. * correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            total_loss += loss.item() * batch_x.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(batch_y).sum().item()
            total += batch_y.size(0)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    
    acc = 100. * correct / total
    return total_loss / total, acc, np.array(all_preds), np.array(all_labels)


def main():
    # 하이퍼파라미터
    DATA_DIR = "/Users/seolwootae/ESP32_YOLO/preprocessing/processed_tensors"
    SAVE_DIR = "/Users/seolwootae/ESP32_YOLO/training/checkpoints"
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    EPOCHS = 50
    BATCH_SIZE = 8
    LR = 1e-3
    NUM_CLASSES = 4
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # DataLoader
    train_loader, val_loader = create_dataloaders(
        DATA_DIR, label_type='action', val_ratio=0.2, batch_size=BATCH_SIZE
    )
    
    # Model
    model = CSIClassifier(num_classes=NUM_CLASSES).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    best_val_acc = 0
    
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, preds, labels = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        
        print(f"Epoch [{epoch}/{EPOCHS}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.1f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.1f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'best_action_model.pth'))
            print(f"  ★ Best model saved! Val Acc: {val_acc:.1f}%")
    
    # 최종 평가
    print("\n" + "="*50)
    print("Final Evaluation on Validation Set")
    print("="*50)
    
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'best_action_model.pth')))
    _, final_acc, preds, labels = evaluate(model, val_loader, criterion, device)
    
    target_names = [ACTION_NAMES[i] for i in range(NUM_CLASSES)]
    print(f"\nBest Validation Accuracy: {final_acc:.1f}%")
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=target_names))
    print("\nConfusion Matrix:")
    print(confusion_matrix(labels, preds))


if __name__ == "__main__":
    main()
