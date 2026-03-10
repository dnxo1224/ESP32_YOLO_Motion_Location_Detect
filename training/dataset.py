import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# 행동 라벨 매핑
ACTION_CLASSES = {'benddown': 0, 'handsup': 1, 'walk': 2, 'stand': 3}
ACTION_NAMES = {v: k for k, v in ACTION_CLASSES.items()}

# 위치 라벨 매핑 (1~16)
POSITION_CLASSES = {str(i): i - 1 for i in range(1, 17)}
POSITION_NAMES = {v: f'pos_{k}' for k, v in POSITION_CLASSES.items()}


class CSIDataset(Dataset):
    """
    .npy 텐서 파일을 직접 로드하는 PyTorch Dataset.
    각 텐서는 (900, 166, 4) 형태이며, 파일명에서 라벨을 추출합니다.
    파일명 형식: {subject}_{action}_{position}.npy
    """
    def __init__(self, file_paths, label_type='action'):
        """
        Args:
            file_paths: .npy 파일 경로 리스트
            label_type: 'action' 또는 'position'
        """
        self.file_paths = file_paths
        self.label_type = label_type
        self.labels = []
        
        for fp in file_paths:
            basename = os.path.splitext(os.path.basename(fp))[0]
            parts = basename.split('_')
            # {subject}_{action}_{position}
            action = parts[1]
            position = parts[2]
            
            if label_type == 'action':
                self.labels.append(ACTION_CLASSES[action])
            else:
                self.labels.append(POSITION_CLASSES[position])
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        # (900, 166, 4) 텐서 로드
        tensor = np.load(self.file_paths[idx]).astype(np.float32)
        
        # PyTorch Conv2d는 (C, H, W) 형식을 기대
        # (900, 166, 4) → (4, 900, 166)
        tensor = np.transpose(tensor, (2, 0, 1))
        
        tensor_torch = torch.from_numpy(tensor)
        label = self.labels[idx]
        
        return tensor_torch, label


def get_all_npy_files(data_dir):
    """processed_tensors 폴더에서 모든 .npy 파일을 재귀적으로 수집"""
    files = []
    for root, dirs, filenames in os.walk(data_dir):
        for fn in filenames:
            if fn.endswith('.npy'):
                files.append(os.path.join(root, fn))
    return sorted(files)


def create_dataloaders(data_dir, label_type='action', val_ratio=0.2, batch_size=8, seed=42):
    """
    학습/검증 DataLoader 생성 (Stratified Random Split)
    """
    all_files = get_all_npy_files(data_dir)
    
    # stand 데이터가 없는 swt 고려: 3가지 행동만 포함된 파일 필터
    if label_type == 'action':
        filtered = [f for f in all_files if any(a in os.path.basename(f) for a in ACTION_CLASSES.keys())]
    else:
        filtered = [f for f in all_files if any(a in os.path.basename(f) for a in ACTION_CLASSES.keys())]
    
    # 라벨 추출
    labels = []
    for fp in filtered:
        parts = os.path.splitext(os.path.basename(fp))[0].split('_')
        if label_type == 'action':
            labels.append(ACTION_CLASSES[parts[1]])
        else:
            labels.append(POSITION_CLASSES[parts[2]])
    
    # Stratified Split
    train_files, val_files, train_labels, val_labels = train_test_split(
        filtered, labels, test_size=val_ratio, random_state=seed, stratify=labels
    )
    
    train_dataset = CSIDataset(train_files, label_type=label_type)
    val_dataset = CSIDataset(val_files, label_type=label_type)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"[{label_type.upper()}] Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    return train_loader, val_loader


if __name__ == "__main__":
    data_dir = "/Users/seolwootae/ESP32_YOLO/preprocessing/processed_tensors"
    
    # 테스트: 행동 분류 DataLoader
    train_loader, val_loader = create_dataloaders(data_dir, label_type='action', batch_size=4)
    
    for batch_x, batch_y in train_loader:
        print(f"Batch shape: {batch_x.shape}, Labels: {batch_y}")
        break
    
    # 테스트: 위치 추정 DataLoader
    train_loader, val_loader = create_dataloaders(data_dir, label_type='position', batch_size=4)
    
    for batch_x, batch_y in train_loader:
        print(f"Batch shape: {batch_x.shape}, Labels: {batch_y}")
        break
