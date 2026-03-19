# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ESP32 Wi-Fi CSI (Channel State Information) based human activity recognition and indoor localization system. Raw CSI data collected from ESP32 devices is preprocessed into tensors, then classified using deep learning models for two tasks:
- **Action classification**: handsup, sit, stand, walk (4 classes)
- **Zone/Position classification**: 16 collection positions mapped to 4 zones (2x2 grid)

## Running Scripts

The project uses a Python venv at `/Users/seolwootae/ESP-CSI/.venv/`. Run scripts with:
```bash
/Users/seolwootae/ESP-CSI/.venv/bin/python <script.py>
```

Device: macOS with Apple Silicon (MPS backend for PyTorch). Training scripts auto-detect MPS vs CPU.

## Key Dependencies

PyTorch, ultralytics (YOLOv8), scikit-learn, pandas, numpy, scipy, matplotlib, seaborn, tqdm

## Architecture

### Data Pipeline (3 stages)

1. **Raw CSI collection** (`13_raw_data/`): CSV files per subject (13 subjects: gyj, jhj, jkw, kjh, kmh, etc.), each with 4 receiver antennas (rx1-rx4). Each row contains complex I/Q values for subcarriers.

2. **Preprocessing** (`preprocessing/`): Three parallel pipelines exist:
   - **Hampel pipeline** (`Hampel_LPF_Spline/`): Step-by-step scripts (1_ through 6_) — amplitude extraction → null subcarrier removal → spline interpolation → Hampel outlier filter → Butterworth low-pass filter → 3D tensor merge → min-max normalization. Outputs `(900, 166, 4)` tensors.
   - **Batch pipeline** (`batch_preprocess.py`): All-in-one version combining the same steps. Reads from `data/` directory, outputs `.npy` files to `processed_tensors/`.
   - **Linear interpolation pipeline** (current): Raw CSI → amplitude extraction → HT-LTF null subcarrier removal (192→114) → 872-frame alignment → linear interpolation with bfill/ffill. Outputs per-RX CSVs to `data_interpolated_linear_872/`.

   Key constants (linear pipeline): 192 raw subcarriers → 114 valid (HT-LTF only, 78 removed). Sequence length: 872 frames. 4 RX antennas concatenated → 456 features (114×4). Combined tensor shape: `(872, 456)`.

3. **Intermediate data formats**:
   - `data_interpolated_linear_872/`: Linearly interpolated CSVs per rx, named `{subject}_{action}_{position}_rx{1-4}_872.csv`. Each CSV: 872 rows × 114 subcarrier columns.
   - `data_interpolated_spline_800/`: Spline-interpolated CSVs per rx, named `{subject}_{action}_{position}_rx{1-4}_800.csv`
   - `processed_tensors/`: Final `.npy` files named `{subject}_{action}_{position}.npy`

### Models (3 approaches)

1. **YOLOv8-based classifier** (`training/`):
   - `model.py`: Wraps YOLOv8n-cls with 4-channel input Conv2d (instead of 3-channel RGB). Pretrained weights in `yolov8n-cls.pt`.
   - `dataset.py`: Loads `.npy` tensors `(900, 166, 4)` → transposes to `(4, 900, 166)` for Conv2d. File naming: `{subject}_{action}_{position}.npy`.
   - `train_action.py`: Action classification (4 classes), 50 epochs, Adam + CosineAnnealing.
   - `train_position.py`: Position classification (16 classes), same setup.
   - Checkpoints saved to `training/checkpoints/`.

2. **Dual-Stream Transformer** (`model/`):
   - `dual_stream_transformer.py`: Cross-attention between temporal stream `(B, 800, 664)` and channel stream `(B, 664, 800)`. Outputs both action (5 classes) and zone (4 classes) predictions simultaneously.
   - `train_dual_stream.py`: Subject-split evaluation (train: swt/kjh/kmh, test: jhj). Uses StandardScaler normalization. Reads directly from `data_interpolated_spline_800/` CSVs.
   - `train_zone_mlp.py`: Simpler MLP baseline for zone classification. Also reads from CSVs.

3. **872×114×4 Linear Interpolation Models** (`13_Data_Processing/`):
   - `dataset_872.py`: Shared Dataset class. Loads 4 RX CSVs from `data_interpolated_linear_872/` → `(872, 456)` tensor. Subject-based train/test split (12 train, kjh test).
   - `visualize_csi.py`: CSI data visualization (heatmaps, time-series, zone profiles). Saves to `13_Data_Processing/results/`.
   - `train_zone_mlp_872.py`: MLP zone classifier. Time-averaged `(B,456)` input → FC layers → 4 zones.
   - `train_zone_lstm_872.py`: LSTM zone classifier. Full sequence `(B,872,456)` → LSTM → 4 zones.
   - `dual_stream_transformer_872.py`: Adapted transformer. `(B,872,456)` input with cross-attention.
   - `train_zone_transformer_872.py`: Transformer zone training script.
   - `train_action_zone_weighted.py`: Zone-weighted action classifier. Predicts zone first, then uses zone-specific action model with 0.7/0.1 weight distribution.
   - Model checkpoints saved to `13_Data_Processing/weights/`.

### Label Conventions

- **Actions (872 pipeline)**: handsup=0, sit=1, stand=2, walk=3 (4 classes)
- **Actions (legacy)**: benddown=0, handsup=1, walk=2, stand=3, empty=4 (5 classes, YOLOv8 pipeline uses only first 4)
- **Zones from positions**: positions 1,2,5,6→Zone0; 3,4,7,8→Zone1; 9,10,13,14→Zone2; 11,12,15,16→Zone3
- File naming pattern: `{subject}_{action}_{position}` (e.g., `jhj_walk_3`)
- **Train/Test split (872 pipeline)**: 12 subjects train, kjh test. 13 subjects: gyj, jhj, jkw, kjh, kmh, kms, kye, lsi, mhe, phr, stk, swt, ysj

## Important Notes

- The `dual_stream_transformer.py` has dead code after the first `return` statement (duplicate classification block).
- Training scripts have hardcoded absolute paths to data directories that need updating per machine.
- Three tensor shape conventions: YOLOv8 uses `(4, 900, 166)` image-like input; legacy Dual-Stream uses `(800, 664)` sequence input; 872 pipeline uses `(872, 456)` sequence input.
- 절대 코드를 임의로 실행하지 말 것. 파일을 생성하는 것은 가능하되, 실행은 반드시 사용자가 직접 할 것.

## Coding Conventions

- **딥러닝 학습 스크립트에는 반드시 `tqdm`을 사용할 것.** epoch 루프, batch 루프 모두에 tqdm 프로그레스 바를 적용하여 학습 진행상황을 실시간으로 확인할 수 있게 한다.
  ```python
  from tqdm import tqdm
  for epoch in tqdm(range(EPOCHS), desc="Epochs"):
      for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
          ...
  ```
