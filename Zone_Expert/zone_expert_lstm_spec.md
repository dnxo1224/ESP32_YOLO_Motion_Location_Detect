# Zone Expert Action Classifier — 모델 스펙

## 1. 개요

| 항목 | 값 |
|---|---|
| 모델 이름 | ZoneExpertLSTM |
| 태스크 | Zone별 행동 분류 (4-class) |
| 입력 원본 | Wi-Fi CSI — 4 RX × 166 서브캐리어 |
| 전문가 수 | 4개 (Zone 0 ~ Zone 3 각 1개) |
| 학습 파일 | `Zone_Expert/train.py` |
| 체크포인트 | `Zone_Expert/weights/zone_expert_action_{0~3}_best.pt` |

---

## 2. 입력 파이프라인

### 2-1. 원시 데이터 구조

```
NPZ 파일 (LightGBM/processed/*.npz)
  grids: (4, 800, 166)
    → .transpose(1, 2, 0) → (800, 166, 4)
    → .reshape(800, 664)  → (800, 664)
```

664차원 벡터 내 RX 배치 **(인터리브 구조)**:

| RX 채널 | 코드 인덱스 | 664D 내 위치 |
|---|---|---|
| RX0 (Zone 0 대응) | 0 | 0, 4, 8, …, 660 |
| RX1 (Zone 1 대응) | 1 | 1, 5, 9, …, 661 |
| RX2 (Zone 2 대응) | 2 | 2, 6, 10, …, 662 |
| RX3 (Zone 3 대응) | 3 | 3, 7, 11, …, 663 |

### 2-2. 전처리 순서

```
(800, 664) 원시 데이터
    ↓
StandardScaler 정규화
    train 데이터 기준 fit → feature별 mean=0, std=1
    ↓
슬라이딩 윈도우 생성
    window_size = 200,  stride = 20
    → (200, 664) 윈도우
    ↓
윈도우별 평균 제거
    window = window - window.mean(axis=0)
    추론 시에도 동일하게 적용 가능 (학습/추론 조건 일치)
```

### 2-3. 데이터 분할

| 분할 | 피험자 | 비고 |
|---|---|---|
| Train | 12명 (kms 제외) | 전 Zone 데이터 포함 |
| Test | kms (1명) | Zone ZONE_ID 데이터만 필터링하여 평가 |

### 2-4. Zone별 오버샘플링 (의도적 데이터 불균형)

```python
train_win.oversample_zone(ZONE_ID, factor=2)
```

| Zone | 오버샘플링 전 | 오버샘플링 후 |
|---|---|---|
| Zone ZONE_ID (담당) | N개 | N × 2개 |
| 나머지 3개 Zone | M개 (각) | M개 (변동 없음) |

담당 Zone 데이터를 2배로 복제하여 모델이 더 많이 학습하도록 의도적 불균형을 생성한다.

---

## 3. 모델 구조 (`ZoneExpertLSTM`)

### 레이어 구성

```
입력: (B, 200, 664)
  ↓
LSTM (단방향, 2층)
  input_size  = 664
  hidden_size = 128
  num_layers  = 2
  dropout     = 0.3  (레이어 간)
  bidirectional = False
  → hn[-1]: (B, 128)  ← 마지막 레이어 최종 hidden state
  ↓
Linear(128 → 64)
ReLU
Dropout(0.3)
Linear(64 → 4)
  ↓
출력: logits (B, 4)
```

### 파라미터 수 추정

| 레이어 | 파라미터 |
|---|---|
| LSTM layer 1 | 4 × (664 + 128 + 1) × 128 ≈ 408,576 |
| LSTM layer 2 | 4 × (128 + 128 + 1) × 128 ≈ 131,584 |
| FC (128→64) | 128 × 64 + 64 = 8,256 |
| FC (64→4) | 64 × 4 + 4 = 260 |
| **합계** | **≈ 548,676** |

---

## 4. 학습 설정

| 항목 | 값 |
|---|---|
| Optimizer | Adam |
| Learning Rate | 1e-3 |
| Weight Decay | 1e-4 |
| LR Scheduler | CosineAnnealingLR (T_max=50) |
| Loss | CrossEntropyLoss |
| Epochs | 50 |
| Batch Size | 64 |
| Gradient Clipping | max_norm = 1.0 |
| 체크포인트 기준 | Test accuracy 최고점 (윈도우 단위) |

---

## 5. 평가 방식

- 실시간 추론 조건과 동일: 윈도우 1개 → 즉시 action 예측
- 테스트 데이터: kms의 **Zone ZONE_ID 데이터만** 필터링하여 평가

```
테스트 윈도우 (200, 664)
    ↓ rx_weight 적용 (추론 모드: 항상 적용)
    ↓ LSTM → hidden state
    ↓ FC → logits
    → argmax → action 예측
```

---

## 6. Zone → RX 매핑

| Zone | 사용자 명칭 | 코드 인덱스 | 담당 위치 번호 |
|---|---|---|---|
| Zone 0 | RX1 | RX0 | 1, 2, 5, 6 |
| Zone 1 | RX2 | RX1 | 3, 4, 7, 8 |
| Zone 2 | RX3 | RX2 | 9, 10, 13, 14 |
| Zone 3 | RX4 | RX3 | 11, 12, 15, 16 |

---

## 7. 레이블 정의

### Action (행동)

| 레이블 | 행동 |
|---|---|
| 0 | handsup |
| 1 | sit |
| 2 | stand |
| 3 | walk |

### Zone (구역)

| 레이블 | 위치 번호 |
|---|---|
| 0 | 1, 2, 5, 6 |
| 1 | 3, 4, 7, 8 |
| 2 | 9, 10, 13, 14 |
| 3 | 11, 12, 15, 16 |

---

## 8. 추론 파이프라인 (전체 시스템)

```
입력 데이터 (800, 664)
    ↓ StandardScaler 정규화 (저장된 scaler 적용)
    ↓ 슬라이딩 윈도우 (200, stride=10)
    ↓
    ├─→ ZoneMLP (mean_only)         → zone 예측 (0~3)
    │       윈도우 시간축 평균 → MLP
    │
    └─→ 윈도우별 평균 제거
            → ZoneExpertLSTM[zone]  → action 예측 (0~3)
```

---

## 9. 출력 파일

| 파일 | 내용 |
|---|---|
| `weights/zone_expert_action_{0~3}_best.pt` | Zone별 최적 가중치 |
| `results/zone_expert_action_{0~3}_cm.png` | Zone별 혼동행렬 (학습 평가) |
| `results/pipeline_action_cm.png` | 전체 파이프라인 행동 분류 혼동행렬 |
| `results/pipeline_zone{0~3}_expert_cm.png` | 파이프라인 Zone별 행동 분류 혼동행렬 |

---

## 10. 파일 구조

```
Zone_Expert/
├── dataset.py                      ← 데이터 로드 / 정규화 / 슬라이딩 윈도우
├── model.py                        ← ZoneExpertLSTM 모델 정의
├── train.py                        ← 학습 스크립트 (ZONE_ID 변수로 전문가 선택)
├── inference.py                    ← ZoneMLP + ZoneExpertLSTM 파이프라인 평가
├── zone_expert_lstm_spec.md        ← 본 문서
├── weights/
│   ├── zone_expert_action_0_best.pt
│   ├── zone_expert_action_1_best.pt
│   ├── zone_expert_action_2_best.pt
│   └── zone_expert_action_3_best.pt
└── results/
    ├── zone_expert_action_{0~3}_cm.png
    ├── pipeline_zone_cm.png
    ├── pipeline_action_cm.png
    └── pipeline_zone{0~3}_expert_cm.png
```
