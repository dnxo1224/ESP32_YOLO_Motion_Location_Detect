# Zone Expert CNN — 모델 스펙 (논문용)

## 1. 전체 파이프라인 위치

```
WiFi CSI 원시 데이터 (4 RX × 166 subcarriers × 800 frames)
    │
    ├─ ZoneMLP  →  Zone 예측 (0~3)
    │
    └─ ZoneExpertCNN[predicted_zone]  →  행동 분류 (4 classes)
```

Zone별로 독립된 CNN 전문가 모델 4개를 학습하며,
추론 시 ZoneMLP의 예측 결과에 따라 해당 Zone 전문가 모델로 라우팅한다.

---

## 2. 입력 데이터 구성

| 항목 | 값 |
|------|----|
| 원시 시퀀스 길이 | 800 frames |
| RX 안테나 수 | 4 |
| 유효 서브캐리어 수 | 166 |
| 특징 벡터 차원 | 4 × 166 = **664** |
| 원시 데이터 형상 | (800, 664) per sample |

**인터리브 구조:** 664D 벡터는 RX 채널이 stride=4로 인터리브 배열됨
→ RX_k 채널: 인덱스 `k, k+4, k+8, ..., k+660` (166개, k = 0,1,2,3)

---

## 3. 전처리

### 3-1. 정규화
- 학습 데이터 전체로 `StandardScaler` (zero-mean, unit-variance) 적합
- 학습·테스트 데이터 모두에 동일 스케일러 적용 (train-fit, test-transform)

### 3-2. 슬라이딩 윈도우

| 항목 | 학습 | 추론 |
|------|------|------|
| 윈도우 크기 | 200 frames | 200 frames |
| 스트라이드 | 20 frames | 10 frames |
| 샘플당 윈도우 수 | 31개 | 61개 |
| 윈도우 형상 | (200, 664) | (200, 664) |

### 3-3. 윈도우별 평균 제거 (Mean Removal)
각 윈도우에 대해 서브캐리어 축 기준 시간 평균을 제거:

$$\tilde{X}_{t,f} = X_{t,f} - \frac{1}{T}\sum_{t=1}^{T} X_{t,f}$$

여기서 $T = 200$ (윈도우 길이), $f$ = 서브캐리어 인덱스 (0~663)

---

## 4. Zone-Aware RX 채널 가중치

각 Zone 전문가 모델은 해당 Zone에 물리적으로 인접한 RX 안테나 채널을
입력 단계에서 강조하는 **정적(static) 채널 가중치 벡터**를 보유한다.

| Zone | 담당 RX 채널 (코드 인덱스) | 물리 위치 (측정 포지션) |
|------|--------------------------|-----------------------|
| Zone 0 | RX 0 | 1, 2, 5, 6 |
| Zone 1 | RX 1 | 3, 4, 7, 8 |
| Zone 2 | RX 2 | 9, 10, 13, 14 |
| Zone 3 | RX 3 | 11, 12, 15, 16 |

가중치 벡터 $\mathbf{w} \in \mathbb{R}^{664}$ 는 학습 파라미터가 아니며, 모델 초기화 시 고정된다:

$$w_i = \begin{cases} w_{\text{high}} & \text{if } i \bmod 4 = \text{RX\_code\_idx} \\ w_{\text{low}} & \text{otherwise} \end{cases}$$

입력에 element-wise 곱으로 적용:

$$\hat{X} = X \odot \mathbf{w}$$

---

## 5. 모델 아키텍처

### 5-1. 입출력 형상

| 단계 | 텐서 형상 |
|------|----------|
| 입력 | (B, 200, 664) |
| RX 가중치 적용 후 | (B, 200, 664) |
| Permute (Conv1d 형식) | (B, 664, 200) |
| Conv Block 1~2 출력 | (B, 128, 200) |
| MaxPool1d 출력 | (B, 128, 50) |
| Conv Block 3~4 출력 | (B, 128, 50) |
| AdaptiveAvgPool1d 출력 | (B, 128, 1) → squeeze → (B, 128) |
| 분류기 출력 (logits) | (B, 4) |

### 5-2. 레이어 구성

```
Input: (B, 200, 664)
  │
  ├─ [RX Weight]  element-wise × w(664,)
  │
  └─ permute(0,2,1) → (B, 664, 200)
        │
        ├─ Conv Block 1
        │    Conv1d(664 → 128, kernel=5, padding=2)
        │    BatchNorm1d(128)
        │    ReLU
        │    → (B, 128, 200)
        │
        ├─ Conv Block 2
        │    Conv1d(128 → 128, kernel=5, padding=2)
        │    BatchNorm1d(128)
        │    ReLU
        │    → (B, 128, 200)
        │
        ├─ MaxPool1d(kernel=4, stride=4)
        │    → (B, 128, 50)
        │
        ├─ Conv Block 3
        │    Conv1d(128 → 128, kernel=5, padding=2)
        │    BatchNorm1d(128)
        │    ReLU
        │    → (B, 128, 50)
        │
        ├─ Conv Block 4
        │    Conv1d(128 → 128, kernel=3, padding=1)
        │    BatchNorm1d(128)
        │    ReLU
        │    → (B, 128, 50)
        │
        └─ AdaptiveAvgPool1d(1) → squeeze
               → (B, 128)
                    │
                    ├─ Dropout(0.3)
                    ├─ Linear(128 → 64)
                    ├─ ReLU
                    ├─ Dropout(0.3)
                    └─ Linear(64 → 4)
                         → logits (B, 4)
```

### 5-3. 레이어별 파라미터 수

| 레이어 | 파라미터 수 |
|--------|------------|
| Conv Block 1: Conv1d(664→128, k=5) | 664×128×5 + 128 = **425,088** |
| Conv Block 1: BatchNorm1d(128) | 128×2 = **256** |
| Conv Block 2: Conv1d(128→128, k=5) | 128×128×5 + 128 = **82,048** |
| Conv Block 2: BatchNorm1d(128) | **256** |
| Conv Block 3: Conv1d(128→128, k=5) | 128×128×5 + 128 = **82,048** |
| Conv Block 3: BatchNorm1d(128) | **256** |
| Conv Block 4: Conv1d(128→128, k=3) | 128×128×3 + 128 = **49,280** |
| Conv Block 4: BatchNorm1d(128) | **256** |
| Linear(128 → 64) | 128×64 + 64 = **8,256** |
| Linear(64 → 4) | 64×4 + 4 = **260** |
| **합계** | **~648,004** |

> BatchNorm의 running_mean, running_var는 학습 파라미터가 아니므로 제외

---

## 6. 학습 설정

| 항목 | 값 |
|------|----|
| 손실 함수 | CrossEntropyLoss |
| 옵티마이저 | Adam |
| 학습률 | 1e-3 |
| Weight Decay | 1e-4 |
| LR 스케줄러 | CosineAnnealingLR (T\_max = 50) |
| Gradient Clipping | max\_norm = 1.0 |
| 배치 크기 | 64 |
| 에폭 수 | 50 |
| Dropout | 0.3 |

---

## 7. 데이터셋 구성

| 항목 | 값 |
|------|----|
| 전체 피험자 수 | 13명 |
| 학습 피험자 수 | 12명 |
| 테스트 피험자 | 1명 (Leave-One-Subject-Out) |
| 행동 클래스 | 4종 (handsup, sit, stand, walk) |
| Zone 수 | 4 |
| 측정 포지션 수 | 16 (Zone당 4개) |
| 샘플 수 (학습) | 12명 × 64샘플 = 768 |
| 샘플 수 (테스트) | 1명 × 64샘플 = 64 |
| 학습 윈도우 수 (오버샘플링 전) | 768 × 31 = 23,808 |
| 테스트 윈도우 수 (Zone 필터 후) | 16 × 31 = 496 per Zone |

**오버샘플링:** 학습 시 해당 Zone 데이터를 ×2 복제하여 Zone 특화 학습 강화

---

## 8. 평가 방식

- **윈도우 단위 즉시 예측 (window-level prediction)**
- **Best checkpoint 기준:** 테스트 정확도가 가장 높은 에폭의 가중치 저장
- 최종 평가: best checkpoint 로드 후 Classification Report + Confusion Matrix 출력

---

## 9. 추론 파이프라인

```
테스트 윈도우 (B, 200, 664)
    │
    ├─ ZoneMLP (시간축 평균 → 664D MLP) → zone_pred (B,)
    │
    ├─ 윈도우 평균 제거: x -= x.mean(dim=1, keepdim=True)
    │
    └─ for z in [0,1,2,3]:
           mask = (zone_pred == z)
           logits = ZoneExpertCNN[z](x[mask])   ← rx_weight 항상 적용
           action_pred[mask] = logits.argmax(1)
```
