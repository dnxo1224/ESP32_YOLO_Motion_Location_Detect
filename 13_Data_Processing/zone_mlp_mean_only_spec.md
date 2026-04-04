# ZoneMLP (mean_only 모드) 모델 스펙

## 1. 개요

| 항목 | 값 |
|---|---|
| 모델 이름 | ZoneMLP (mean_only) |
| 태스크 | Zone 분류 (4-class) |
| 입력 원본 | Wi-Fi CSI — 4 RX × 166 서브캐리어 |
| 학습 파일 | `/train_zone_mlp_872.py` |
| 체크포인트 | `/weights/zone_mlp_mean_only_best.pt` |

---

## 2. 입력 파이프라인

### 2-1. 원시 데이터 형태
- NPZ 파일 (`LightGBM/processed/*.npz`)에서 로드
- `grids`: `(4, 800, 166)` → transpose + reshape → `(800, 664)`
  - 800: 시퀀스 길이(프레임)
  - 664 = 166 서브캐리어 × 4 RX 안테나

### 2-2. 슬라이딩 윈도우
| 파라미터 | 값 |
|---|---|
| window_size | 200 |
| stride | 10 |
| 윈도우당 샘플 shape | `(200, 664)` |

### 2-3. 정규화
- `StandardScaler` — 학습 데이터로 fit, 학습/테스트 모두 transform
- feature 단위(664차원)로 적용 (`reshape(-1, 664)` 후 fit)

### 2-4. 데이터 분할
| 분할 | 피험자 |
|---|---|
| Train | 12명 (kms 제외 전체) |
| Test | kms (1명, 외부 피험자 평가) |

---

## 3. 특징 추출 (mean_only)

```
입력: (B, 200, 664)
    ↓  시간축(dim=1) 평균
출력: (B, 664)
```

- 시간축 평균(mean)만 사용 — std, max, min 통계 미사용
- 시계열 정보를 단일 벡터로 압축

---

## 4. 모델 구조

```
입력: (B, 664)
  Linear(664 → 512)
  BatchNorm1d(512)
  ReLU
  Dropout(p=0.3)
  Linear(512 → 256)
  BatchNorm1d(256)
  ReLU
  Dropout(p=0.3)
  Linear(256 → 128)
  BatchNorm1d(128)
  ReLU
  Dropout(p=0.2)
  Linear(128 → 4)
출력: (B, 4)  — 4개 zone 클래스 logit
```

### 레이어 요약

| 레이어 | 입력 → 출력 | 추가 |
|---|---|---|
| Linear 1 | 664 → 512 | BN + ReLU + Dropout(0.3) |
| Linear 2 | 512 → 256 | BN + ReLU + Dropout(0.3) |
| Linear 3 | 256 → 128 | BN + ReLU + Dropout(0.2) |
| Linear 4 (출력) | 128 → 4 | — |

---

## 5. 학습 설정

| 항목 | 값 |
|---|---|
| Optimizer | Adam |
| Learning Rate | 1e-3 |
| Weight Decay | 1e-4 |
| LR Scheduler | ReduceLROnPlateau (patience=5, factor=0.5) |
| Loss | CrossEntropyLoss |
| Epochs | 50 |
| Batch Size | 256 |
| 체크포인트 기준 | Test accuracy 최고점 |

---

## 6. 레이블 정의

### Zone 매핑 (수집 위치 → Zone)

| Zone | 수집 위치 번호 |
|---|---|
| Zone 0 | 1, 2, 5, 6 |
| Zone 1 | 3, 4, 7, 8 |
| Zone 2 | 9, 10, 13, 14 |
| Zone 3 | 11, 12, 15, 16 |

- 총 4-class 분류 (2×2 그리드 기반 구역)

---

## 7. 출력

- 혼동 행렬 이미지: `/results/zone_mlp_mean_only_cm.png`
- 최적 가중치: `/weights/zone_mlp_mean_only_best.pt`

