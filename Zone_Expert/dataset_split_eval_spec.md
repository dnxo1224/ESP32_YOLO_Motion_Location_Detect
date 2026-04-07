# 데이터셋 · 분할 · 평가 지표 스펙

## 1. 원시 데이터셋 개요

| 항목 | 값 |
|---|---|
| 총 피험자 수 | 13명 |
| 행동 클래스 | 4개 (handsup / sit / stand / walk) |
| 측정 위치 수 | 16개 위치 → 4개 Zone으로 그룹화 |
| 피험자당 샘플 수 | 64개 (4 행동 × 16 위치) |
| 전체 샘플 수 | 약 832개 (13 × 64) |
| RX 안테나 수 | 4개 (RX0 ~ RX3) |
| 유효 서브캐리어 수 | 166개 (원시 192개 중 26개 NULL 제거) |
| 샘플당 타임스텝 | 800 프레임 |
| 특징 차원 (1 프레임) | 664 = 166 서브캐리어 × 4 RX |

---

## 2. NPZ 파일 구조

**파일 경로:** `LightGBM/processed/{subject}_{action}_{position}.npz`

**파일명 예시:**
```
gyj_handsup_1.npz
kms_sit_5.npz
stk_walk_12.npz
```

**NPZ 내부 필드:**

| 필드 | 타입 | Shape | 설명 |
|---|---|---|---|
| `grids` | float32 | (4, 800, 166) | CSI 진폭 데이터 |
| `action` | int32 | scalar | 행동 레이블 (0~3) |
| `zone` | int32 | scalar | Zone 레이블 (0~3) |
| `subject` | str | scalar | 피험자 코드 (예: 'kms') |
| `position` | int32 | scalar | 원본 측정 위치 번호 (1~16) |

**grids 차원 의미:**

```
(4, 800, 166)
  │   │    └── 유효 서브캐리어 수
  │   └─────── 시간 프레임 수 (800)
  └─────────── RX 안테나 인덱스 (0~3)
```

---

## 3. NPZ 생성 파이프라인 (`LightGBM/preprocess.py`)

```
원시 CSV (RX0~RX3 각 1개)
    ↓
복소수 CSI에서 진폭(Amplitude) 추출
    ↓
NULL 서브캐리어 26개 제거
    (인덱스: 0~5, 32, 59~65, 123~133, 191)
    → 남은 서브캐리어: 166개
    ↓
Seq-ID 기반 800-그리드 정렬
    SKIP_HEAD = 50  (초기 50 시퀀스 제거)
    패킷 손실 구간 → NaN 삽입
    ↓
3-시그마 이상치 제거 (OUTLIER_SIGMA = 3.0)
    ↓
선형 보간 (NaN 채움)
    전방 채움(ffill) / 후방 채움(bfill) 폴백
    NaN 비율 > 50%이면 경고
    ↓
NPZ 저장 (grids, action, zone, subject, position)
```

**주요 파라미터:**

| 파라미터 | 값 |
|---|---|
| GRID_SIZE | 800 |
| SKIP_HEAD | 50 |
| OUTLIER_SIGMA | 3.0 |
| NAN_THRESHOLD | 0.5 |
| 유효 서브캐리어 | 166개 |

---

## 4. 레이블 정의

### 4-1. 행동 (Action)

| 레이블 | 행동 | 설명 |
|---|---|---|
| 0 | handsup | 손 들기 |
| 1 | sit | 앉기 |
| 2 | stand | 서기 |
| 3 | walk | 걷기 |

### 4-2. Zone (구역)

| Zone 레이블 | 포함 위치 번호 | 담당 RX (코드 0-indexed) |
|---|---|---|
| 0 | 1, 2, 5, 6 | RX0 |
| 1 | 3, 4, 7, 8 | RX1 |
| 2 | 9, 10, 13, 14 | RX2 |
| 3 | 11, 12, 15, 16 | RX3 |

> **위치 번호 → Zone 매핑:**
> ```python
> ZONE_MAP = {
>     1: 0, 2: 0, 5: 0, 6: 0,
>     3: 1, 4: 1, 7: 1, 8: 1,
>     9: 2, 10: 2, 13: 2, 14: 2,
>     11: 3, 12: 3, 15: 3, 16: 3,
> }
> ```

---

## 5. 피험자 목록 및 Train / Test 분할

### 5-1. 전체 피험자

| 코드 | Train/Test |
|---|---|
| gyj | Train |
| jhj | Train |
| jkw | Train |
| kjh | Train |
| kmh | Train |
| **kms** | **Test** |
| kye | Train |
| lsi | Train |
| mhe | Train |
| phr | Train |
| stk | Train |
| swt | Train |
| ysj | Train |

### 5-2. 분할 요약

| 분할 | 피험자 수 | 샘플 수 (원시) |
|---|---|---|
| Train | 12명 | 12 × 64 = **768** |
| Test | 1명 (kms) | 1 × 64 = **64** |
| **합계** | **13명** | **832** |

### 5-3. 분할 방식 특성

- **Validation Set 없음** — Train / Test 2분할만 사용
- **피험자 기반 분할 (Subject-Independent)** — 테스트 피험자(kms)의 데이터는 학습에 전혀 포함되지 않음
- **Leave-One-Subject-Out (LOSO)** 스타일: 일반화 성능(미지 피험자 대응) 측정에 적합
- 조기 종료(Early Stopping) 기준: **테스트 정확도 최고점** (별도 Validation 사용 안 함)

> ⚠️ 별도 Validation Set이 없으므로 체크포인트 저장 기준이 테스트 데이터에 의존한다.
> 엄밀한 하이퍼파라미터 탐색이 필요할 경우 LOSO-k fold 적용을 권장한다.

---

## 6. 전처리 및 슬라이딩 윈도우

### 6-1. 전처리 순서

```
(800, 664) 원시 데이터 (grids 변환 후)
    ↓
StandardScaler 정규화
    train 데이터 전체로 fit → feature별 mean=0, std=1
    test 데이터에는 transform만 적용
    ↓
슬라이딩 윈도우 생성
    window_size = 200,  stride = 20
    → (200, 664) 윈도우
    ↓
윈도우별 평균 제거
    window = window - window.mean(axis=0)
    → 추론 시에도 동일하게 적용 (학습/추론 조건 일치)
```

### 6-2. 664D 벡터 RX 인터리브 구조

```
(4, 800, 166)
    .transpose(1, 2, 0) → (800, 166, 4)
    .reshape(800, 664)

결과 664D: [sc0_rx0, sc0_rx1, sc0_rx2, sc0_rx3, sc1_rx0, sc1_rx1, ...]
                        stride = 4 간격으로 RX 인터리브
```

| RX | 664D 내 인덱스 |
|---|---|
| RX0 | 0, 4, 8, …, 660 |
| RX1 | 1, 5, 9, …, 661 |
| RX2 | 2, 6, 10, …, 662 |
| RX3 | 3, 7, 11, …, 663 |

### 6-3. 슬라이딩 윈도우 수 추정

| 파라미터 | 값 |
|---|---|
| window_size | 200 |
| stride | 20 |
| 샘플당 윈도우 수 | (800 − 200) / 20 + 1 = **31** |

| 분할 | 원시 샘플 | 윈도우 수 |
|---|---|---|
| Train (전체) | 768 | 768 × 31 = **23,808** |
| Train (Zone N 오버샘플 후) | — | Zone N 윈도우 ×2 추가 |
| Test (전체) | 64 | 64 × 31 = **1,984** |
| Test (Zone N 필터 후) | 16 (Zone당) | 16 × 31 = **496** |

> Zone당 위치 수: 4 위치 × 4 행동 = 16 샘플 → 496 윈도우

---

## 7. Zone 전문가별 데이터 구성

### 7-1. 학습 데이터 오버샘플링

```python
train_win.oversample_zone(ZONE_ID, factor=2)
```

| 구분 | 오버샘플 전 | 오버샘플 후 |
|---|---|---|
| 담당 Zone (ZONE_ID) | N 윈도우 | N × 2 윈도우 |
| 나머지 3개 Zone | M 윈도우 (각) | M 윈도우 (변동 없음) |

- 담당 Zone 데이터를 1회 추가 복제 → 의도적 클래스 불균형
- 전문가 모델이 담당 Zone의 신호 패턴을 더 집중 학습하도록 유도

### 7-2. 테스트 데이터 필터링

```python
test_win_zone = test_win.filter_by_zone(ZONE_ID)
```

- 테스트는 **담당 Zone 데이터만** 사용
- 다른 Zone 데이터는 평가에서 제외
- 실제 파이프라인(ZoneMLP → ZoneExpertLSTM)과 동일한 조건 유지

---

## 8. 평가 지표

### 8-1. 기본 평가 방식

| 항목 | 값 |
|---|---|
| 예측 단위 | **윈도우 1개 → 즉시 예측** |
| Majority Voting | **미사용** |
| 이유 | 실시간 추론 조건 재현 (짧은 패킷만으로 즉시 판단) |

### 8-2. 사용 지표

#### (1) Accuracy (정확도)

$$\text{Accuracy} = \frac{\text{정답 윈도우 수}}{\text{전체 윈도우 수}}$$

- 학습 중 매 에폭마다 윈도우 단위 정확도 계산
- 최고 테스트 정확도 기준으로 체크포인트 저장

#### (2) Classification Report

`sklearn.metrics.classification_report` 사용

각 클래스별 출력:

| 지표 | 수식 | 의미 |
|---|---|---|
| Precision | TP / (TP + FP) | 예측한 것 중 실제 정답 비율 |
| Recall | TP / (TP + FN) | 실제 정답 중 맞게 예측한 비율 |
| F1-score | 2 × P × R / (P + R) | Precision · Recall의 조화 평균 |
| Support | — | 해당 클래스의 실제 샘플 수 |

Macro / Weighted 평균도 함께 출력.

#### (3) Confusion Matrix (혼동행렬)

- 행: 실제 레이블 (True)
- 열: 예측 레이블 (Predicted)
- `seaborn.heatmap`으로 시각화 후 PNG 저장

### 8-3. 전체 파이프라인 평가 (`inference.py`)

| 평가 항목 | 내용 |
|---|---|
| Zone Accuracy | ZoneMLP의 Zone 분류 정확도 (전체 테스트 윈도우) |
| Action Accuracy | 파이프라인 전체의 행동 분류 정확도 |
| Zone Classification Report | Zone 0~3 각 클래스별 P/R/F1 |
| Action Classification Report | handsup/sit/stand/walk 각 클래스별 P/R/F1 |
| Zone별 전문가 CM | 예측 Zone으로 라우팅된 샘플에 대한 행동 분류 결과 |

### 8-4. 출력 파일

| 파일 | 내용 |
|---|---|
| `results/zone_expert_action_{0~3}_cm.png` | Zone별 전문가 단독 평가 혼동행렬 |
| `results/pipeline_zone_cm.png` | 파이프라인 Zone 분류 혼동행렬 |
| `results/pipeline_action_cm.png` | 파이프라인 전체 행동 분류 혼동행렬 |
| `results/pipeline_zone{0~3}_expert_cm.png` | 파이프라인 Zone별 전문가 행동 분류 혼동행렬 |

---

## 9. 요약 테이블

| 항목 | 값 |
|---|---|
| 데이터 형식 | Wi-Fi CSI 진폭, NPZ |
| 총 피험자 | 13명 |
| Train 피험자 | 12명 |
| Test 피험자 | 1명 (kms) |
| Validation | 없음 |
| 분할 방식 | Subject-Independent (LOSO 스타일) |
| 행동 클래스 | 4개 |
| Zone 클래스 | 4개 |
| 총 원시 샘플 | ~832 |
| 학습 원시 샘플 | ~768 |
| 테스트 원시 샘플 | ~64 |
| 윈도우 크기 | 200 프레임 |
| 윈도우 스트라이드 | 20 프레임 |
| 학습 윈도우 (Zone 오버샘플 전) | ~23,808 |
| 테스트 윈도우 (Zone 필터 후) | ~496 (Zone당) |
| 예측 단위 | 윈도우 1개 (즉시 예측) |
| 주요 평가 지표 | Accuracy, F1-score, Confusion Matrix |
