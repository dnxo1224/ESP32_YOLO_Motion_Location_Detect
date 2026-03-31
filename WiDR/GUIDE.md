# WiDR 논문 재현 가이드

**논문:** Wi-Fi CSI-Based Human Activity Recognition and Indoor Localization with Sampling Irregularity Mitigation
**저자:** Jaekwon Lee, Kar-Ann Toh (Yonsei University)
**학술지:** IEEE Internet of Things Journal (accepted 2025)
**DOI:** 10.1109/JIOT.2025.3602515

> 이 문서 하나만 읽으면 WiDR 시스템을 완전히 재현할 수 있도록, 논문의 모든 핵심 내용(수식, 아키텍처, 데이터셋, 평가 프로토콜, 하이퍼파라미터)을 담았다.

---

## Section 1. 논문 개요

### WiDR (Wi-Fi Dual-task Recognition) — 핵심 기여 3가지

| 컴포넌트 | 설명 | 논문 섹션 |
|---|---|---|
| **Neural CDEs** | Wi-Fi CSI의 불규칙 샘플링 문제 해결 | Section III-A |
| **Dual-Stream Cross-Attention** | Temporal/Channel 스트림 상관관계 추출 | Section III-B |
| **Parameter-Shared Dual-Task** | 두 태스크의 상호 보완으로 과적합 방지 | Section III-C |

### 문제 정의

- Wi-Fi CSI는 **패킷 손실로 불규칙하게 샘플링**됨 (xt ∈ R^d, timestamps t0 < t1 < ... < tL이 균일하지 않음)
- 기존 방법들은 단일 태스크(AR 또는 IL)에 특화되어 태스크 간 상관관계를 무시
- 해결책:
  1. Neural CDEs → 불규칙 샘플링을 연속 경로로 변환
  2. Dual-Stream Cross-Attention → Temporal × Channel 상관관계 추출
  3. Parameter Sharing Regularization → 두 태스크 파라미터 공유로 상호 보완

---

## Section 2. 시스템 아키텍처 (Fig. 1)

```
Raw Wi-Fi CSI (불규칙 샘플링)
  xt ∈ R^d, at timestamps t0 < t1 < ... < tL
       │
       ▼ (a) Neural CDEs — 불규칙 샘플링 처리 (Section III-A)
  Interpolation: X: [t0, tL] → R^(d+1)  (natural cubic spline)
  z_t0 = ζθ(t0, x0)         ← 초기 은닉 상태
  dz/dt = fθ(z_t) dX/dt     ← CDE 동역학
  x̂_t = ℓθ(z_t)             ← 출력 복원
  CDE output: X̂ ∈ R^(L×d)
       │
       ▼ (b) Dual-Stream Cross-Attention (Section III-B)
  Temporal Stream: X̂ ∈ R^(L×d)    → Q^CA
  Channel Stream:  X̂^T ∈ R^(d×L) → K^CA, V^CA
  Cross-Attention → MHA (self) → FFN
       │
       ┌──────────┴──────────┐
       ▼                     ▼
  (c) Branch 1 (AR)    Branch 2 (IL)     (Section III-C)
  파라미터 Θ_(1)         파라미터 Θ_(2)
  동일 구조, 별도 파라미터
  L_reg = λ ||Θ_(1) - Θ_(2)||²
       │                     │
       ▼                     ▼
  Activity Recognition  Indoor Localization
  (Ŷ_1)                 (Ŷ_2)
```

---

## Section 3. Component 1: Neural CDEs (논문 Section III-A)

### 배경

Wi-Fi CSI 시계열 (t0,x0), ..., (tL,xL), xi ∈ R^d 는 패킷 손실로 불규칙하게 샘플링된다.
Neural CDEs(Controlled Differential Equations)는 이 **불규칙 타임스탬프를 연속 경로로 변환**하여 처리한다.

### 핵심 수식

**[Eq.1] — CDE 은닉 상태 진화:**
```
z_t = z_t0 + ∫[t0→t] fθ(z_s) dX(s)   for t ∈ (t0, tL]
```
- `fθ`: R^w → R^(w×(d+1)) — vector field (은닉 상태 동역학)
- `X`: [t0, tL] → R^(d+1) — natural cubic spline으로 보간된 연속 경로 (d feature + 1 time channel)
- `w` — 은닉 노드 수 (**최적값: 400**, Fig.8(a))

**[Eq.2] — ODE로 변환 (수치 계산 효율화):**
```
z_t = z_t0 + ∫[t0→t] fθ(z_s) · dX(s)/ds · ds
```
→ 일반 ODE 솔버(e.g. `torchdiffeq`)로 효율적으로 풀 수 있음

**[Eq.3] — 출력 복원:**
```
x̂_t = ℓθ(z_t)   for t ∈ (t0, tL]
```
- `ℓθ`: R^w → R^d — 출력 매핑 함수
- 결과: X̂ = [x̂_0, x̂_1, ..., x̂_L] ∈ R^(L×d)

### torchcde 구현 방법

```python
import torchcde

# 1. Natural cubic spline 계수 계산
#    x: (B, L_raw, d+1) — d feature + 1 time channel
#    t: (L_raw,) — 불규칙 타임스탬프 (초 단위)
coeffs = torchcde.natural_cubic_spline_coeffs(x, t)
X = torchcde.CubicSpline(coeffs)

# 2. 초기 은닉 상태 (ζθ: R^(d+1) → R^w)
z0 = zeta_theta(X.evaluate(X.interval[0]))  # (B, w)

# 3. CDE 풀기 (균일 그리드 t_eval에서 출력)
t_eval = torch.linspace(t[0], t[-1], steps=L_fixed)
z_T = torchcde.cdeint(X=X, func=f_theta, z0=z0, t=t_eval)  # (B, L_fixed, w)

# 4. 출력 복원 (ℓθ: R^w → R^d)
x_hat = ell_theta(z_T)  # (B, L_fixed, d)
```

### CDEFunc (fθ) 구조

```python
class CDEFunc(nn.Module):
    """논문 fθ: R^w → R^(w×(d+1))"""
    def __init__(self, hidden_dim, input_dim):
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, 128)
        self.linear2 = nn.Linear(128, hidden_dim * (input_dim + 1))
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

    def forward(self, t, z):
        # z: (B, w)
        z = self.linear1(z).tanh()
        z = self.linear2(z)  # (B, w * (d+1))
        z = z.view(*z.shape[:-1], self.hidden_dim, self.input_dim + 1)
        return z  # (B, w, d+1)
```

### 현재 Raw 데이터에 Neural CDE 적용 방법

Raw CSV의 `seq_id`를 **불규칙 타임스탬프**로 활용한다:
```
t_i = seq_id_i / 30.0   (명목 샘플링 레이트: 30 Hz)
seq_id 갭 = 누락된 패킷 = 불규칙 샘플링 → Neural CDE가 자연스럽게 처리
```

전처리 없이 Raw I/Q → 진폭 변환 + null subcarrier 제거만 수행 후 CDE 적용 (상세는 Section 7 참조).

---

## Section 4. Component 2: Dual-Stream Cross-Attention (논문 Section III-B)

### 개념

**입력:** CDE 출력 X̂ ∈ R^(L×d)

두 스트림을 서로 교차하여 상관관계를 포착:
- **Temporal stream**: X̂ ∈ R^(L×d) — 시간 관점 → **Query**
- **Channel stream**: X̂^T ∈ R^(d×L) — 채널(주파수) 관점 → **Key, Value**

### 핵심 수식

**[Eq.4] — Cross-Attention 기본 공식:**
```
f_a(Q, K, V) = softmax(QK^T / √d_k) V
Q = X̂ W_q,    W_q ∈ R^(d×d_k)
K = X̂^T W_k,  W_k ∈ R^(L×d_k)
V = X̂^T W_v,  W_v ∈ R^(L×d_k)
```

**[Eq.12] — CA 레이어 상세:**
```
Q^CA = X̂ W_q^CA,         W_q^CA ∈ R^(d×d_k)
K^CA = X̂^T W_k^CA,       W_k^CA ∈ R^(L×d_k)
V^CA = X̂^T W_v^CA,       W_v^CA ∈ R^(L×d_k)
Z^CA = softmax(Q^CA (K^CA)^T / √d_k) V^CA W_o^CA
```
Θ^CA = {W_q^CA, W_k^CA, W_v^CA, W_o^CA}

**[Eq.13] — MHA 레이어 (각 head h ∈ {1,...,H}):**
```
Q_h = Z^CA W_q^MHA,   W_q^MHA ∈ R^(d_k×d_k)
K_h = Z^CA W_k^MHA,   W_k^MHA ∈ R^(d_k×d_k)
V_h = Z^CA W_v^MHA,   W_v^MHA ∈ R^(d_k×d_k)
Z_h = softmax(Q_h K_h^T / √d_k) V_h
```

**[Eq.14] — MHA 출력 합산:**
```
Z^MHA = (||_{h=1}^H Z_h) W_o^MHA,   W_o^MHA ∈ R^((H×d_k)×d)
```
Θ^MHA = {W_q^MHA, W_k^MHA, W_v^MHA, W_o^MHA}

**[Eq.15] — FFN (GELU 활성화):**
```
Z^FFN = GELU(Z^MHA W_1^FFN + b_1^FFN) W_2^FFN + b_2^FFN
W_1^FFN ∈ R^(d×d_FFN),  b_1^FFN ∈ R^(d_FFN)
W_2^FFN ∈ R^(d_FFN×d),  b_2^FFN ∈ R^d
```
Θ^FFN = {W_1^FFN, b_1^FFN, W_2^FFN, b_2^FFN}

### 텐서 Shape 흐름 (현재 데이터셋: d=114, 4 RX → d=456, L=L_fixed)

```
X̂: (B, L_fixed, 456)
  ↓ temporal_proj: Linear(456, d_k)
Q^CA: (B, L_fixed, d_k)

X̂^T: (B, 456, L_fixed)
  ↓ channel_proj: Linear(L_fixed, d_k)
K^CA, V^CA: (B, 456, d_k)

CrossAttention(Q^CA, K^CA, V^CA): (B, L_fixed, d_k)
  ↓ Residual + LayerNorm
Z^CA: (B, L_fixed, d_k)

SelfAttention(Z^CA, Z^CA, Z^CA): (B, L_fixed, d_k)
  ↓ Residual + LayerNorm
Z^MHA: (B, L_fixed, d_k)

FFN(Z^MHA): (B, L_fixed, d_k)
  ↓ Residual + LayerNorm
Z^FFN: (B, L_fixed, d_k)

GlobalAveragePooling(dim=1): (B, d_k)  ← 최종 특징 벡터
```

**주의:** `d_k`는 코드에서 `d_model`로 표기, 논문 최적값 = **128**. `channel_proj`의 입력 차원은 CDE 출력 길이 `L_fixed`에 따라 결정된다.

---

## Section 5. Component 3: Parameter-Shared Dual-Task (논문 Section III-C)

### 핵심 아이디어

두 태스크(Activity Recognition, Indoor Localization)가 동일 구조의 CA+MHA+FFN을 각자의 파라미터로 학습하되, L2 거리 정규화로 파라미터 공유를 강제한다.

**이론적 근거 (Eq.5, Rademacher 복잡도):**
```
|R(h) - R̂(h)| ≤ 2Rn(H) ≤ 2||Θ_(1) - Θ_(2)||_2 / √n
```
→ 두 브랜치 파라미터 차이를 줄이면 일반화 오차가 감소함

### 핵심 수식

**[Eq.7] — 파라미터 집합 정의:**
```
Θ_(1) = Θ_(1)^CA ∪ Θ_(1)^MHA ∪ Θ_(1)^FFN   (Task 1: AR)
Θ_(2) = Θ_(2)^CA ∪ Θ_(2)^MHA ∪ Θ_(2)^FFN   (Task 2: IL)
```

**[Eq.6] — Parameter Sharing Regularization:**
```
L_reg = λ ||Θ_(1) - Θ_(2)||^2
```
- λ > 0: 정규화 강도 (**최적값: 0.7**, Fig.8(c))

**[Eq.16] — 최종 예측 (FC 분류기):**
```
Ŷ_1 = softmax(Z^FFN_(1) W^FC_(1) + b^FC_(1))   ← AR 예측
Ŷ_2 = softmax(Z^FFN_(2) W^FC_(2) + b^FC_(2))   ← IL 예측
```
**FC hidden dimension = 200** (논문 고정값, Section IV-B-2)

**[Eq.8] — 전체 손실 함수:**
```
L_Total = Σ_{i=1}^{2} [ -1/N Σ_{n=1}^N Σ_{c=1}^{C_i} y_{n,c}^(i) log(ŷ_{n,c}^(i)) ] + λ L_reg
        = CE_1(Ŷ_1, Y_1) + CE_2(Ŷ_2, Y_2) + λ ||Θ_(1) - Θ_(2)||^2
```

**[Eq.9-11] — 두 브랜치 연산:**
```
Z^CA_(1)   = CrossAttention(X̂, X̂^T, Θ^CA_(1))     # AR 브랜치
Z^CA_(2)   = CrossAttention(X̂, X̂^T, Θ^CA_(2))     # IL 브랜치

Z^MHA_(1)  = MHA(Z^CA_(1), Θ^MHA_(1))
Z^MHA_(2)  = MHA(Z^CA_(2), Θ^MHA_(2))

Z^FFN_(1)  = FFN(Z^MHA_(1), Θ^FFN_(1))
Z^FFN_(2)  = FFN(Z^MHA_(2), Θ^FFN_(2))
```

### Python 구현 핵심 (param_reg_loss)

```python
def param_reg_loss(self):
    """L_reg = ||Θ1 - Θ2||^2 (논문 Eq.6)"""
    loss = torch.tensor(0.0, device=next(self.parameters()).device)
    for p1, p2 in zip(self.encoder_action.parameters(),
                      self.encoder_zone.parameters()):
        loss = loss + torch.sum((p1 - p2) ** 2)
    for p1, p2 in zip(self.classifier_action.parameters(),
                      self.classifier_zone.parameters()):
        loss = loss + torch.sum((p1 - p2) ** 2)
    return loss
```

---

## Section 6. 데이터 증강: Mixup (논문 Eq.17)

**[Eq.17] — Mixup 공식:**
```
X̃ = ηX + (1-η)X',   Ỹ = ηY + (1-η)Y',   η ~ Beta(α, α)
```
- (X, Y), (X', Y'): 독립적으로 샘플링한 두 훈련 샘플
- η ∈ [0, 1]: 보간 비율
- α > 0: Beta 분포 파라미터 (**최적값: 0.4**, Fig.10)
- Dual-task이므로 Y는 (y_action, y_zone) 쌍

**Mixup 손실:**
```
L_mixup = η · CE(pred, y_a) + (1-η) · CE(pred, y_b)
```

### 논문 데이터 증강 비교 결과 (Table V, ARIL 데이터셋)

```
방법              AR(%)   IL(%)   Mean(%)
────────────────────────────────────────
w/o DA           90.28   95.68   92.98
w/ Jitter        86.33   97.48   91.90
w/ Scaling       85.25   93.53   89.39
w/ Permutation   85.61   97.12   91.37
w/ Time warp     83.45   97.48   90.47
w/ Fmix          89.20   94.24   91.72
w/ Cutmix        88.84   93.52   91.18
w/ Mixup (pre-)  94.60   99.28   96.94  ← 최고 (본 논문 채택)
w/ Mixup (post-) 88.84   98.56   93.70
────────────────────────────────────────
```
- pre-Mixup: CDE 처리 전에 Mixup 적용 (논문 채택)
- post-Mixup: CDE 처리 후에 Mixup 적용

---

## Section 7. 데이터셋

### 7.1 ARIL 데이터셋 (논문 참고문헌 9)

- **태스크:** Activity Recognition (AR) + Indoor Localization (IL)
- **활동 6종:** Up, Down, Left, Right, Circle, Cross
- **위치:** 16곳 (4×4 그리드)
- **수집:** 각 활동 × 각 위치 × 15회 반복 = 1,440 샘플
  - 이상치 수동 제거 후: **1,116 train + 278 test** (고정 분할)
- **장비:** Universal Software Radio Peripherals (USRPs)
- **데이터 형태:** 각 샘플 = 52 subcarriers × 192 packets → 텐서 shape: **(52, 192)**

### 7.2 CSIDA 데이터셋 (논문 참고문헌 34)

- **태스크:** Activity Recognition (AR) + Identity Recognition (IR)
- **활동 6종:** Pull Left, Pull Right, Lift Up, Press Down, Draw Circle, Draw Zigzag
- **피험자:** 5명, **위치:** 5곳, **반복:** 10회
- **총 샘플:** 1,500개 → 누락 데이터 제거 후 **1,421개**
  - 분할: **80% train, 20% test**
- **장비:** Atheros CSI 도구, 안테나 3쌍, 114 subcarriers
- **데이터 형태:** 1,800 packets — 텐서 shape: **(3, 114, 1800)**

### 7.3 현재 사용 데이터셋 (ESP32 수집 — Raw Data)

#### 데이터셋 비교

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
항목               ARIL            현재 데이터셋
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
피험자             N/A             13명
활동               6종             4종
위치               16곳            16곳 (4×4 그리드)
샘플 수            1,394           832
Raw 텐서 shape     (52, 192)       (L_raw, 192) — L_raw 불규칙
Subcarrier         52              192 raw → 114 (null 제거 후)
태스크             AR + IL         AR + Zone (4 zones)
Neural CDE 입력    실험 데이터      raw seq_id로 불규칙 타임스탬프 구성
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

#### Raw 데이터 파일 구조

```
13_raw_data/ (또는 data/ 디렉토리)
└── {subject}_{action}_{position}_rx{1-4}.csv  ← _872 없음, 정렬 없음
    예: gyj_handsup_1_rx1.csv
    - 행 수: 샘플마다 다름 (패킷 손실로 불규칙)
    - header=None (헤더 없음)
    - 열 구조:
        col[2]    = seq_id (패킷 시퀀스 번호, 정수)
        col[3:-1] = CSI I/Q 데이터 (복소수 실수/허수 쌍)
        col[-1]   = timestamp (초 단위)
    - CSI 형식: R0, I0, R1, I1, ..., R191, I191 — 192 subcarrier × 2 = 384 값
```

#### Raw → 모델 입력 변환 파이프라인 (전체)

```
[Raw CSV 로드]
  ↓
1. seq_id 추출 → 불규칙 타임스탬프
   t_i = seq_id_i / 30.0   (명목 샘플링 레이트: 30 Hz)

2. I/Q → 진폭 변환
   real = csi_raw[:, 0::2]   # (L_raw, 192) 실수부
   imag = csi_raw[:, 1::2]   # (L_raw, 192) 허수부
   amplitude = sqrt(real² + imag²)  # (L_raw, 192)

3. Null Subcarrier 제거 (HT-LTF 기준, 192→114)
   NULL_IDX = {0,1,2,3,4,5, 32, 59,60,61,62,63,64,65,
               123,124,125,126,127,128,129,130,131,132,133, 191}
   = 총 78개 제거 → 114개 유효 subcarrier
   amplitude = amplitude[:, valid_idx]  # (L_raw, 114)

4. time channel 추가 → Neural CDE 입력
   x_with_time: (L_raw, 115) = [amplitude | t.reshape(-1,1)]

5. Natural Cubic Spline (torchcde)
   coeffs = torchcde.natural_cubic_spline_coeffs(x_with_time, t)
   X = torchcde.CubicSpline(coeffs)

6. Neural CDE 풀기 → 균일 그리드 출력
   t_eval = torch.linspace(t[0], t[-1], steps=L_fixed)
   z_T = torchcde.cdeint(X=X, func=f_theta, z0=z0, t=t_eval)
   x_hat = ell_theta(z_T)  # (B, L_fixed, 114)

7. 4 RX 안테나 concat
   X̂_full: (B, L_fixed, 456) = concat(rx1, rx2, rx3, rx4)
```

#### 레이블 정의

```python
ACTION_MAP = {'handsup': 0, 'sit': 1, 'stand': 2, 'walk': 3}

ZONE_MAP = {
    1: 0, 2: 0, 5: 0, 6: 0,      # Zone 0 (상좌)
    3: 1, 4: 1, 7: 1, 8: 1,      # Zone 1 (상우)
    9: 2, 10: 2, 13: 2, 14: 2,   # Zone 2 (하좌)
    11: 3, 12: 3, 15: 3, 16: 3,  # Zone 3 (하우)
}

ALL_SUBJECTS = [
    'gyj', 'jhj', 'jkw', 'kjh', 'kmh', 'kms',
    'kye', 'lsi', 'mhe', 'phr', 'stk', 'swt', 'ysj'
]  # 총 13명
```

#### Raw 데이터 로딩 예시 코드

```python
import pandas as pd
import numpy as np

NULL_IDX = (list(range(0, 6)) + [32] +
            list(range(59, 66)) + list(range(123, 134)) + [191])
VALID_IDX = [i for i in range(192) if i not in NULL_IDX]  # 114개

def load_raw_rx(raw_dir, subject, action, position, rx):
    """
    Raw CSV 1개 RX → (L_raw, 114) 진폭 + (L_raw,) 타임스탬프

    Returns:
        amplitude: np.array, shape (L_raw, 114)
        timestamps: np.array, shape (L_raw,), 초 단위
    """
    path = os.path.join(raw_dir, f"{subject}_{action}_{position}_rx{rx}.csv")
    df = pd.read_csv(path, header=None)

    # 불규칙 타임스탬프 (seq_id → 초)
    seq_ids = df.iloc[:, 2].values.astype(int)
    timestamps = seq_ids / 30.0

    # CSI I/Q 파싱 (col[3] 부터 col[-2] 까지, 384개 값)
    csi_raw = df.iloc[:, 3:-1].values.astype(np.float32)  # (L_raw, 384)
    real = csi_raw[:, 0::2]   # (L_raw, 192)
    imag = csi_raw[:, 1::2]   # (L_raw, 192)
    amplitude = np.sqrt(real**2 + imag**2)  # (L_raw, 192)

    # Null subcarrier 제거
    amplitude = amplitude[:, VALID_IDX]  # (L_raw, 114)

    return amplitude, timestamps


def load_raw_sample(raw_dir, subject, action, position):
    """
    4 RX → 각 (L_raw, 114) + 공통 타임스탬프

    Note: 4개 RX의 행 수가 다를 수 있으므로
    공통 타임스탬프 집합의 교집합을 기준으로 정렬하거나,
    각 RX를 독립적으로 CDE에 입력하는 방식 선택 가능.
    """
    rxs = []
    timestamps_list = []
    for rx in range(1, 5):
        amp, ts = load_raw_rx(raw_dir, subject, action, position, rx)
        rxs.append(amp)
        timestamps_list.append(ts)
    return rxs, timestamps_list  # 각각 list of (L_i, 114), (L_i,)
```

---

## Section 8. 하이퍼파라미터 (논문 최적값)

### 모델 하이퍼파라미터

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
파라미터               최적값    탐색 범위                       논문 위치
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
w (CDE hidden nodes)   400       {10,25,50,100,200,400,800,1000}  Fig.8(a)
d_k / d_model          128       —                                (파생)
d_FFN (FFN hidden)     256       {128,256,512,1024,2048}          Fig.8(a)
h (MHA heads)          4         {1,4,8,16,32}                    Fig.8(b)
λ (param sharing)      0.7       {0.001~5}                        Fig.8(c)
α (Mixup Beta)         0.4       {0.1,0.2,...,0.9}                Fig.10
FC hidden dim          200       고정값 (경험적)                   Section IV-B-2
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 학습 설정

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
설정 항목          값                  비고
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Optimizer          Adam
Learning Rate      1e-3
Weight Decay       1e-4
Batch Size         32
Epochs             ~20 수렴            Fig.15: SOTA 60-100에폭 대비 빠름
LR Scheduler       CosineAnnealingLR
Gradient Clip      1.0
Normalization      StandardScaler (train 기준 fit, test transform)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## Section 9. 평가 프로토콜 (논문 Section IV-B-1)

### Experiment I — 하이퍼파라미터 탐색

- 데이터 분할: train set의 80% 학습, 20% validation
- **5-fold CV**: 5회 반복 평균 = 검증 정확도
- Test set은 이 단계에서 사용 안 함

### Experiment II — 주요 컴포넌트 Ablation Study

ARIL 데이터셋 사용, 비교 대상:

```
방법    설명
────────────────────────────────────────────────────────────────
STS     Single Temporal Stream: temporal stream만 사용
SCS     Single Channel Stream: channel stream만 사용
TTC     Two-Tower Concatenation: 두 스트림 독립 처리 후 concat
Proposed  Cross-Attention (본 논문)
────────────────────────────────────────────────────────────────
```

### Experiment III — SOTA 비교

최종 test set으로 평가. 비교 방법: LSTM, GRU, BiLSTM, CNN, CNN-LSTM, THAT, JARIL, ResNet, InceptionTime, ARIL

#### 논문 최종 결과 (Table VII, VIII) — ARIL 데이터셋 (AR + IL)

```
방법            AR w/o DA  IL w/o DA  AR w/ Mixup  IL w/ Mixup
──────────────────────────────────────────────────────────────
LSTM            60.09      94.17      65.47        94.96
JARIL           88.85      96.76      91.73        99.64
Proposed        90.64      97.12      93.88        98.20
──────────────────────────────────────────────────────────────
Mean (Proposed)                       96.04%  ← Table VIII
```

### 현재 데이터셋 평가 방법 (논문에 없는 추가 평가)

**Mode A — Subject-Split (논문 방식에 가장 가까운 generalization 평가):**
```
Train: 11명 (ALL_SUBJECTS에서 kjh, kms 제외)
Test:  kjh, kms 2명
→ 결과: Action Accuracy (%), Zone Accuracy (%), Mean (%)
```

**Mode B — Subject-wise 5-Fold Cross-Validation:**
```
Fold 1: test = [gyj, jhj, jkw]
Fold 2: test = [kjh, kmh, kms]
Fold 3: test = [kye, lsi, mhe]
Fold 4: test = [phr, stk]
Fold 5: test = [swt, ysj]
→ 5개 fold 평균 ± 표준편차 보고
```

---

## Section 10. 파일 구조 및 실행

### 권장 파일 구조

```
ESP32_YOLO_Motion_Location_Detect/
├── WiDR/
│   ├── GUIDE.md          ← 이 파일
│   ├── dataset.py        ← Raw CSI 로드, Neural CDE 전처리, PyTorch Dataset
│   ├── model.py          ← CDEFunc + WiCDE + DualStreamCA + WiDRNet
│   ├── train.py          ← 학습 스크립트 (Mode A + Mode B 5-fold CV)
│   ├── results/          ← confusion matrix PNG 저장
│   └── weights/          ← 모델 체크포인트 .pt 저장
├── 13_raw_data/          ← Raw CSI CSV 파일들
│   └── {subject}_{action}_{position}_rx{1-4}.csv
└── data_interpolated_linear_872/
    └── {subject}_{action}_{position}_rx{1-4}_872.csv  (선형보간 완료본)
```

### 실행 명령

```bash
cd /Users/seolwootae/ESP-CSI/ESP32_YOLO_Motion_Location_Detect
/Users/seolwootae/ESP-CSI/.venv/bin/python WiDR/train.py
```

### 필수 패키지

```bash
# 핵심 (Neural CDE)
pip install torchcde  # Patrick Kidger, v0.2.5 이상 — 이미 설치됨

# 이미 설치된 패키지
# torch, sklearn, pandas, numpy, scipy, matplotlib, seaborn, tqdm
```

### dataset.py 핵심 구조 (재현 시 참고)

```python
class RawCSIDataset(Dataset):
    """
    Raw CSI 로드 + Neural CDE 전처리 → PyTorch Dataset

    __getitem__ 반환: (x_hat, y_action, y_zone)
      x_hat: (L_fixed, 456) — CDE 출력, 4 RX concat
      y_action: (,) — long tensor
      y_zone: (,) — long tensor
    """
    def __init__(self, subjects, raw_dir, l_fixed=300, hidden_dim=400):
        # 1. Raw CSV 스캔
        # 2. load_raw_rx() × 4 → amplitude + timestamps
        # 3. Natural cubic spline coeffs
        # 4. CDEFunc + WiCDE 적용 (배치 처리 권장)
        # 5. 4 RX concat → (L_fixed, 456)
        ...
```

### model.py 핵심 구조 (재현 시 참고)

```python
class CDEFunc(nn.Module):
    """fθ: R^w → R^(w×(d+1)), Neural CDE vector field"""
    ...

class WiCDE(nn.Module):
    """Neural CDE: (B, L_raw, d+1) → (B, L_fixed, d)"""
    ...

class DualStreamCrossAttention(nn.Module):
    """
    단일 브랜치: CA → MHA → FFN → GAP
    입력: (B, L_fixed, 456)
    출력: (B, d_model)
    """
    ...

class WiDRNet(nn.Module):
    """
    전체 WiDR 모델
    - encoder_action: DualStreamCrossAttention (Θ1)
    - encoder_zone:   DualStreamCrossAttention (Θ2)
    - classifier_action: FC(d_model → 200 → num_actions)  ← FC hidden=200
    - classifier_zone:   FC(d_model → 200 → num_zones)
    - param_reg_loss(): L_reg = ||Θ1 - Θ2||²
    """
    ...
```

---

## Section 11. 논문 수식 ↔ 코드 완전 대응표 (Quick Reference)

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
논문 Eq.  내용                        코드 파일   함수/클래스
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Eq.1      CDE 은닉 상태 진화            model.py   WiCDE.forward() → torchcde.cdeint
Eq.2      CDE → ODE 변환               model.py   CDEFunc.forward() (fθ)
Eq.3      CDE 출력 복원                 model.py   WiCDE.output_map (ℓθ)
Eq.4      Cross-Attention 기본식        model.py   DualStreamCrossAttention.cross_attention
Eq.5      Rademacher 복잡도 경계        —          이론적 근거 (코드 구현 없음)
Eq.6      Parameter Sharing Reg        model.py   WiDRNet.param_reg_loss()
Eq.7      파라미터 집합 정의             model.py   encoder_action vs encoder_zone
Eq.8      전체 손실 함수                train.py   train_one_split() 내 loss 계산
Eq.9-11   두 브랜치 CA/MHA/FFN         model.py   WiDRNet.forward()
Eq.12     CA 레이어 상세               model.py   temporal_proj, channel_proj
Eq.13     MHA per-head 연산           model.py   DualStreamCrossAttention.self_attention
Eq.14     MHA 출력 합산                model.py   nn.MultiheadAttention (PyTorch 내장)
Eq.15     FFN (GELU)                  model.py   DualStreamCrossAttention.ffn
Eq.16     최종 분류 (FC)               model.py   WiDRNet.classifier_action / classifier_zone
Eq.17     Mixup 증강                   train.py   mixup_data(), mixup_criterion()
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 빠른 재현 체크리스트

다른 Claude가 이 문서만 보고 재현 시 확인할 사항:

- [ ] `torchcde >= 0.2.5` 설치 확인
- [ ] Raw CSV 형식 확인: `header=None`, `col[2]=seq_id`, I/Q 쌍
- [ ] Null subcarrier 제거: 192 → 114 (NULL_IDX 78개 제거)
- [ ] 타임스탬프: `t_i = seq_id_i / 30.0`
- [ ] CDE 은닉 노드: `w = 400`
- [ ] FC hidden dim: `200` (not 128)
- [ ] λ = 0.7, α = 0.4, h = 4, d_FFN = 256
- [ ] Loss = CE_action + CE_zone + 0.7 × param_reg_loss()
- [ ] Mixup: `η ~ Beta(0.4, 0.4)`, pre-CDE 적용
- [ ] Mode A: 11 train / kjh+kms test
- [ ] Mode B: 5-fold, folds = [[gyj,jhj,jkw],[kjh,kmh,kms],[kye,lsi,mhe],[phr,stk],[swt,ysj]]
