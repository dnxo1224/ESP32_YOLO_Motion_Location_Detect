# 실시간 CSI 위치·행동 추론 파이프라인 정리

ESP32 WiFi CSI를 실시간으로 수신해 **위치(Zone 4개)** 와 **행동(handsup / sit / stand / walk)** 을
분류하는 파이프라인. 기존 오프라인 파이프라인(`Zone_Expert/inference.py`, cnn_norx_noos)을
스트리밍 구조로 옮긴 것으로, 동일 입력에 대해 오프라인과 100% 같은 예측을 내도록 검증됨.

---

## 1. 전체 구조 (End-to-End)

```
[ESP32 TX (csi_send)]  ──WiFi 패킷(초당 ~30개, payload에 공유 seq 카운터)──▶

[ESP32 RX ×4 (csi_recv)]
  · TX MAC 필터링 후 CSI 추출 (384값 = 192 서브캐리어 × 실수/허수)
  · USB 시리얼 2,000,000 baud로 "CSI_DATA,..." 한 줄씩 출력
        │
        ▼  ① SerialSource (sources.py) — RX당 스레드 1개
  라인 파싱: seq_id + 진폭 166개 추출 (csi_parse.py)
  필터: RSSI(0 초과/-100 미만 폐기), CSI 길이 384 확인
        │
        ▼  ② FrameSynchronizer (synchronizer.py)
  4개 RX 스트림을 seq_id 기준으로 정렬해 "프레임" 생성
  프레임 = (664,) 벡터 = 166 서브캐리어 × 4 RX
  미수신 RX/유실 패킷 = NaN
        │
        ▼  ③ FrameRingBuffer (ring_buffer.py)
  최근 200프레임(≈6.7초) 유지, 새 프레임 10개마다 ↓ 실행
        │
        ▼  ④ 윈도우 전처리 (window_prep.py)
  3σ 이상치 제거 → 선형 보간(NaN 채움) → StandardScaler 정규화
  (스케일러는 학습 데이터로 미리 fit해 artifacts/scaler_train.npz에 저장)
        │
        ▼  ⑤ 2단계 추론 (engine.py + models.py)
  [1단계] ZoneMLP(mean_only): 윈도우 시간축 평균 → Zone 0~3 예측
  [2단계] 윈도우 평균 제거 → ZoneExpertCNN[예측된 zone] → 행동 예측
        │
        ▼  ⑥ 후처리·출력 (postprocess.py)
  다수결 스무딩(최근 5개 예측) → 콘솔 출력
  (ResultSink 인터페이스 — 대시보드 등으로 교체 가능)
```

### 오프라인 코드와의 대응 관계

| 실시간 모듈 | 대응하는 기존 코드 | 관계 |
|---|---|---|
| `csi_parse.py` | `LightGBM/preprocess.py`의 `load_raw_rx` | 배치(pandas) → 라인 단위 재구현, 결과 완전 일치 검증됨 |
| `synchronizer.py` | `preprocess.py`의 `compute_start_seq`/`align_to_grid` | 800-그리드 정렬의 스트리밍 등가물 (skip 50 seq, NaN 그리드 동일) |
| `window_prep.py` | `preprocess.py`의 `remove_outliers`/`interpolate_grid` + `dataset.py`의 `normalize_datasets` | 3σ 통계만 800프레임→200프레임 윈도우로 근사 (검증 결과 예측 차이 0) |
| `models.py`/`engine.py` | `Zone_Expert/inference.py`의 ZoneMLP + `model_cnn.py`의 ZoneExpertCNN | 동일 가중치 로드, 추론 순서 동일 |
| `export_scaler.py` | `dataset.py`의 `normalize_datasets` | 매번 재fit하던 스케일러를 1회 fit 후 파일로 영속화 |

사용 가중치:
- ZoneMLP: `13_Data_Processing/weights/zone_mlp_mean_only_best.pt`
- 행동 전문가: `Zone_Expert/weights_cnn_norx_noos/zone_expert_action_{0..3}_best.pt` (zone별 4개)

---

## 2. 시간 구조 — "몇 초 데이터로, 언제, 얼마나 자주 판단하는가"

패킷 레이트 30Hz 기준 (실측 25~30Hz):

| 항목 | 값 | 의미 |
|---|---|---|
| 윈도우 크기 | 200프레임 ≈ **6.7초** | 한 번의 추론이 보는 데이터 분량 |
| 추론 주기 (stride) | 10프레임 ≈ **0.33초** | 새 패킷 10개가 쌓일 때마다 재추론 |
| 첫 예측까지 | **약 8.4초** | 워밍업(seq 50개 스킵 ≈1.7s) + 버퍼 채움(200프레임 ≈6.7s) |
| 추론 연산 시간 | 1~3ms (CPU) | 0.33초 예산 대비 여유 100배 이상 |

### "몇 초 전의 데이터가 추론되는가?"

한 번의 예측은 **직전 6.7초 구간 전체**를 입력으로 본다. 즉:

- 윈도우의 **가장 새 데이터**: 약 0.2~0.5초 전 (동기화 지연 ~0.17초 + stride 대기 최대 0.33초)
- 윈도우의 **가장 오래된 데이터**: 약 7초 전
- 행동이 **바뀌는 순간**을 기준으로 보면: 새 행동이 윈도우의 과반을 차지해야 예측이 넘어가므로,
  행동 전환이 출력에 반영되기까지 체감 **약 3~5초**가 걸린다.
  (실측 로그에서도 sit → handsup → stand 전환 시 예측이 여러 윈도우에 걸쳐 서서히 넘어가는 것이 관찰됨)

### "최종 판단은 프레임마다? 다수결?"

두 층으로 나뉜다:

1. **원시 예측 (raw)** — 프레임마다가 아니라 **윈도우마다** (0.33초 간격).
   매 stride마다 최근 200프레임 윈도우 하나를 모델에 넣어 나온 **단일 추론 결과**다.
   윈도우 내부 프레임들의 다수결이 아니다.
2. **스무딩 예측 (smoothed)** — 원시 예측을 다시 **최근 5개(≈1.7초 치)의 다수결**로 안정화한 것.
   순간적으로 튀는 오분류 1~2개를 걸러준다. **최종 판단으로는 이 값을 쓰면 된다.**
   (`--smooth-k`로 개수 조절: 크게 = 안정적·둔감, 작게 = 민감·깜빡임)

---

## 3. 콘솔 출력 한 줄 해석

```
[    8.0s] win@0     Zone 0 (67%) | sit     (94%) | smoothed: Zone 0 / sit | NaN 15.5% | 71ms
   │        │         │      │       │       │        │                       │        │
   │        │         │      │       │       │        │                       │        └ 이 윈도우의 추론 소요 시간
   │        │         │      │       │       │        └ 최근 5개 예측 다수결 (최종 판단용)
   │        │         │      │       │       └ 행동 confidence (softmax 확률)
   │        │         │      │       └ 이 윈도우 1회 추론의 행동 예측 (raw)
   │        │         │      └ zone confidence
   │        │         └ 이 윈도우 1회 추론의 zone 예측 (raw)
   │        └ 윈도우 시작 프레임 번호 (0, 10, 20, ... — stride 10 간격)
   └ 파이프라인 시작 후 경과 시간
```

- **NaN 15.5%**: 전처리(보간) 전 윈도우에서 값이 비어 있던 비율.
  = 패킷 유실 + 미수신 RX 채널. 유실분은 선형 보간으로 채워진 뒤 추론된다.
  1-RX 테스트에서 8~15%는 정상 범위. 4-RX 정상 운용에서 지속적으로 높으면
  (>20%) 수신 환경(거리·간섭·baud)을 점검할 것.
- **71ms → 이후 1~3ms**: 첫 추론만 PyTorch 내부 초기화 때문에 느리고, 이후는 1~3ms.
- `[stats]` 라인: `frames=누적 프레임, preds=누적 예측, gaps=한 RX도 못 받은 슬롯 수,
  incomplete=일부 RX만 받은 프레임 수, late_dropped=이미 방출된 슬롯에 늦게 온 패킷(워밍업
  스킵 50 seq × 채널 수 ≈ 200은 정상), RX0:145/146줄(err:1)=파싱 성공/수신 라인/파싱 실패`

---

## 4. 단계별 상세

### ① 시리얼 수신 + 파싱

- RX 보드당 데몬 스레드 1개가 `readline()` → `parse_line()`.
- 한 라인 = 409필드: `CSI_DATA, recv_mac, seq_id, mac, rssi, ..., len(384), first_word, "[CSI 384값]"`.
- 진폭 계산: 필드 26~407의 real/imag 인터리브 → `sqrt(re²+im²)` → 널 서브캐리어 26개 제거 → **166개**.
- 수집 스크립트(multi_rx_collect_csi_v4.py)와 동일 조건: baud 2,000,000, RSSI 필터, len=384 필터.

### ② seq_id 동기화

TX가 패킷 payload에 심은 **공유 카운터(seq_id)** 를 4개 RX가 모두 수신하므로 이것으로 정렬한다.

- 워밍업: 각 RX가 고유 seq 50개를 볼 때까지 대기 → `base_seq = max(각 RX의 50번째 seq)`
  (오프라인 전처리의 `compute_start_seq`와 동일 — 부팅 직후 불안정 구간 제거)
- 슬롯 방출 조건 (먼저 충족되는 것):
  ① 4개 RX 모두 도착 ② 5슬롯(≈0.17초) 더 새 패킷이 도착 ③ 0.25초 타임아웃
- 아무 RX도 못 받은 슬롯은 전-NaN 프레임으로 채워 시간축을 균일하게 유지
  (오프라인의 NaN 그리드와 같은 의미 → 이후 보간으로 복원)

### ③~④ 링버퍼 + 윈도우 전처리

- 링버퍼가 최근 200프레임을 유지, 프레임 10개마다 스냅샷 (200, 664) 생성.
- 전처리 순서 (오프라인과 동일한 순서):
  1. **3σ 이상치 제거**: 윈도우 내 컬럼별 평균±3σ 밖 값을 NaN으로
  2. **선형 보간**: 시간축 보간 + 앞/뒤 채움 + 잔여 0 채움
  3. **정규화**: `(x - mean) / scale` — 학습 피험자 12명 전체 프레임(614,400개)으로
     fit한 StandardScaler (`artifacts/scaler_train.npz`)
- 유일한 의도적 근사: 3σ 통계를 오프라인은 샘플 전체(800프레임), 실시간은 윈도우(200프레임)
  기준으로 계산. 검증 결과 예측 차이 없음.

### ⑤ 2단계 추론 (MoE 구조)

```
정규화된 윈도우 (1, 200, 664)
   ├─▶ ZoneMLP: 시간축 평균 (664,) → MLP(512→256→128) → Zone 0~3
   │      · "6.7초 동안의 평균적인 채널 상태"로 위치를 판단
   │      · 평균 제거 전 데이터 사용 (위치 정보는 신호의 절대 수준에 있음)
   └─▶ 윈도우 평균 제거 (x - 시간축 평균)
          → ZoneExpertCNN[예측된 zone]: 1D CNN → 행동 4클래스
          · "평균 대비 시간적 변동 패턴"으로 행동을 판단
          · zone마다 별도 학습된 전문가 모델 4개 중 해당 zone 것만 사용
```

### ⑥ 후처리

- `MajorityVoteSmoother(k=5)`: zone과 action 각각 최근 5개 예측의 다수결.
- `ConsoleSink`: 콘솔 출력. `ResultSink` 인터페이스를 구현하면 웹 대시보드,
  파일 로깅 등으로 출력을 교체/추가 가능.

---

## 5. 검증 결과 (하드웨어 없이 재현 가능)

| 검증 | 방법 | 결과 |
|---|---|---|
| 파서 정확성 | 라인 파서 vs 오프라인 `load_raw_rx` (866패킷) | 완전 일치 (diff 0) |
| A. 엔진 정확성 | 실시간 엔진 vs 오프라인 배치 추론 (kms 3,904윈도우) | zone/action **100% 일치** |
| B. 엔드투엔드 | raw CSV 4개를 리플레이로 전체 파이프라인 통과 vs 오프라인 (244윈도우) | zone/action **100% 일치** |
| 성능 | 윈도우당 추론 시간 | 1~3ms (예산 333ms) |
| 참고 정확도 | 테스트 피험자(kms) 기준 | zone 97.6% / action 94.7% |

재현 명령:
```bash
python Realtime/verify_engine.py    # 검증 A
python Realtime/verify_replay.py    # 검증 B
```

---

## 6. 실행 모드

```bash
# 정상 운용 (RX 4대 — 포트 순서 = rx1~rx4, 수집 당시 매핑과 동일해야 함)
python Realtime/main.py --mode live --ports COM7 COM10 COM11 COM9

# 1-RX 동작 테스트 (RX 1대 패킷을 4채널에 복제 — 배관 검증 전용)
python Realtime/main.py --mode live --ports COM3 --mirror

# 녹화 데이터 리플레이 (하드웨어 불필요)
python Realtime/main.py --mode replay --subject kms --action walk --position 5
```

### ⚠️ 1-RX(--mirror) 모드의 한계

mirror 모드의 zone/action 예측값은 **의미가 없다.** 모델은 "4개 RX에 신호가 서로 다르게
잡히는 공간적 패턴"으로 위치를 학습했는데, mirror는 4채널에 같은 데이터를 복제하므로
공간 정보가 사라진다. 이 모드는 수신→파싱→동기화→추론의 **배관이 실기기에서 도는지**
확인하는 용도다. 의미 있는 추론은 RX 4대를 학습 당시와 같은 배치로 설치해야 한다.

---

## 7. 파일 구성

```
Realtime/
  main.py            # 엔트리: 모드 선택, 파이프라인 루프
  paths.py           # 경로 상수, Zone_Expert/LightGBM 모듈 import 설정
  csi_parse.py       # CSI_DATA 라인 → (seq_id, 진폭 166) 파서
  sources.py         # SerialSource(라이브) / ReplaySource(리플레이) — 동일 인터페이스
  synchronizer.py    # 4-RX seq_id 동기화 → 프레임(664,) 방출
  ring_buffer.py     # 200프레임 링버퍼
  window_prep.py     # 3σ 제거 + 보간 + 정규화
  models.py          # ZoneMLP + ZoneExpertCNN 로더
  engine.py          # 2단계 추론 엔진
  postprocess.py     # 다수결 스무딩 + 출력 Sink
  export_scaler.py   # [1회] 학습 스케일러 fit → artifacts/scaler_train.npz
  verify_engine.py   # 검증 A
  verify_replay.py   # 검증 B
  config_example.json# 라이브 포트/baud 설정 예시
  artifacts/scaler_train.npz  # 영속화된 StandardScaler (mean/scale)
```

펌웨어 (github.com/dnxo1224/esp-idf):
- TX: `examples/get-started/csi_send` — 30Hz로 seq 카운터를 실은 패킷 송신
- RX: `examples/get-started/csi_recv` — TX MAC 필터 후 CSI를 시리얼로 출력
- (ESP-IDF v6.0 빌드 시 `WIFI_BW_HT40→WIFI_BW40`, `WIFI_BW_HT20→WIFI_BW20` 치환 필요 — 반영 완료)
