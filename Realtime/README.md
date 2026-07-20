# Realtime — 실시간 CSI 위치·행동 분류 파이프라인

Zone_Expert 오프라인 파이프라인(cnn_norx_noos)을 실시간 스트리밍으로 옮긴 구현.

```
ESP32 RX1~4 (USB 시리얼) ──► 파싱 (csi_parse) ──► seq_id 동기화 (synchronizer)
        ──► 링버퍼 200프레임 (ring_buffer) ──► [새 프레임 10개마다]
        ──► 3σ 제거·보간·정규화 (window_prep) ──► ZoneMLP → zone
        ──► 윈도우 평균 제거 ──► ZoneExpertCNN[zone] → action ──► 콘솔
```

첫 예측: 워밍업(seq 50개 스킵) + 200프레임(~6.7s@30Hz) 후. 이후 ~0.33s 간격.

## 사용 순서

```bash
# 0. 의존성 (라이브 모드만 pyserial 필요)
.venv/Scripts/pip install pyserial

# 1. [1회] 스케일러 영속화 (train 12명 NPZ에서 fit → artifacts/scaler_train.npz)
python Realtime/export_scaler.py

# 2. 검증 A: 엔진 == 오프라인 배치 (100% 일치 기대)
python Realtime/verify_engine.py

# 3. 검증 B: raw CSV 리플레이 엔드투엔드 vs 오프라인 (zone ≥90% 기대)
python Realtime/verify_replay.py

# 4. 리플레이 실행 (하드웨어 불필요, 실제 속도 재생)
python Realtime/main.py --mode replay --subject kms --action walk --position 5

# 5. 라이브 실행 (ports는 rx1~rx4 순서 — 수집 때의 Rx1~Rx4 포트 매핑과 동일하게)
python Realtime/main.py --mode live --ports COM7 COM10 COM11 COM9
# 또는: python Realtime/main.py --mode live --config Realtime/config_example.json
```

## 라이브 모드 (ESP32 연동)

펌웨어는 esp-idf fork(github.com/dnxo1224/esp-idf)의
`examples/get-started/csi_recv`(RX) / `csi_send`(TX)를 그대로 사용한다.
수집 스크립트 `examples/get-started/tools/multi_rx_collect_csi_v4.py`가 하던
시리얼 수신을 SerialSource가 대체하며, 동일 조건을 적용한다:

- baud **2,000,000** / 8N1, 수신 버퍼 64KB, 시작 시 입력 버퍼 리셋
- RSSI 필터 (0 초과 / -100 미만 폐기), CSI len=384 라인만 통과
- 펌웨어가 TX MAC 필터링을 이미 수행하므로 PC측 MAC 필터 불필요
- seq_id는 TX 패킷 payload의 공유 카운터 → 4 RX 정렬에 그대로 사용
- **--ports 순서 = rx1~rx4** (학습 데이터의 RX 순서와 일치해야 함)
- 클래식 ESP32 타깃 전용 (C5/C6 펌웨어는 출력 필드 수가 달라 거부됨)

## 1-RX 동작 테스트 (TX 1대 + RX 1대)

배관(수신→파싱→동기화→윈도우→추론) 동작 확인 전용. `--mirror`가 RX 1대의
패킷을 4채널에 복제해 NaN 없이 전체 경로를 돌린다.
**예측 zone/action은 의미 없음** — 모델은 4-RX 공간 패턴으로 학습됐다.

```bash
# 하드웨어 없는 사전 검증 (rx1 CSV 1개만 사용)
python Realtime/main.py --mode replay --subject kms --action walk --position 5 --mirror

# 실기기: RX 1대만 연결
python Realtime/main.py --mode live --ports COM7 --mirror
```

`--mirror` 없이 포트 1~3개만 주면 미수신 채널을 NaN→0으로 채워 돌긴 하지만
경고가 출력된다 (동작 테스트는 mirror 권장).

## 4Tx–1Rx 신규 파이프라인 (개발 중)

설계: 5GHz CSI 측정(C5) / 2.4GHz 데이터 전송(S3) 대역 분리, 송신 ESP가 서버로 TCP 직결.
프레임 사양은 [FRAME_SPEC.md](FRAME_SPEC.md) 참조 (436B 바이너리, CRC16).

| 파일 | 역할 |
|---|---|
| `csi_frame.py` | 프레임 v1 인코더/디코더 + 스트림 파서 (`python Realtime/csi_frame.py`로 자가 검증) |
| `tcp_source.py` | TCP 프레임 서버 — `python Realtime/tcp_source.py --port 5010` |
| `fake_tx_client.py` | 서버 검증용 가짜 클라이언트 (쓰레기/CRC 오염 주입) |

펌웨어 (esp-idf fork `examples/get-started/`):
- `csi_tx_relay/` — 송신 ESP (S3): UART→TCP 바이트 파이프. `TEST_FRAME_GEN=1`이면
  합성 프레임 자체 생성(2단계), 0이면 UART 릴레이(3단계). WiFi/서버 설정은 app_main.c 상단.
- `csi_fake_rx/` — 가짜 Rx (S3): 436B 프레임을 UART1(GPIO17, 2M baud)로 방출.

## 주의사항

- `[stats]` 라인의 RX별 카운트로 실효 수신율 확인
  (STALL! 표시 = 2초 이상 무수신).
- **의도적 근사**: 오프라인은 3σ 이상치 통계를 800프레임 전체에서 계산하지만
  실시간은 200프레임 윈도우 기준 (verify_replay.py가 영향 정량화).
- 스케일러는 sklearn pickle이 아닌 npz(mean/scale)로 저장 — 버전 독립적.
- 대시보드 확장: `postprocess.ResultSink` 구현체를 추가하면 됨.
