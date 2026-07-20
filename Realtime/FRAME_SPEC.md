# CSI 바이너리 프레임 사양 v1

4Tx–1Rx 파이프라인의 공용 계약. 이 문서를 참조하는 곳:
- C5 Rx 펌웨어 (프레임 생성)
- csi_fake_rx / 테스트 생성기 (프레임 생성)
- 송신 ESP(csi_tx_relay)는 프레임을 **해석하지 않음** (바이트 파이프)
- 서버 `Realtime/csi_frame.py` (인코딩/디코딩 단일 구현) + `tcp_source.py` (경계 탐색)

## 프레임 레이아웃 — 총 436 바이트, 리틀엔디언

| 오프셋 | 크기 | 필드 | 타입 | 설명 |
|---|---|---|---|---|
| 0 | 2 | magic | `A5 5A` | 프레임 경계 탐색용 고정 헤더 |
| 2 | 1 | version | u8 | 현재 `0x01` |
| 3 | 1 | (예약) | u8 | `0x00` |
| 4 | 1 | tx_id | u8 | 송신 Tx 번호 (0~3) |
| 5 | 1 | flags | u8 | 비트 플래그 (예약, 현재 0) |
| 6 | 4 | seq | u32 | Tx가 심은 공유 seq 카운터 |
| 10 | 8 | rx_timestamp_us | u64 | Rx 하드웨어 클럭 (µs, esp_timer 기준) |
| 18 | 1 | rssi | i8 | dBm |
| 19 | 1 | noise_floor | i8 | dBm |
| 20 | 1 | channel | u8 | 수신 채널 |
| 21 | 2 | csi_len | u16 | CSI 유효 바이트 수 (정상 = 384) |
| 23 | 27 | (예약) | — | 0 채움. 향후 확장용 |
| 50 | 384 | csi | i8×384 | raw I/Q (192 서브캐리어 × I,Q) |
| 434 | 2 | crc | u16 | CRC16-CCITT, 오프셋 4~433 (메타+CSI, 헤더 제외) |

- 헤더(4B) + 메타(46B) + CSI(384B) + CRC(2B) = **436B**
- CRC16-CCITT(FALSE): poly `0x1021`, init `0xFFFF`, XOR out 없음, 리틀엔디언으로 기록
- 파서 규칙: magic 스캔 → version 확인 → 436B 확보 → CRC 검증.
  실패 시 1바이트 전진 후 재탐색 (스트림 재동기화)

## 전송 규약

| 항목 | 값 |
|---|---|
| Rx → 송신ESP | UART, **2,000,000 baud**, 8N1. 프레임을 그대로 연속 송출 |
| 송신ESP → 서버 | raw TCP, 서버 포트 **5010**, TCP_NODELAY. 바이트 스트림 그대로 릴레이 |
| 유실 정책 | 재전송 없음. 깨진/유실 프레임은 서버가 폐기 → SyncEngine이 결측 슬롯 처리 |

## UART 배선 (Rx ↔ 송신 S3)

| 신호 | 송신 S3 핀 | 상대(Rx/fake_rx) 핀 |
|---|---|---|
| 데이터 | GPIO18 (UART1 RX) | GPIO17 (UART1 TX) |
| GND | GND | GND (공통 필수) |

버전 변경 시 `version` 바이트를 올리고 이 문서에 이력을 남길 것.
