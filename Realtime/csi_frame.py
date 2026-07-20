"""
CSI 바이너리 프레임 v1 인코더/디코더 — FRAME_SPEC.md의 단일 구현.

프레임(436B, LE):
  [0:2]   magic A5 5A
  [2]     version 0x01
  [3]     예약
  [4:50]  메타 46B: tx_id u8, flags u8, seq u32, rx_timestamp_us u64,
                    rssi i8, noise i8, channel u8, csi_len u16, 예약 27B
  [50:434] csi i8×384
  [434:436] crc16-ccitt(FALSE) — [4:434] 대상, LE
"""
import struct
from dataclasses import dataclass

MAGIC = b'\xa5\x5a'
VERSION = 0x01
HEADER_LEN = 4
META_LEN = 46
CSI_LEN = 384
CRC_LEN = 2
FRAME_LEN = HEADER_LEN + META_LEN + CSI_LEN + CRC_LEN   # 436

_META_FMT = '<BBIQbbBH27x'
assert struct.calcsize(_META_FMT) == META_LEN

# ── CRC16-CCITT(FALSE): poly 0x1021, init 0xFFFF ─────────────────────────────

_CRC_TABLE = []
for _b in range(256):
    _c = _b << 8
    for _ in range(8):
        _c = ((_c << 1) ^ 0x1021) if (_c & 0x8000) else (_c << 1)
    _CRC_TABLE.append(_c & 0xFFFF)


def crc16(data: bytes, init: int = 0xFFFF) -> int:
    crc = init
    for byte in data:
        crc = ((crc << 8) & 0xFFFF) ^ _CRC_TABLE[((crc >> 8) ^ byte) & 0xFF]
    return crc


# ── 프레임 구조 ───────────────────────────────────────────────────────────────

@dataclass
class Frame:
    tx_id: int
    flags: int
    seq: int
    rx_timestamp_us: int
    rssi: int
    noise_floor: int
    channel: int
    csi_len: int
    csi: bytes              # 384B raw i8 I/Q


def build_frame(tx_id: int, seq: int, rx_timestamp_us: int, csi: bytes,
                rssi: int = -50, noise_floor: int = -96, channel: int = 36,
                csi_len: int = CSI_LEN, flags: int = 0) -> bytes:
    assert len(csi) == CSI_LEN, f'csi must be {CSI_LEN}B, got {len(csi)}'
    header = MAGIC + bytes([VERSION, 0x00])
    meta = struct.pack(_META_FMT, tx_id, flags, seq, rx_timestamp_us,
                       rssi, noise_floor, channel, csi_len)
    body = meta + csi
    crc = crc16(body)
    return header + body + struct.pack('<H', crc)


def _decode_body(body: bytes) -> Frame:
    tx_id, flags, seq, ts, rssi, noise, ch, csi_len = \
        struct.unpack(_META_FMT, body[:META_LEN])
    return Frame(tx_id=tx_id, flags=flags, seq=seq, rx_timestamp_us=ts,
                 rssi=rssi, noise_floor=noise, channel=ch, csi_len=csi_len,
                 csi=body[META_LEN:])


# ── 스트림 파서 (경계 탐색 + CRC 검증 + 재동기화) ─────────────────────────────

class FrameParser:
    """
    TCP/UART 바이트 스트림 → Frame 리스트.
    feed()에 아무 크기의 청크를 밀어 넣으면 완성된 프레임들을 돌려준다.
    magic 스캔 → version 확인 → CRC 검증. 실패 시 1바이트 전진(재동기화).
    """

    def __init__(self):
        self._buf = bytearray()
        self.n_ok = 0
        self.n_crc_fail = 0
        self.n_resync_bytes = 0     # 프레임 밖에서 버린 바이트 수

    def feed(self, chunk: bytes):
        self._buf.extend(chunk)
        frames = []
        buf = self._buf
        pos = 0
        while True:
            idx = buf.find(MAGIC, pos)
            if idx < 0:
                # magic 없음 — 마지막 1바이트(다음 청크와 이어질 수 있음)만 남김
                self.n_resync_bytes += max(0, len(buf) - pos - 1)
                keep = buf[-1:] if len(buf) - pos > 0 else b''
                del buf[:]
                buf.extend(keep)
                break
            self.n_resync_bytes += idx - pos
            if len(buf) - idx < FRAME_LEN:
                # 프레임 미완성 — magic부터 남기고 대기
                del buf[:idx]
                break
            if buf[idx + 2] != VERSION:
                pos = idx + 1
                continue
            body = bytes(buf[idx + HEADER_LEN: idx + HEADER_LEN + META_LEN + CSI_LEN])
            crc_stored, = struct.unpack_from('<H', buf, idx + FRAME_LEN - CRC_LEN)
            if crc16(body) != crc_stored:
                self.n_crc_fail += 1
                pos = idx + 1
                continue
            frames.append(_decode_body(body))
            self.n_ok += 1
            pos = idx + FRAME_LEN
        return frames

    def stats(self) -> str:
        return (f'ok={self.n_ok} crc_fail={self.n_crc_fail} '
                f'resync_bytes={self.n_resync_bytes}')


# ── 자가 검증 ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import random
    random.seed(0)

    # 1) 인코딩→디코딩 왕복
    csi = bytes((i * 7 + 3) % 256 for i in range(CSI_LEN))
    raw = build_frame(tx_id=2, seq=12345, rx_timestamp_us=987654321,
                      csi=csi, rssi=-43, channel=36)
    assert len(raw) == FRAME_LEN
    p = FrameParser()
    out = p.feed(raw)
    assert len(out) == 1 and out[0].seq == 12345 and out[0].tx_id == 2
    assert out[0].csi == csi and out[0].rssi == -43
    print('[1] round-trip OK')

    # 2) 임의 크기 청크로 쪼개서 주입 (스트림 재조립)
    stream = b''.join(build_frame(0, s, s * 33333, csi) for s in range(100))
    p = FrameParser()
    got = []
    i = 0
    while i < len(stream):
        n = random.randint(1, 700)
        got += p.feed(stream[i:i + n])
        i += n
    assert [f.seq for f in got] == list(range(100)), 'chunked reassembly failed'
    print('[2] chunked reassembly OK (100 frames)')

    # 3) 쓰레기 바이트 + CRC 오염 주입
    p = FrameParser()
    good, bad = 0, 0
    payload = bytearray()
    for s in range(200):
        if s % 10 == 5:
            payload += bytes(random.randrange(256) for _ in range(random.randint(1, 50)))
        f = bytearray(build_frame(1, s, s * 33333, csi))
        if s % 7 == 3:
            f[100] ^= 0xFF     # CSI 한 바이트 오염 → CRC 실패해야 함
            bad += 1
        else:
            good += 1
        payload += f
    got = p.feed(bytes(payload))
    assert len(got) == good, f'expected {good} ok frames, got {len(got)}'
    assert p.n_crc_fail >= bad, f'crc_fail={p.n_crc_fail} < corrupted={bad}'
    print(f'[3] garbage+corruption OK ({p.stats()})')

    print('csi_frame self-check PASS')
