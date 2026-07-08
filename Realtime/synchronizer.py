"""
4-RX seq_id 동기화기.

오프라인 preprocess.py의 compute_start_seq / align_to_grid의 스트리밍 등가물:

  1. 워밍업: 각 RX가 서로 다른 seq_id를 skip_head(50)개 이상 볼 때까지 대기.
     base_seq = max(각 RX의 50번째(0-index) 고유 seq)  ← compute_start_seq와 동일.
     워밍업 중 수신한 패킷은 버퍼링 후 base 확정 시 재주입한다.
  2. seq_id → 슬롯: base_seq 기준 상대 인덱스. 65536 랩어라운드는
     직전 seq와의 부호 있는 차분으로 언랩 처리.
  3. 슬롯별 (166, 4) 조립. 방출 조건(먼저 충족되는 것):
       - 4개 RX 모두 도착
       - max_seen_slot - slot >= lookahead_slots (기본 5, ≈165ms@30Hz)
       - 최초 관측 후 slot_timeout_s(기본 0.25s) 경과  ← poll()에서 검사
  4. 방출은 항상 슬롯 순서대로. 중간에 아무 RX도 못 본 슬롯은 전-NaN
     프레임으로 채워 방출한다(오프라인 NaN 그리드와 동일 의미).
     이미 방출된 슬롯에 늦게 도착한 패킷은 폐기(카운트만 유지).

프레임 평탄화: (166, 4).reshape(664) — RX 인덱스가 빠르게 변하는 인터리브.
dataset.py의 grids.transpose(1, 2, 0).reshape(800, 664)와 동일 순서.
"""
import time
from dataclasses import dataclass

import numpy as np

NUM_RX = 4
NUM_SUBCARRIERS = 166
FEATURE_DIM = NUM_SUBCARRIERS * NUM_RX
MAX_SEQ = 65536


@dataclass
class Frame:
    slot: int
    data: np.ndarray        # (664,) float32, 결측 = NaN
    n_rx: int               # 이 슬롯에 실제 도착한 RX 수 (0 = 갭 슬롯)


class FrameSynchronizer:
    def __init__(self,
                 skip_head: int = 50,
                 lookahead_slots: int = 5,
                 slot_timeout_s: float = 0.25,
                 max_pending: int = 512):
        self.skip_head = skip_head
        self.lookahead_slots = lookahead_slots
        self.slot_timeout_s = slot_timeout_s
        self.max_pending = max_pending

        # 워밍업 상태
        self._warm = True
        self._warm_seqs = [set() for _ in range(NUM_RX)]
        self._warm_packets = []          # (rx, seq_id, amp, t_recv)
        self.base_seq = None

        # 언랩 상태
        self._last_sid = None
        self._last_abs = None

        # 슬롯 조립
        self._pending = {}               # slot -> [grid(166,4), first_seen_t]
        self._next_emit = 0              # 다음에 방출해야 할 슬롯
        self._max_seen = -1

        # 통계
        self.n_emitted = 0
        self.n_gap_frames = 0
        self.n_late_dropped = 0
        self.n_incomplete = 0            # 4개 미만 RX로 방출된 프레임 수

    # ── 내부 ──────────────────────────────────────────────────────────────────

    def _unwrap(self, sid: int) -> int:
        """seq_id → base_seq 기준 절대 슬롯 (65536 랩어라운드 언랩)."""
        if self._last_sid is None:
            d = (sid - self.base_seq) % MAX_SEQ
            if d > MAX_SEQ // 2:
                d -= MAX_SEQ
            self._last_sid = sid
            self._last_abs = d
            return d
        d = (sid - self._last_sid) % MAX_SEQ
        if d > MAX_SEQ // 2:
            d -= MAX_SEQ
        self._last_sid = sid
        self._last_abs += d
        return self._last_abs

    def _emit_ready(self, now: float, force_all: bool = False):
        """방출 조건을 만족한 슬롯들을 순서대로 방출."""
        frames = []
        while self._pending or force_all:
            if not self._pending:
                break
            s = min(self._pending)
            grid, first_t = self._pending[s]
            n_rx = NUM_RX - int(np.isnan(grid[0]).sum())
            ready = (
                force_all
                or n_rx == NUM_RX
                or (self._max_seen - s) >= self.lookahead_slots
                or (now - first_t) > self.slot_timeout_s
                or len(self._pending) > self.max_pending
            )
            if not ready:
                break

            # 갭 슬롯 채움
            while self._next_emit < s:
                frames.append(Frame(
                    slot=self._next_emit,
                    data=np.full(FEATURE_DIM, np.nan, dtype=np.float32),
                    n_rx=0,
                ))
                self.n_gap_frames += 1
                self._next_emit += 1

            del self._pending[s]
            if n_rx < NUM_RX:
                self.n_incomplete += 1
            frames.append(Frame(slot=s, data=grid.reshape(FEATURE_DIM), n_rx=n_rx))
            self._next_emit = s + 1

        self.n_emitted += len(frames)
        return frames

    def _ingest(self, rx: int, sid: int, amp: np.ndarray, t_recv: float):
        slot = self._unwrap(sid)
        if slot < self._next_emit:
            self.n_late_dropped += 1
            return
        if slot > self._max_seen:
            self._max_seen = slot
        if slot not in self._pending:
            self._pending[slot] = [
                np.full((NUM_SUBCARRIERS, NUM_RX), np.nan, dtype=np.float32),
                t_recv,
            ]
        self._pending[slot][0][:, rx] = amp

    # ── 공개 API ──────────────────────────────────────────────────────────────

    def push(self, rx: int, seq_id: int, amp: np.ndarray,
             t_recv: float = None):
        """패킷 1개 주입 → 방출 가능해진 Frame 리스트 반환."""
        if t_recv is None:
            t_recv = time.monotonic()

        if self._warm:
            self._warm_seqs[rx].add(seq_id)
            self._warm_packets.append((rx, seq_id, amp, t_recv))
            if all(len(s) > self.skip_head for s in self._warm_seqs):
                # compute_start_seq: 각 RX의 skip_head번째 고유 seq 중 최댓값
                self.base_seq = max(
                    sorted(s)[self.skip_head] for s in self._warm_seqs
                )
                self._warm = False
                buffered, self._warm_packets = self._warm_packets, []
                for p_rx, p_sid, p_amp, p_t in buffered:
                    self._ingest(p_rx, p_sid, p_amp, p_t)
                return self._emit_ready(t_recv)
            return []

        self._ingest(rx, seq_id, amp, t_recv)
        return self._emit_ready(t_recv)

    def poll(self):
        """패킷이 없어도 타임아웃된 슬롯을 방출 (라이브 모드 유휴 시 호출)."""
        if self._warm:
            return []
        return self._emit_ready(time.monotonic())

    def flush(self):
        """스트림 종료 시 잔여 슬롯 전부 방출 (리플레이 모드)."""
        if self._warm:
            return []
        return self._emit_ready(time.monotonic(), force_all=True)

    def stats(self) -> str:
        return (f'emitted={self.n_emitted} gaps={self.n_gap_frames} '
                f'incomplete={self.n_incomplete} late_dropped={self.n_late_dropped} '
                f'pending={len(self._pending)}')
