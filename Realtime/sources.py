"""
패킷 소스: 라이브 시리얼(SerialSource) + 녹화 CSV 리플레이(ReplaySource).

두 소스 모두 동일한 queue.Queue[Packet]에 패킷을 넣으므로
다운스트림(동기화기 이후)은 모드와 무관하게 동일하게 동작한다.
스트림이 끝나는 소스(리플레이)는 마지막에 None 센티널을 넣는다.
"""
import glob
import heapq
import os
import queue
import threading
import time
from dataclasses import dataclass

import numpy as np

from csi_parse import parse_line, iter_csv_packets


@dataclass
class Packet:
    rx: int                 # 0..3
    seq_id: int
    amp: np.ndarray         # (166,) float32
    t_recv: float           # time.monotonic()


class PacketSource:
    def start(self, out_queue: 'queue.Queue') -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError


# ── 라이브: USB 시리얼 4포트 ──────────────────────────────────────────────────

class SerialSource(PacketSource):
    """
    RX당 데몬 스레드 1개가 시리얼 라인을 읽어 파싱 후 큐에 넣는다.

    설정은 학습 데이터 수집 스크립트(esp-idf fork의
    examples/get-started/tools/multi_rx_collect_csi_v4.py)와 동일:
      - baud 2,000,000 / 8N1 / 수신 버퍼 64KB
      - 시작 시 reset_input_buffer()로 부팅 로그 등 잔류 데이터 폐기
      - RSSI·len 필터는 csi_parse.parse_line에서 동일하게 적용
    """

    def __init__(self, ports: list, baud: int = 2_000_000,
                 mirror: bool = False):
        assert 1 <= len(ports) <= 4, 'COM 포트는 1~4개 (정상 운용은 4개)'
        if mirror and len(ports) != 1:
            raise ValueError('mirror 모드는 포트 1개일 때만 사용')
        if len(ports) < 4 and not mirror:
            print(f'[SerialSource] 경고: 포트 {len(ports)}개 — 미수신 RX 채널은 '
                  f'NaN→0 처리됨 (1포트 동작 테스트는 --mirror 권장)')
        self.ports = ports
        self.n_rx = len(ports)
        self.baud = baud
        self.mirror = mirror
        self._stop = threading.Event()
        self._threads = []
        # 상태 (메인 루프에서 주기 출력)
        self.pkt_counts = [0] * self.n_rx
        self.parse_errors = [0] * self.n_rx
        self.line_counts = [0] * self.n_rx    # CSI 여부 무관 수신 라인 수
        self.last_recv = [0.0] * self.n_rx

    def start(self, out_queue: 'queue.Queue') -> None:
        import serial  # pyserial — 라이브 모드에서만 필요

        for rx, port in enumerate(self.ports):
            t = threading.Thread(
                target=self._read_loop, args=(rx, port, out_queue, serial),
                daemon=True, name=f'serial-rx{rx}',
            )
            t.start()
            self._threads.append(t)

    def _read_loop(self, rx, port, out_queue, serial_mod):
        try:
            ser = serial_mod.Serial(
                port, self.baud, bytesize=8, parity='N', stopbits=1, timeout=1,
            )
            try:
                ser.set_buffer_size(rx_size=65536, tx_size=65536)
            except Exception:
                pass                     # Windows 전용 API — 실패해도 무방
            ser.reset_input_buffer()     # 잔류 부팅 로그/옛 데이터 폐기
        except Exception as e:
            print(f'[SerialSource] RX{rx} {port} open 실패: {e}')
            return
        print(f'[SerialSource] RX{rx} <- {port} @ {self.baud} baud')

        while not self._stop.is_set():
            try:
                raw = ser.readline()
            except Exception as e:
                print(f'[SerialSource] RX{rx} read 오류: {e}')
                time.sleep(0.5)
                continue
            if not raw:
                continue
            self.line_counts[rx] += 1
            line = raw.decode('utf-8', errors='replace')
            parsed = parse_line(line)
            if parsed is None:
                if line.startswith('CSI_DATA'):
                    self.parse_errors[rx] += 1
                continue
            sid, amp = parsed
            now = time.monotonic()
            self.pkt_counts[rx] += 1
            self.last_recv[rx] = now
            # mirror: RX 1대의 패킷을 4채널 전부에 복제 (동작 테스트용)
            targets = range(4) if self.mirror else (rx,)
            for r in targets:
                pkt = Packet(r, sid, amp, now)
                try:
                    out_queue.put_nowait(pkt)
                except queue.Full:
                    # 백프레셔: 가장 오래된 패킷 폐기
                    try:
                        out_queue.get_nowait()
                    except queue.Empty:
                        pass
                    out_queue.put_nowait(pkt)
        ser.close()

    def stop(self) -> None:
        self._stop.set()

    def health(self) -> str:
        now = time.monotonic()
        parts = []
        for rx in range(self.n_rx):
            silent = now - self.last_recv[rx] if self.last_recv[rx] else -1
            flag = ' STALL!' if (silent > 2 or silent < 0) else ''
            err = (f'(err:{self.parse_errors[rx]})'
                   if self.parse_errors[rx] else '')
            parts.append(f'RX{rx}:{self.pkt_counts[rx]}/'
                         f'{self.line_counts[rx]}줄{err}{flag}')
        return ' '.join(parts)


# ── 리플레이: 녹화 CSV 4개 ────────────────────────────────────────────────────

class ReplaySource(PacketSource):
    """
    기존 raw CSV 4개(rx1~rx4)를 재생한다.

    speed > 0 : RX별 스레드가 CSV 마지막 필드의 timestamp 간격/speed 로 페이싱
                (실제 수신 타이밍 재현)
    speed == 0: 단일 스레드가 4개 파일을 seq_id 순으로 병합해 최고 속도로 주입
                (결정론적 — 검증용)
    mirror    : CSV 1개(rx1)만 받아 각 패킷을 4채널에 복제 (1-RX 동작 테스트의
                하드웨어 없는 사전 검증용)
    """

    def __init__(self, csv_paths: list, speed: float = 1.0,
                 mirror: bool = False):
        if mirror:
            assert len(csv_paths) == 1, 'mirror 모드는 CSV 1개만'
        else:
            assert len(csv_paths) == 4, 'rx1~rx4 CSV 4개가 필요합니다'
        self.csv_paths = csv_paths
        self.mirror = mirror
        self.speed = speed
        self._stop = threading.Event()
        self._threads = []
        self._n_done = 0
        self._done_lock = threading.Lock()
        self.finished = threading.Event()

    @classmethod
    def from_sample(cls, data_dir: str, subject: str, action: str,
                    position: int, speed: float = 1.0,
                    mirror: bool = False) -> 'ReplaySource':
        rx_range = (1,) if mirror else (1, 2, 3, 4)
        paths_ = [
            os.path.join(data_dir, subject,
                         f'{subject}_{action}_{position}_rx{rx}.csv')
            for rx in rx_range
        ]
        missing = [p for p in paths_ if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(missing)
        return cls(paths_, speed=speed, mirror=mirror)

    def start(self, out_queue: 'queue.Queue') -> None:
        if self.speed == 0:
            t = threading.Thread(target=self._merged_loop, args=(out_queue,),
                                 daemon=True, name='replay-merged')
            t.start()
            self._threads.append(t)
        else:
            for rx, path in enumerate(self.csv_paths):
                t = threading.Thread(target=self._paced_loop,
                                     args=(rx, path, out_queue),
                                     daemon=True, name=f'replay-rx{rx}')
                t.start()
                self._threads.append(t)

    def _merged_loop(self, out_queue):
        if self.mirror:
            # 단일 파일 → 순차 읽기, 패킷마다 4채널 복제
            for sid, amp, _ts in iter_csv_packets(self.csv_paths[0]):
                if self._stop.is_set():
                    break
                now = time.monotonic()
                for r in range(4):
                    out_queue.put(Packet(r, sid, amp, now))
            out_queue.put(None)
            self.finished.set()
            return

        streams = []
        for rx, path in enumerate(self.csv_paths):
            gen = iter_csv_packets(path)
            streams.append((rx, gen))
        # seq_id 기준 힙 병합
        heap = []
        for rx, gen in streams:
            for sid, amp, _ts in gen:
                heapq.heappush(heap, (sid, rx, amp.tobytes()))
                break
        gens = {rx: gen for rx, gen in streams}
        while heap and not self._stop.is_set():
            sid, rx, amp_b = heapq.heappop(heap)
            amp = np.frombuffer(amp_b, dtype=np.float32)
            out_queue.put(Packet(rx, sid, amp, time.monotonic()))
            for nsid, namp, _ts in gens[rx]:
                heapq.heappush(heap, (nsid, rx, namp.tobytes()))
                break
        out_queue.put(None)
        self.finished.set()

    def _paced_loop(self, rx, path, out_queue):
        prev_ts = None
        for sid, amp, ts in iter_csv_packets(path):
            if self._stop.is_set():
                break
            if prev_ts is not None and ts is not None:
                dt = (ts - prev_ts) / self.speed
                if 0 < dt < 5:
                    time.sleep(dt)
            prev_ts = ts
            now = time.monotonic()
            targets = range(4) if self.mirror else (rx,)
            for r in targets:
                out_queue.put(Packet(r, sid, amp, now))
        with self._done_lock:
            self._n_done += 1
            if self._n_done == len(self.csv_paths):
                out_queue.put(None)
                self.finished.set()

    def stop(self) -> None:
        self._stop.set()
