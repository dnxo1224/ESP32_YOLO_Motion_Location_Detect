"""
TcpFrameServer — 송신 ESP의 raw TCP 스트림을 받아 프레임으로 복원.

설계 문서(REALTIME 4TX-1RX PIPELINE) 5번 모듈:
  고정 헤더로 스트림 내 프레임 경계 탐색 → 체크섬 검증 → 깨진 프레임 폐기

복원된 Frame은 콜백(on_frame)으로 전달된다. 이후 단계(CsiSyncEngine)가
이 콜백에 연결될 예정이며, 지금은 단독 실행 시 통계만 출력한다.

단독 실행:
    python Realtime/tcp_source.py --port 5010
"""
import argparse
import socket
import threading
import time

from csi_frame import FrameParser


class TcpFrameServer:
    def __init__(self, host: str = '0.0.0.0', port: int = 5010,
                 on_frame=None):
        self.host = host
        self.port = port
        self.on_frame = on_frame
        self._stop = threading.Event()
        self._threads = []
        self._sock = None

        # 통계 (여러 연결 누적)
        self.parser = FrameParser()
        self.n_bytes = 0
        self.n_conns = 0
        self.n_disconnects = 0
        self.last_frame = None
        # seq 연속성 (게이트 2·3: 테스트 생성기는 전역 seq가 1씩 증가.
        #  진짜 Rx 연결 후에는 Tx별 seq가 되므로 tx_id별 추적으로 바꿀 것)
        self.n_seq_missing = 0
        self._last_seq = None
        self._lock = threading.Lock()

    # ── 수신 ──────────────────────────────────────────────────────────────────

    def start(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self.host, self.port))
        self._sock.listen(4)
        self._sock.settimeout(0.5)
        t = threading.Thread(target=self._accept_loop, daemon=True,
                             name='tcp-accept')
        t.start()
        self._threads.append(t)
        print(f'[TcpFrameServer] listening on {self.host}:{self.port}')

    def _accept_loop(self):
        while not self._stop.is_set():
            try:
                conn, addr = self._sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            self.n_conns += 1
            print(f'[TcpFrameServer] client connected: {addr}')
            t = threading.Thread(target=self._recv_loop, args=(conn, addr),
                                 daemon=True, name=f'tcp-recv-{addr[1]}')
            t.start()
            self._threads.append(t)

    def _recv_loop(self, conn: socket.socket, addr):
        conn.settimeout(1.0)
        try:
            while not self._stop.is_set():
                try:
                    chunk = conn.recv(65536)
                except socket.timeout:
                    continue
                except OSError:
                    break
                if not chunk:
                    break
                with self._lock:
                    self.n_bytes += len(chunk)
                    frames = self.parser.feed(chunk)
                for fr in frames:
                    if self._last_seq is not None and fr.seq > self._last_seq + 1:
                        self.n_seq_missing += fr.seq - self._last_seq - 1
                    self._last_seq = fr.seq
                    self.last_frame = fr
                    if self.on_frame is not None:
                        self.on_frame(fr)
        finally:
            conn.close()
            self.n_disconnects += 1
            print(f'[TcpFrameServer] client disconnected: {addr}')

    def stop(self):
        self._stop.set()
        if self._sock is not None:
            self._sock.close()

    def stats(self) -> str:
        lf = self.last_frame
        last = (f'last: tx{lf.tx_id} seq={lf.seq} rssi={lf.rssi}'
                if lf else 'last: -')
        return (f'{self.parser.stats()} seq_missing={self.n_seq_missing} '
                f'bytes={self.n_bytes} '
                f'conns={self.n_conns}/-{self.n_disconnects} | {last}')


def main():
    ap = argparse.ArgumentParser(description='CSI TCP 프레임 서버 (단독 통계 모드)')
    ap.add_argument('--host', default='0.0.0.0')
    ap.add_argument('--port', type=int, default=5010)
    ap.add_argument('--interval', type=float, default=2.0,
                    help='통계 출력 주기 (초)')
    args = ap.parse_args()

    server = TcpFrameServer(args.host, args.port)
    server.start()

    prev_ok = 0
    try:
        while True:
            time.sleep(args.interval)
            ok = server.parser.n_ok
            fps = (ok - prev_ok) / args.interval
            prev_ok = ok
            print(f'[stats] {server.stats()} | {fps:.1f} fps')
    except KeyboardInterrupt:
        print('\n[TcpFrameServer] 종료')
        server.stop()


if __name__ == '__main__':
    main()
