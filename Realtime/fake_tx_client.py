"""
가짜 송신 클라이언트 — tcp_source.py 검증용 (하드웨어 불필요).

송신 ESP가 보낼 스트림을 흉내 낸다:
  - 정상 프레임 (4 Tx 순환, seq 증가, 합성 CSI)
  - --garbage-every N : N프레임마다 앞에 쓰레기 바이트 삽입 (경계 재동기화 시험)
  - --corrupt-every N : N프레임마다 1바이트 오염 (CRC 검증 시험)
  - 임의 크기 청크로 쪼개 전송 (TCP 세그먼트 경계 재조립 시험)

실행:
    python Realtime/tcp_source.py --port 5010          # 터미널 1
    python Realtime/fake_tx_client.py --count 2000     # 터미널 2
"""
import argparse
import math
import random
import socket
import time

from csi_frame import CSI_LEN, build_frame


def synth_csi(seq: int, tx_id: int) -> bytes:
    """합성 I/Q — 서브캐리어별 사인 패턴 (내용은 무엇이든 무방)."""
    out = bytearray(CSI_LEN)
    for k in range(CSI_LEN // 2):
        amp = 40 + 20 * math.sin(k * 0.1 + seq * 0.05 + tx_id)
        out[2 * k] = int(amp) & 0xFF                       # I
        out[2 * k + 1] = int(amp * 0.5) & 0xFF             # Q
    return bytes(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=5010)
    ap.add_argument('--count', type=int, default=2000, help='송신 프레임 수')
    ap.add_argument('--rate', type=float, default=0,
                    help='초당 프레임 수 (0 = 전속력)')
    ap.add_argument('--garbage-every', type=int, default=50)
    ap.add_argument('--corrupt-every', type=int, default=25)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    sock = socket.create_connection((args.host, args.port))
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    n_good = n_corrupt = n_garbage = 0
    pending = bytearray()
    t0 = time.perf_counter()

    for i in range(args.count):
        if args.garbage_every and i % args.garbage_every == args.garbage_every - 1:
            junk = bytes(random.randrange(256)
                         for _ in range(random.randint(1, 64)))
            pending += junk
            n_garbage += 1

        tx_id = i % 4
        frame = bytearray(build_frame(
            tx_id=tx_id, seq=i, rx_timestamp_us=i * 33333,
            csi=synth_csi(i, tx_id), rssi=-45 - tx_id, channel=36,
        ))
        if args.corrupt_every and i % args.corrupt_every == args.corrupt_every - 1:
            frame[random.randrange(4, len(frame))] ^= 0xFF
            n_corrupt += 1
        else:
            n_good += 1
        pending += frame

        # 임의 크기 청크로 밀어내기 (프레임 경계와 무관한 세그먼트 재현)
        while len(pending) > 512:
            n = random.randint(1, 1000)
            sock.sendall(bytes(pending[:n]))
            del pending[:n]

        if args.rate > 0:
            time.sleep(1.0 / args.rate)

    sock.sendall(bytes(pending))
    sock.close()

    dt = time.perf_counter() - t0
    print(f'[fake_tx_client] 송신 완료: 정상 {n_good}, 오염 {n_corrupt}, '
          f'쓰레기 삽입 {n_garbage}회, {args.count / dt:.0f} fps')
    print(f'[기대값] 서버 ok={n_good}, crc_fail>={n_corrupt}')


if __name__ == '__main__':
    main()
