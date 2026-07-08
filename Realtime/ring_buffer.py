"""고정 크기 프레임 링버퍼 (200, 664)."""
import numpy as np


class FrameRingBuffer:
    def __init__(self, capacity: int = 200, dim: int = 664):
        self.capacity = capacity
        self.dim = dim
        self._buf = np.full((capacity, dim), np.nan, dtype=np.float32)
        self._head = 0          # 다음 쓰기 위치
        self._count = 0         # 저장된 프레임 수 (최대 capacity)

    def push(self, frame: np.ndarray) -> None:
        self._buf[self._head] = frame
        self._head = (self._head + 1) % self.capacity
        if self._count < self.capacity:
            self._count += 1

    def full(self) -> bool:
        return self._count == self.capacity

    def snapshot(self) -> np.ndarray:
        """시간순 (capacity, dim) 복사본. full() 전 호출 금지."""
        if not self.full():
            raise RuntimeError('ring buffer not full yet')
        return np.concatenate(
            (self._buf[self._head:], self._buf[:self._head]), axis=0
        ).copy()
