"""예측 후처리: majority-vote 스무딩 + 결과 출력 Sink."""
import time
from collections import Counter, deque

from engine import Prediction
from models import ACTION_NAMES, ZONE_NAMES


class MajorityVoteSmoother:
    """최근 k개 예측의 다수결로 (zone, action)을 안정화한다."""

    def __init__(self, k: int = 5):
        self.k = k
        self._zones = deque(maxlen=k)
        self._actions = deque(maxlen=k)

    def update(self, pred: Prediction):
        self._zones.append(pred.zone)
        self._actions.append(pred.action)
        zone = Counter(self._zones).most_common(1)[0][0]
        action = Counter(self._actions).most_common(1)[0][0]
        return zone, action


class ResultSink:
    """대시보드 등 확장을 위한 추상 인터페이스."""

    def emit(self, pred: Prediction, smoothed=None) -> None:
        raise NotImplementedError


class ConsoleSink(ResultSink):
    def __init__(self):
        self._t0 = time.monotonic()

    def emit(self, pred: Prediction, smoothed=None) -> None:
        t = pred.t_wall - self._t0
        line = (f'[{t:7.1f}s] win@{pred.window_start:<5d} '
                f'{ZONE_NAMES[pred.zone]} ({pred.zone_conf:.0%}) | '
                f'{ACTION_NAMES[pred.action]:<7s} ({pred.action_conf:.0%})')
        if smoothed is not None:
            sz, sa = smoothed
            line += f' | smoothed: {ZONE_NAMES[sz]} / {ACTION_NAMES[sa]}'
        line += f' | NaN {pred.nan_ratio:.1%} | {pred.latency_ms:.0f}ms'
        print(line)


class RecordingSink(ResultSink):
    """검증용: 예측을 리스트에 누적."""

    def __init__(self):
        self.predictions = []

    def emit(self, pred: Prediction, smoothed=None) -> None:
        self.predictions.append(pred)
