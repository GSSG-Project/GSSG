"""Minimal per-frame FPS meter. One context-manager call per processed frame;
prints mean + recent-window FPS every N frames.

Uses print() rather than logging so it's always visible regardless of the
project's log_level (mapper/tracker subprocesses set root logger to WARNING).
"""

import time
from collections import deque


class FPSMeter:
    def __init__(
        self, name: str = "FPS", log_every: int = 20, window: int = 20, silent: bool = False
    ):
        self.name = name
        self.log_every = max(1, int(log_every))
        self.recent = deque(maxlen=max(1, int(window)))
        self.total_ms = 0.0
        self.frames = 0
        self.silent = silent
        self._t0 = None

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._t0 is None:
            return False
        dt_ms = (time.perf_counter() - self._t0) * 1000.0
        self._t0 = None
        self.recent.append(dt_ms)
        self.total_ms += dt_ms
        self.frames += 1
        if not self.silent and self.frames % self.log_every == 0:
            self._log()
        return False

    @property
    def recent_ms(self) -> float:
        return sum(self.recent) / len(self.recent) if self.recent else 0.0

    @property
    def recent_fps(self) -> float:
        ms = self.recent_ms
        return 1000.0 / ms if ms > 0 else 0.0

    def _log(self):
        recent_ms = self.recent_ms
        recent_fps = 1000.0 / recent_ms if recent_ms > 0 else 0.0
        print(
            f"[{self.name}]  {recent_fps:5.2f} FPS ({recent_ms:6.1f} ms)  frame {self.frames:5d}",
            flush=True,
        )
