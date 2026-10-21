"""Observability — the Shared-Foundation layer every agent reports into.

A tiny dependency-free metrics registry (counters + latency samples) plus structured
logging. The API exposes a snapshot at ``GET /metrics``. This is deliberately in-process
and thread-safe enough for the single-worker demo; swap the registry for Prometheus/
OTel in production without touching call sites (same ports-and-adapters spirit).
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from collections import defaultdict
from contextlib import contextmanager


class Metrics:
    """In-process counters and latency samples with a JSON-able snapshot."""

    def __init__(self, *, max_samples: int = 1000) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, float] = defaultdict(float)
        self._latencies: dict[str, list[float]] = defaultdict(list)
        self._max_samples = max_samples

    def incr(self, name: str, n: float = 1.0) -> None:
        with self._lock:
            self._counters[name] += n

    def observe_ms(self, name: str, ms: float) -> None:
        with self._lock:
            s = self._latencies[name]
            s.append(ms)
            if len(s) > self._max_samples:
                del s[: len(s) - self._max_samples]

    @contextmanager
    def timer(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.observe_ms(name, (time.perf_counter() - start) * 1000.0)

    def snapshot(self) -> dict:
        with self._lock:
            counters = dict(sorted(self._counters.items()))
            latencies = {}
            for name, samples in sorted(self._latencies.items()):
                if samples:
                    latencies[name] = _latency_stats(samples)
        return {"counters": counters, "latency_ms": latencies}

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._latencies.clear()


def _latency_stats(samples: list[float]) -> dict:
    ordered = sorted(samples)
    n = len(ordered)

    def pct(p: float) -> float:
        if n == 1:
            return round(ordered[0], 1)
        idx = min(n - 1, int(round((p / 100.0) * (n - 1))))
        return round(ordered[idx], 1)

    return {
        "count": n,
        "avg": round(sum(ordered) / n, 1),
        "p50": pct(50),
        "p95": pct(95),
        "max": round(ordered[-1], 1),
    }


# Process-wide singleton used across the orchestrator, gateway, and API.
metrics = Metrics()

_LOG_CONFIGURED = False


def configure_logging(level: int = logging.INFO) -> None:
    """Idempotent structured-ish logging to stdout (one line per event, key=value)."""
    global _LOG_CONFIGURED
    if _LOG_CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    root = logging.getLogger("careerpilot")
    root.handlers[:] = [handler]
    root.setLevel(level)
    root.propagate = False
    _LOG_CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(f"careerpilot.{name}")
