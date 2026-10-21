"""Rate limiting — a token-bucket limiter used by the LLM Gateway and the API.

In-process and dependency-free (fine for the single-worker demo). Each key (a session
token, or a client IP for session creation) gets its own bucket that refills at a steady
rate up to a burst capacity. Swap for Redis in a multi-instance deployment without
changing call sites.
"""

from __future__ import annotations

import threading
import time


class _Bucket:
    __slots__ = ("tokens", "last")

    def __init__(self, capacity: float, now: float) -> None:
        self.tokens = capacity
        self.last = now


class RateLimiter:
    def __init__(self, *, rate_per_sec: float, capacity: float) -> None:
        self._rate = rate_per_sec
        self._capacity = capacity
        self._buckets: dict[str, _Bucket] = {}
        self._lock = threading.Lock()

    def allow(self, key: str, cost: float = 1.0) -> bool:
        """Consume ``cost`` tokens for ``key``; return False if the bucket is empty."""
        now = time.monotonic()
        with self._lock:
            b = self._buckets.get(key)
            if b is None:
                b = self._buckets[key] = _Bucket(self._capacity, now)
            # Refill since last check.
            b.tokens = min(self._capacity, b.tokens + (now - b.last) * self._rate)
            b.last = now
            if b.tokens >= cost:
                b.tokens -= cost
                return True
            return False

    def retry_after(self, key: str, cost: float = 1.0) -> float:
        """Seconds until ``cost`` tokens would be available (for a 429 Retry-After)."""
        with self._lock:
            b = self._buckets.get(key)
            tokens = b.tokens if b else self._capacity
        deficit = max(0.0, cost - tokens)
        return round(deficit / self._rate, 1) if self._rate else 0.0
