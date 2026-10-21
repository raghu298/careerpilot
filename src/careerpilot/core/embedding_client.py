"""Embedding gateway seam — same ports-and-adapters pattern as the LLM gateway."""

from __future__ import annotations

import hashlib
import os
from collections import OrderedDict
from typing import Protocol


class EmbeddingClient(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text, in order."""
        ...


class CachingEmbeddingClient:
    """Shared-Foundation embedding cache: memoizes vectors by text so identical queries
    or re-seeded documents aren't re-embedded. Real latency/cost win with a paid model;
    harmless offline. LRU-bounded; records hit/miss metrics."""

    def __init__(self, inner: EmbeddingClient, *, maxsize: int = 4096) -> None:
        self._inner = inner
        self._maxsize = maxsize
        self._cache: "OrderedDict[str, list[float]]" = OrderedDict()

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        from careerpilot.core.observability import metrics

        results: list[list[float] | None] = [None] * len(texts)
        missing_idx: list[int] = []
        missing_text: list[str] = []
        for i, t in enumerate(texts):
            key = self._key(t)
            hit = self._cache.get(key)
            if hit is not None:
                self._cache.move_to_end(key)
                results[i] = hit
                metrics.incr("embeddings.cache_hit")
            else:
                missing_idx.append(i)
                missing_text.append(t)
                metrics.incr("embeddings.cache_miss")

        if missing_text:
            fresh = await self._inner.embed(missing_text)
            for i, t, vec in zip(missing_idx, missing_text, fresh):
                results[i] = vec
                self._cache[self._key(t)] = vec
                self._cache.move_to_end(self._key(t))
            while len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)

        return [r for r in results if r is not None]


class OpenAIEmbeddingClient:
    def __init__(self, api_key: str | None = None, model: str = "text-embedding-3-small") -> None:
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self._model = model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        resp = await self._client.embeddings.create(model=self._model, input=texts)
        return [item.embedding for item in resp.data]


def default_embedding_client() -> EmbeddingClient:
    return OpenAIEmbeddingClient()
