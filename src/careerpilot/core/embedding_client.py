"""Embedding gateway seam — same ports-and-adapters pattern as the LLM gateway."""

from __future__ import annotations

import os
from typing import Protocol


class EmbeddingClient(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text, in order."""
        ...


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
