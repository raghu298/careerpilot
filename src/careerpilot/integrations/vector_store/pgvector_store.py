"""Vector store seam: a protocol plus a Postgres/pgvector adapter and an in-memory
fake used by the tests. Documents carry an ``access_scope`` label so the Knowledge
Agent's retrieval is filtered by persona capability inside the query, before any
chunk reaches the model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol


@dataclass
class RetrievedChunk:
    content: str
    source: str
    score: float


class VectorStore(Protocol):
    async def add(
        self, *, content: str, embedding: list[float], source: str, access_scope: str
    ) -> None: ...

    async def search(
        self, *, embedding: list[float], scopes: tuple[str, ...], k: int = 5
    ) -> list[RetrievedChunk]: ...


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class InMemoryVectorStore:
    """Fake store for tests and offline demos — exact cosine search, scope-filtered."""

    def __init__(self) -> None:
        self._rows: list[tuple[str, list[float], str, str]] = []

    async def add(self, *, content: str, embedding: list[float], source: str, access_scope: str) -> None:
        self._rows.append((content, embedding, source, access_scope))

    async def search(self, *, embedding: list[float], scopes: tuple[str, ...], k: int = 5) -> list[RetrievedChunk]:
        scored = [
            RetrievedChunk(content=c, source=s, score=_cosine(embedding, e))
            for (c, e, s, scope) in self._rows
            if scope in scopes
        ]
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:k]


class PgVectorStore:
    """Real adapter over Postgres + pgvector. Scope filtering happens in SQL so a
    persona can never retrieve a document outside its capability set."""

    def __init__(self, pool) -> None:  # asyncpg pool, injected
        self._pool = pool

    async def add(self, *, content: str, embedding: list[float], source: str, access_scope: str) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO kb_chunks (content, embedding, source, access_scope) "
                "VALUES ($1, $2, $3, $4)",
                content,
                embedding,
                source,
                access_scope,
            )

    async def search(self, *, embedding: list[float], scopes: tuple[str, ...], k: int = 5) -> list[RetrievedChunk]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT content, source, 1 - (embedding <=> $1) AS score "
                "FROM kb_chunks WHERE access_scope = ANY($2) "
                "ORDER BY embedding <=> $1 LIMIT $3",
                embedding,
                list(scopes),
                k,
            )
        return [RetrievedChunk(content=r["content"], source=r["source"], score=r["score"]) for r in rows]
