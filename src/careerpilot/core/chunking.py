"""Document chunking for the Knowledge Agent's RAG pipeline.

A small, dependency-free splitter: break text on paragraph boundaries first, then
pack paragraphs into ~``chunk_size``-character windows with a trailing ``overlap`` so
a fact split across a boundary still appears whole in at least one chunk. Defaults are
tuned for short policy/finance docs; retune for a very different corpus.
"""

from __future__ import annotations


def chunk_document(text: str, *, chunk_size: int = 400, chunk_overlap: int = 60) -> list[str]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        if not current:
            current = para
        elif len(current) + 2 + len(para) <= chunk_size:
            current = f"{current}\n\n{para}"
        else:
            chunks.append(current)
            tail = current[-chunk_overlap:] if chunk_overlap else ""
            current = f"{tail}\n\n{para}".strip() if tail else para

    if current:
        chunks.append(current)

    # A single paragraph longer than chunk_size is hard-split so nothing is dropped.
    out: list[str] = []
    for chunk in chunks:
        if len(chunk) <= chunk_size:
            out.append(chunk)
            continue
        step = chunk_size - chunk_overlap
        for start in range(0, len(chunk), step):
            out.append(chunk[start : start + chunk_size])
    return out
