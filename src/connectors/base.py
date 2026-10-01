"""RAG connector interface (spec §11).

Umbra only ever calls a RAG system's public query interface: question in,
retrieved chunks + generated answer out.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

# Keys RAG APIs commonly use for the chunk list and for chunk text. from_dict
# accepts any of them so the HTTP connector works with most REST RAG services.
_CHUNK_LIST_KEYS = ("chunks", "contexts", "source_documents", "documents", "sources", "results")
_CHUNK_TEXT_KEYS = ("text", "content", "page_content", "chunk", "document")
_ANSWER_KEYS = ("answer", "result", "response", "output")


@dataclass
class RetrievedChunk:
    text: str
    chunk_id: Optional[str] = None
    score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_obj(cls, obj: Any, position: int) -> "RetrievedChunk":
        if isinstance(obj, str):
            return cls(text=obj, chunk_id=str(position))
        if not isinstance(obj, dict):
            raise ValueError(f"unsupported chunk type: {type(obj).__name__}")
        text = next((obj[k] for k in _CHUNK_TEXT_KEYS if isinstance(obj.get(k), str)), None)
        if text is None:
            raise ValueError(f"chunk has no text field (tried {_CHUNK_TEXT_KEYS}): {list(obj)}")
        chunk_id = obj.get("chunk_id", obj.get("id"))
        score = obj.get("score")
        metadata = obj.get("metadata") or {
            k: v for k, v in obj.items() if k not in (*_CHUNK_TEXT_KEYS, "chunk_id", "id", "score")
        }
        return cls(
            text=text,
            chunk_id=str(chunk_id) if chunk_id is not None else str(position),
            score=float(score) if score is not None else None,
            metadata=metadata,
        )


@dataclass
class RAGResponse:
    question: str
    chunks: List[RetrievedChunk]
    answer: str = ""
    error: Optional[str] = None  # set when the query failed; chunks is then empty

    @property
    def chunk_texts(self) -> List[str]:
        return [c.text for c in self.chunks]

    @property
    def chunk_ids(self) -> List[str]:
        return [c.chunk_id or str(i) for i, c in enumerate(self.chunks)]

    @classmethod
    def from_dict(cls, data: Dict[str, Any], question: Optional[str] = None) -> "RAGResponse":
        raw_chunks = next((data[k] for k in _CHUNK_LIST_KEYS if isinstance(data.get(k), list)), None)
        if raw_chunks is None:
            raise ValueError(f"response has no chunk list (tried {_CHUNK_LIST_KEYS}): {list(data)}")
        answer = next((data[k] for k in _ANSWER_KEYS if isinstance(data.get(k), str)), "")
        q = question if question is not None else data.get("question", data.get("query", ""))
        return cls(
            question=q,
            chunks=[RetrievedChunk.from_obj(c, i) for i, c in enumerate(raw_chunks)],
            answer=answer,
        )


class RAGConnector(ABC):
    @abstractmethod
    async def query(self, question: str) -> RAGResponse:
        """Submit a query and get retrieved chunks + generated answer."""

    async def query_many(
        self,
        questions: Sequence[str],
        concurrency: int = 50,
        on_result: Optional[Callable[[RAGResponse], None]] = None,
    ) -> List[RAGResponse]:
        """`on_result` is called with each response as soon as it arrives, in
        completion order (the returned list is in input order)."""
        # one bad probe shouldn't kill a 1000-probe audit, so failures come back as RAGResponse(error=...)
        sem = asyncio.Semaphore(concurrency)

        async def one(q: str) -> RAGResponse:
            async with sem:
                try:
                    resp = await self.query(q)
                except Exception as exc:  # noqa: BLE001 - recorded per probe
                    resp = RAGResponse(question=q, chunks=[], error=f"{type(exc).__name__}: {exc}")
            if on_result:
                on_result(resp)
            return resp

        return await asyncio.gather(*(one(q) for q in questions))

    async def aclose(self) -> None:
        """Release resources (HTTP clients etc.)."""

    async def __aenter__(self) -> "RAGConnector":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()
