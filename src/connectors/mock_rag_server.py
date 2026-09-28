"""Minimal RAG service over a local KB, for testing Umbra end to end.

Retrieval: cosine similarity between the query and paragraph chunks, both
embedded with all-MiniLM-L6-v2. Generation: extractive (the sentence in the
retrieved chunks most similar to the query). Like many real RAG systems, it
always answers, even when nothing relevant was retrieved.

Contract (same as HTTPRAGConnector expects):
    POST /query  {"query": str, "top_k": int = 5}
      -> {"question": str,
          "chunks": [{"chunk_id", "doc_id", "text", "score"}, ...],
          "answer": str}
    GET  /health -> {"status": "ok", "documents": int, "chunks": int, "traffic": {...}}

It can also act like a slow, rate-limited production endpoint, to exercise
the connector's concurrency and Retry-After handling:
    latency     seconds added to every /query
    rate_limit  max /query requests per rate_window seconds; the rest get a
                429 with Retry-After set to the time left in the window

Run:
    UMBRA_KB_PATH=data/synthetic_kb uvicorn src.connectors.mock_rag_server:app --port 8765
    (UMBRA_MOCK_LATENCY, UMBRA_MOCK_RATE_LIMIT, UMBRA_MOCK_RATE_WINDOW set the above)
"""

from __future__ import annotations

import asyncio
import math
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.data.kb_loader import load_chunks, load_documents
from src.embeddings import embed

DEFAULT_KB_PATH = "data/synthetic_kb"


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)


class ChunkOut(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    score: float


class QueryResponse(BaseModel):
    question: str
    chunks: List[ChunkOut]
    answer: str


class VectorIndex:
    def __init__(self, kb_path: str | Path):
        self.n_documents = len(load_documents(kb_path))
        self.chunks = load_chunks(kb_path)
        self.matrix = embed([c.text for c in self.chunks])  # L2-normalized

    def search(self, query: str, top_k: int) -> List[ChunkOut]:
        q = embed([query])[0]
        sims = self.matrix @ q
        top = np.argsort(-sims)[:top_k]
        return [
            ChunkOut(chunk_id=self.chunks[i].chunk_id, doc_id=self.chunks[i].doc_id,
                     text=self.chunks[i].text, score=float(sims[i]))
            for i in top
        ]


def extractive_answer(query: str, chunks: List[ChunkOut]) -> str:
    sentences = [s for c in chunks for s in re.split(r"(?<=[.!?])\s+", c.text) if s.strip()]
    if not sentences:
        return "I could not find any information about that."
    embs = embed([query] + sentences)
    best = int(np.argmax(embs[1:] @ embs[0]))
    return f"Based on the available context: {sentences[best]}"


@dataclass
class Traffic:
    served: int = 0
    rejected: int = 0
    in_flight: int = 0
    max_in_flight: int = 0
    max_served_per_window: int = 0


class FixedWindowLimiter:
    def __init__(self, limit: int, window: float):
        self.limit, self.window = limit, window
        self.window_start, self.count = time.monotonic(), 0

    def admit(self) -> Optional[float]:
        """None if admitted, else seconds until the window resets."""
        now = time.monotonic()
        if now - self.window_start >= self.window:
            self.window_start, self.count = now, 0
        if self.count < self.limit:
            self.count += 1
            return None
        return self.window - (now - self.window_start)


def _env_float(name: str) -> Optional[float]:
    value = os.environ.get(name)
    return float(value) if value else None


def create_app(
    kb_path: str | Path | None = None,
    *,
    latency: Optional[float] = None,
    rate_limit: Optional[int] = None,
    rate_window: Optional[float] = None,
) -> FastAPI:
    kb_path = kb_path or os.environ.get("UMBRA_KB_PATH", DEFAULT_KB_PATH)
    latency = latency if latency is not None else (_env_float("UMBRA_MOCK_LATENCY") or 0.0)
    if rate_limit is None and os.environ.get("UMBRA_MOCK_RATE_LIMIT"):
        rate_limit = int(os.environ["UMBRA_MOCK_RATE_LIMIT"])
    rate_window = rate_window if rate_window is not None else (_env_float("UMBRA_MOCK_RATE_WINDOW") or 1.0)
    index = VectorIndex(kb_path)
    app = FastAPI(title="Umbra mock RAG server")
    traffic = app.state.traffic = Traffic()
    limiter = FixedWindowLimiter(rate_limit, rate_window) if rate_limit else None

    @app.middleware("http")
    async def throttle(request: Request, call_next):
        if request.url.path != "/query":
            return await call_next(request)
        if limiter and (wait := limiter.admit()) is not None:
            traffic.rejected += 1
            # Real servers send whole seconds; hundredths keep the tests fast.
            return JSONResponse({"detail": "rate limited"}, status_code=429,
                                headers={"Retry-After": f"{math.ceil(wait * 100) / 100:.2f}"})
        if limiter:
            traffic.max_served_per_window = max(traffic.max_served_per_window, limiter.count)
        traffic.in_flight += 1
        traffic.max_in_flight = max(traffic.max_in_flight, traffic.in_flight)
        try:
            if latency:
                await asyncio.sleep(latency)
            response = await call_next(request)
        finally:
            traffic.in_flight -= 1
        traffic.served += 1
        return response

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "documents": index.n_documents, "chunks": len(index.chunks),
                "traffic": asdict(traffic)}

    @app.post("/query", response_model=QueryResponse)
    def query(req: QueryRequest) -> QueryResponse:
        chunks = index.search(req.query, req.top_k)
        return QueryResponse(question=req.query, chunks=chunks, answer=extractive_answer(req.query, chunks))

    return app


def __getattr__(name: str):
    # `uvicorn src.connectors.mock_rag_server:app` builds the index on first
    # access instead of at import time, so importing this module stays cheap.
    if name == "app":
        globals()["app"] = create_app()
        return globals()["app"]
    raise AttributeError(name)
