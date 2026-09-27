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
    GET  /health -> {"status": "ok", "documents": int, "chunks": int}

Run:
    UMBRA_KB_PATH=data/synthetic_kb uvicorn src.connectors.mock_rag_server:app --port 8765
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List

import numpy as np
from fastapi import FastAPI
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


def create_app(kb_path: str | Path | None = None) -> FastAPI:
    kb_path = kb_path or os.environ.get("UMBRA_KB_PATH", DEFAULT_KB_PATH)
    index = VectorIndex(kb_path)
    app = FastAPI(title="Umbra mock RAG server")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "documents": index.n_documents, "chunks": len(index.chunks)}

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
