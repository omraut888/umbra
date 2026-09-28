"""Native Qdrant connector (spec §11): retrieve straight from a collection,
optionally generate an answer through an LLM endpoint.

Umbra's scores only use the retrieved chunks, so `llm_endpoint` is optional;
without it the answer is left empty. With it, each probe POSTs
{"question": str, "contexts": [str, ...]} and reads the answer from the JSON
reply ("answer"/"result"/"response"/"output") or the raw body.

The query vector has to come from the model the collection was built with.
The default is Umbra's own all-MiniLM-L6-v2, which is what `umbra index-qdrant`
writes; for someone else's collection pass their embedder as `embed_fn`.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
from qdrant_client import AsyncQdrantClient, models

from src.connectors.base import _ANSWER_KEYS, _CHUNK_TEXT_KEYS, RAGConnector, RAGResponse, RetrievedChunk
from src.connectors.http import RetryingClient
from src.embeddings import embed

EmbedFn = Callable[[Sequence[str]], np.ndarray]


class QdrantRAGConnector(RAGConnector):
    def __init__(
        self,
        qdrant_url: Optional[str] = None,
        collection: str = "umbra_kb",
        llm_endpoint: Optional[str] = None,
        *,
        api_key: Optional[str] = None,
        client: Optional[AsyncQdrantClient] = None,
        top_k: int = 5,
        vector_name: Optional[str] = None,
        text_key: Optional[str] = None,
        embed_fn: EmbedFn = embed,
        llm_auth_header: Optional[str] = None,
        **llm_client_kwargs: Any,
    ):
        if client is None and qdrant_url is None:
            raise ValueError("pass qdrant_url or client")
        self.client = client or AsyncQdrantClient(url=qdrant_url, api_key=api_key)
        self.collection = collection
        self.llm_endpoint = llm_endpoint
        self.top_k = top_k
        self.vector_name = vector_name
        self.text_key = text_key
        self.embed_fn = embed_fn
        self.llm = RetryingClient(llm_auth_header, **llm_client_kwargs) if llm_endpoint else None
        self._vectors: Dict[str, np.ndarray] = {}

    async def query(self, question: str) -> RAGResponse:
        vector = self._vectors.pop(question, None)
        if vector is None:
            [vector] = await asyncio.to_thread(self.embed_fn, [question])
        result = await self.client.query_points(
            self.collection, query=vector.tolist(), using=self.vector_name, limit=self.top_k, with_payload=True,
        )
        chunks = [self._chunk(p, i) for i, p in enumerate(result.points)]
        answer = await self._generate(question, chunks) if self.llm else ""
        return RAGResponse(question=question, chunks=chunks, answer=answer)

    async def query_many(self, questions: Sequence[str], concurrency: int = 50) -> List[RAGResponse]:
        # one batched encode instead of len(questions) calls fighting over the model lock
        unique = list(dict.fromkeys(questions))
        vectors = await asyncio.to_thread(self.embed_fn, unique)
        self._vectors.update(zip(unique, vectors))
        try:
            return await super().query_many(questions, concurrency)
        finally:
            self._vectors.clear()

    def _chunk(self, point: models.ScoredPoint, position: int) -> RetrievedChunk:
        payload = dict(point.payload or {})
        if self.text_key and self.text_key not in _CHUNK_TEXT_KEYS:
            payload["text"] = payload.pop(self.text_key, None)
        chunk = RetrievedChunk.from_obj({**payload, "score": point.score}, position)
        # payload chunk_id if the indexer wrote one, else Qdrant's point id
        if "chunk_id" not in payload and "id" not in payload:
            chunk.chunk_id = str(point.id)
        return chunk

    async def _generate(self, question: str, chunks: List[RetrievedChunk]) -> str:
        resp = await self.llm.post_json(self.llm_endpoint, {"question": question, "contexts": [c.text for c in chunks]})
        try:
            data = resp.json()
        except ValueError:
            return resp.text
        if isinstance(data, str):
            return data
        return next((data[k] for k in _ANSWER_KEYS if isinstance(data.get(k), str)), "")

    async def aclose(self) -> None:
        await self.client.close()
        if self.llm:
            await self.llm.aclose()


async def index_chunks(
    client: AsyncQdrantClient,
    collection: str,
    chunks: Sequence[Any],
    *,
    embed_fn: EmbedFn = embed,
    recreate: bool = False,
    batch_size: int = 256,
) -> int:
    """Load kb_loader.Chunk objects into `collection` (cosine, one unnamed
    vector), in the payload shape QdrantRAGConnector reads back."""
    if recreate and await client.collection_exists(collection):
        await client.delete_collection(collection)
    vectors = await asyncio.to_thread(embed_fn, [c.text for c in chunks])
    if not await client.collection_exists(collection):
        await client.create_collection(
            collection, vectors_config=models.VectorParams(size=vectors.shape[1], distance=models.Distance.COSINE),
        )
    points = [
        # Qdrant ids must be ints or UUIDs; derive a stable UUID so re-indexing overwrites
        models.PointStruct(id=str(uuid.uuid5(uuid.NAMESPACE_URL, c.chunk_id)), vector=v.tolist(),
                           payload={"chunk_id": c.chunk_id, "doc_id": c.doc_id, "text": c.text})
        for c, v in zip(chunks, vectors)
    ]
    for i in range(0, len(points), batch_size):
        await client.upsert(collection, points=points[i:i + batch_size])
    return len(points)
