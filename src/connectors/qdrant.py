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
from qdrant_client.common.client_exceptions import ResourceExhaustedResponse
from qdrant_client.http.exceptions import UnexpectedResponse

from src.connectors.base import _ANSWER_KEYS, _CHUNK_TEXT_KEYS, RAGConnector, RAGResponse, RetrievedChunk
from src.connectors.http import RateLimitGate, RetryingClient, backoff
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
        max_rate_limit_retries: int = 20,
        max_retry_after: float = 60.0,
        backoff_base: float = 0.5,
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
        self.max_rate_limit_retries = max_rate_limit_retries
        self.backoff_base = backoff_base
        # Qdrant Cloud rate-limits too; same shared-cooldown handling as the HTTP connector
        self.gate = RateLimitGate(max_retry_after)
        self.llm = (RetryingClient(llm_auth_header, max_rate_limit_retries=max_rate_limit_retries,
                                   max_retry_after=max_retry_after, backoff_base=backoff_base, **llm_client_kwargs)
                    if llm_endpoint else None)
        self._vectors: Dict[str, np.ndarray] = {}

    async def query(self, question: str) -> RAGResponse:
        vector = self._vectors.pop(question, None)
        if vector is None:
            [vector] = await asyncio.to_thread(self.embed_fn, [question])
        result = await self._search(vector)
        chunks = [self._chunk(p, i) for i, p in enumerate(result.points)]
        answer = await self._generate(question, chunks) if self.llm else ""
        return RAGResponse(question=question, chunks=chunks, answer=answer)

    async def query_many(
        self,
        questions: Sequence[str],
        concurrency: int = 50,
        on_result: Optional[Callable[[RAGResponse], None]] = None,
    ) -> List[RAGResponse]:
        # one batched encode instead of len(questions) calls fighting over the model lock
        unique = list(dict.fromkeys(questions))
        vectors = await asyncio.to_thread(self.embed_fn, unique)
        self._vectors.update(zip(unique, vectors))
        try:
            return await super().query_many(questions, concurrency, on_result)
        finally:
            self._vectors.clear()

    @property
    def n_rate_limited(self) -> int:
        return self.gate.n_rate_limited + (self.llm.n_rate_limited if self.llm else 0)

    async def _search(self, vector: np.ndarray) -> models.QueryResponse:
        attempt = 0
        while True:
            await self.gate.wait()
            try:
                return await self.client.query_points(
                    self.collection, query=vector.tolist(), using=self.vector_name, limit=self.top_k,
                    with_payload=True,
                )
            except ResourceExhaustedResponse as exc:  # a 429 with Retry-After
                if attempt >= self.max_rate_limit_retries:
                    raise
                delay = self.gate.retry_after(exc.retry_after_s)
            except UnexpectedResponse as exc:  # qdrant_client raises this for a 429 without one
                if exc.status_code != 429 or attempt >= self.max_rate_limit_retries:
                    raise
                delay = self.gate.retry_after(exc.headers.get("retry-after"))
            self.gate.hit(backoff(min(attempt, 6), self.backoff_base) if delay is None else delay)
            attempt += 1

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
