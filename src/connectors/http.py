"""Generic HTTP connector: works with any RAG system exposing a REST query API."""

from __future__ import annotations

import asyncio
import random
from typing import Any, Dict, Optional

import httpx

from src.connectors.base import RAGConnector, RAGResponse

RETRY_STATUSES = {408, 429, 500, 502, 503, 504}


class HTTPRAGConnector(RAGConnector):
    """POSTs {"query": question, **extra_payload} to `endpoint_url` and parses
    the JSON response with RAGResponse.from_dict.

    Retries 408/429/5xx and connection errors with exponential backoff,
    honoring Retry-After headers (spec §14).
    """

    def __init__(
        self,
        endpoint_url: str,
        auth_header: Optional[str] = None,
        *,
        query_field: str = "query",
        extra_payload: Optional[Dict[str, Any]] = None,
        timeout: float = 60.0,
        max_retries: int = 4,
        backoff_base: float = 0.5,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ):
        self.endpoint_url = endpoint_url
        self.query_field = query_field
        self.extra_payload = extra_payload or {}
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        headers = {"Authorization": auth_header} if auth_header else {}
        self._client = httpx.AsyncClient(headers=headers, timeout=timeout, transport=transport)

    async def query(self, question: str) -> RAGResponse:
        payload = {self.query_field: question, **self.extra_payload}
        for attempt in range(self.max_retries + 1):
            last_attempt = attempt == self.max_retries
            try:
                resp = await self._client.post(self.endpoint_url, json=payload)
            except (httpx.TransportError, httpx.TimeoutException):
                if last_attempt:
                    raise
                await asyncio.sleep(self._backoff(attempt))
                continue
            if resp.status_code in RETRY_STATUSES and not last_attempt:
                await asyncio.sleep(self._retry_after(resp) or self._backoff(attempt))
                continue
            resp.raise_for_status()
            return RAGResponse.from_dict(resp.json(), question=question)
        raise RuntimeError("unreachable")  # pragma: no cover

    def _backoff(self, attempt: int) -> float:
        return self.backoff_base * (2**attempt) * (1 + random.random() * 0.25)

    @staticmethod
    def _retry_after(resp: httpx.Response) -> Optional[float]:
        value = resp.headers.get("retry-after")
        try:
            return min(float(value), 60.0) if value is not None else None
        except ValueError:
            return None  # HTTP-date form: fall back to exponential backoff

    async def aclose(self) -> None:
        await self._client.aclose()
