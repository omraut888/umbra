"""Generic HTTP connector: works with any RAG system exposing a REST query API."""

from __future__ import annotations

import asyncio
import random
import time
from email.utils import parsedate_to_datetime
from typing import Any, Dict, Optional

import httpx

from src.connectors.base import RAGConnector, RAGResponse

RETRY_STATUSES = {408, 429, 500, 502, 503, 504}


class HTTPRAGConnector(RAGConnector):
    """POSTs {"query": question, **extra_payload} to `endpoint_url` and parses
    the JSON response with RAGResponse.from_dict.

    Retries 408/5xx and connection errors with exponential backoff, and 429s
    against a separate, larger budget, honoring Retry-After headers (spec §14).
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
        max_rate_limit_retries: int = 20,
        max_retry_after: float = 60.0,
        backoff_base: float = 0.5,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ):
        self.endpoint_url = endpoint_url
        self.query_field = query_field
        self.extra_payload = extra_payload or {}
        self.max_retries = max_retries
        self.max_rate_limit_retries = max_rate_limit_retries
        self.max_retry_after = max_retry_after
        self.backoff_base = backoff_base
        self.n_rate_limited = 0
        # A 429 is about the client as a whole, not the one request that got it.
        # Every in-flight worker waits until this monotonic time before sending,
        # otherwise 50 workers keep hitting the limit one after another.
        self._cooldown_until = 0.0
        headers = {"Authorization": auth_header} if auth_header else {}
        self._client = httpx.AsyncClient(headers=headers, timeout=timeout, transport=transport)

    async def query(self, question: str) -> RAGResponse:
        payload = {self.query_field: question, **self.extra_payload}
        errors = rate_limits = 0
        while True:
            await self._wait_for_cooldown()
            try:
                resp = await self._client.post(self.endpoint_url, json=payload)
            except (httpx.TransportError, httpx.TimeoutException):
                if errors >= self.max_retries:
                    raise
                await asyncio.sleep(self._backoff(errors))
                errors += 1
                continue
            if resp.status_code == 429 and rate_limits < self.max_rate_limit_retries:
                self.n_rate_limited += 1
                delay = self._retry_after(resp)
                if delay is None:
                    delay = self._backoff(min(rate_limits, 6))
                self._cooldown_until = max(self._cooldown_until, time.monotonic() + delay)
                rate_limits += 1
                continue
            if resp.status_code in RETRY_STATUSES - {429} and errors < self.max_retries:
                await asyncio.sleep(self._retry_after(resp) or self._backoff(errors))
                errors += 1
                continue
            resp.raise_for_status()
            return RAGResponse.from_dict(resp.json(), question=question)

    async def _wait_for_cooldown(self) -> None:
        while (remaining := self._cooldown_until - time.monotonic()) > 0:
            # jitter so the whole pool doesn't fire in the same instant the window reopens
            await asyncio.sleep(remaining * (1 + random.random() * 0.1))

    def _backoff(self, attempt: int) -> float:
        return self.backoff_base * (2**attempt) * (1 + random.random() * 0.25)

    def _retry_after(self, resp: httpx.Response) -> Optional[float]:
        value = resp.headers.get("retry-after")
        if value is None:
            return None
        try:
            seconds = float(value)
        except ValueError:
            try:
                seconds = parsedate_to_datetime(value).timestamp() - time.time()
            except (TypeError, ValueError):
                return None
        return min(max(seconds, 0.0), self.max_retry_after)

    async def aclose(self) -> None:
        await self._client.aclose()
