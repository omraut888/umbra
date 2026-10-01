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


def backoff(attempt: int, base: float) -> float:
    return base * (2**attempt) * (1 + random.random() * 0.25)


class RateLimitGate:
    """The 429 state one rate-limited backend's workers share.

    A 429 is about the client as a whole, not the one request that got it.
    Every in-flight worker waits until the cooldown ends before sending,
    otherwise 50 workers keep hitting the limit one after another.
    """

    def __init__(self, max_retry_after: float = 60.0):
        self.max_retry_after = max_retry_after
        self.n_rate_limited = 0
        self._cooldown_until = 0.0  # time.monotonic()

    def hit(self, delay: float) -> None:
        self.n_rate_limited += 1
        self._cooldown_until = max(self._cooldown_until, time.monotonic() + delay)

    async def wait(self) -> None:
        while (remaining := self._cooldown_until - time.monotonic()) > 0:
            # jitter so the whole pool doesn't fire in the same instant the window reopens
            await asyncio.sleep(remaining * (1 + random.random() * 0.1))

    def retry_after(self, value: Any) -> Optional[float]:
        """Seconds from a Retry-After value (delta-seconds or HTTP-date), capped."""
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


class RetryingClient:
    """httpx.AsyncClient.post with the retry policy every Umbra connector uses.

    Retries 408/5xx and connection errors with exponential backoff, and 429s
    against a separate, larger budget, honoring Retry-After headers (spec §14).
    """

    def __init__(
        self,
        auth_header: Optional[str] = None,
        *,
        timeout: float = 60.0,
        max_retries: int = 4,
        max_rate_limit_retries: int = 20,
        max_retry_after: float = 60.0,
        backoff_base: float = 0.5,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ):
        self.max_retries = max_retries
        self.max_rate_limit_retries = max_rate_limit_retries
        self.backoff_base = backoff_base
        self.gate = RateLimitGate(max_retry_after)
        headers = {"Authorization": auth_header} if auth_header else {}
        self._client = httpx.AsyncClient(headers=headers, timeout=timeout, transport=transport)

    @property
    def n_rate_limited(self) -> int:
        return self.gate.n_rate_limited

    async def post_json(self, url: str, payload: Dict[str, Any]) -> httpx.Response:
        errors = rate_limits = 0
        while True:
            await self.gate.wait()
            try:
                resp = await self._client.post(url, json=payload)
            except (httpx.TransportError, httpx.TimeoutException):
                if errors >= self.max_retries:
                    raise
                await asyncio.sleep(backoff(errors, self.backoff_base))
                errors += 1
                continue
            if resp.status_code == 429 and rate_limits < self.max_rate_limit_retries:
                delay = self.gate.retry_after(resp.headers.get("retry-after"))
                self.gate.hit(backoff(min(rate_limits, 6), self.backoff_base) if delay is None else delay)
                rate_limits += 1
                continue
            if resp.status_code in RETRY_STATUSES - {429} and errors < self.max_retries:
                delay = self.gate.retry_after(resp.headers.get("retry-after"))
                await asyncio.sleep(backoff(errors, self.backoff_base) if delay is None else delay)
                errors += 1
                continue
            resp.raise_for_status()
            return resp

    async def aclose(self) -> None:
        await self._client.aclose()


class HTTPRAGConnector(RAGConnector):
    """POSTs {"query": question, **extra_payload} to `endpoint_url` and parses
    the JSON response with RAGResponse.from_dict. Keyword arguments beyond
    these go to RetryingClient.
    """

    def __init__(
        self,
        endpoint_url: str,
        auth_header: Optional[str] = None,
        *,
        query_field: str = "query",
        extra_payload: Optional[Dict[str, Any]] = None,
        **client_kwargs: Any,
    ):
        self.endpoint_url = endpoint_url
        self.query_field = query_field
        self.extra_payload = extra_payload or {}
        self.http = RetryingClient(auth_header, **client_kwargs)

    @property
    def n_rate_limited(self) -> int:
        return self.http.n_rate_limited

    async def query(self, question: str) -> RAGResponse:
        resp = await self.http.post_json(self.endpoint_url, {self.query_field: question, **self.extra_payload})
        return RAGResponse.from_dict(resp.json(), question=question)

    async def aclose(self) -> None:
        await self.http.aclose()
