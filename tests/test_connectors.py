import asyncio
import time
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime

import httpx
import pytest

from src.connectors import HTTPRAGConnector, RAGResponse
from src.connectors.http import RetryingClient
from src.connectors.mock_rag_server import create_app
from src.data import synthetic_kb_builder


@pytest.fixture(scope="module")
def kb_path(tmp_path_factory):
    out = tmp_path_factory.mktemp("kb")
    synthetic_kb_builder.build(out)
    return out


@pytest.fixture(scope="module")
def app(kb_path):
    return create_app(kb_path)


def connector_for(app, **kwargs) -> HTTPRAGConnector:
    return HTTPRAGConnector("http://mock/query", transport=httpx.ASGITransport(app=app), **kwargs)


class TestRAGResponseParsing:
    def test_mock_server_shape(self):
        r = RAGResponse.from_dict({
            "question": "q", "answer": "a",
            "chunks": [{"chunk_id": "d#0", "doc_id": "d", "text": "hello", "score": 0.7}],
        })
        assert r.chunk_texts == ["hello"] and r.chunk_ids == ["d#0"]
        assert r.chunks[0].score == 0.7 and r.chunks[0].metadata == {"doc_id": "d"}

    def test_plain_string_chunks_and_alternate_keys(self):
        r = RAGResponse.from_dict({"contexts": ["a", "b"], "result": "ans"}, question="q")
        assert r.chunk_texts == ["a", "b"] and r.answer == "ans" and r.chunk_ids == ["0", "1"]

    def test_langchain_style_documents(self):
        r = RAGResponse.from_dict({
            "source_documents": [{"page_content": "x", "metadata": {"source": "s"}}],
            "result": "y",
        }, question="q")
        assert r.chunk_texts == ["x"] and r.chunks[0].metadata == {"source": "s"}

    @pytest.mark.parametrize("payload", [{"answer": "no chunks"}, {"chunks": [{"no_text": 1}]}, {"chunks": [3]}])
    def test_malformed_raises(self, payload):
        with pytest.raises(ValueError):
            RAGResponse.from_dict(payload, question="q")


async def test_health(app):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://mock") as c:
        body = (await c.get("/health")).json()
    assert body["status"] == "ok" and body["documents"] == 39 and body["chunks"] > 39


async def test_query_returns_relevant_chunks(app):
    async with connector_for(app) as conn:
        resp = await conn.query("What carbon to nitrogen ratio should a compost pile have?")
    assert resp.error is None
    assert len(resp.chunks) == 5
    assert resp.chunks[0].metadata["doc_id"].startswith("compost-")
    scores = [c.score for c in resp.chunks]
    assert scores == sorted(scores, reverse=True)
    assert resp.answer.startswith("Based on the available context:")


async def test_extra_payload_top_k(app):
    async with connector_for(app, extra_payload={"top_k": 2}) as conn:
        resp = await conn.query("How do I prune suckers on tomatoes?")
    assert len(resp.chunks) == 2
    assert all(c.metadata["doc_id"].startswith("tomato-") for c in resp.chunks)


async def test_thin_topic_retrieves_its_host_document(app):
    async with connector_for(app, extra_payload={"top_k": 3}) as conn:
        resp = await conn.query("What is the Kratky method?")
    assert "seeds-01-indoor" in [c.metadata["doc_id"] for c in resp.chunks]


async def test_absent_topic_still_answers_confidently(app):
    """The failure mode Umbra exists to detect: no relevant content, fluent answer anyway."""
    async with connector_for(app) as conn:
        resp = await conn.query("How are capital gains on Bitcoin taxed?")
    assert len(resp.chunks) == 5 and max(c.score for c in resp.chunks) < 0.3
    assert resp.answer


async def test_query_many_preserves_order(app):
    qs = ["How do I start tomato seeds?", "What goes in a worm bin?", "How deep should raised beds be?"]
    async with connector_for(app) as conn:
        out = await conn.query_many(qs, concurrency=2)
    assert [r.question for r in out] == qs


async def test_retries_then_succeeds_honoring_retry_after():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(429, headers={"Retry-After": "0"})
        if len(calls) == 2:
            return httpx.Response(503)
        return httpx.Response(200, json={"chunks": ["c"], "answer": "a"})

    conn = HTTPRAGConnector("http://x/query", transport=httpx.MockTransport(handler), backoff_base=0.001)
    resp = await conn.query("q")
    await conn.aclose()
    assert len(calls) == 3 and resp.chunk_texts == ["c"]


async def test_non_retryable_error_is_captured_by_query_many():
    conn = HTTPRAGConnector("http://x/query", transport=httpx.MockTransport(lambda r: httpx.Response(400)))
    [resp] = await conn.query_many(["q"])
    await conn.aclose()
    assert resp.error and "400" in resp.error and resp.chunks == []


async def test_auth_header_and_query_field_are_sent():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = request.read()
        return httpx.Response(200, json={"chunks": [], "answer": ""})

    conn = HTTPRAGConnector("http://x/q", auth_header="Bearer t", query_field="question",
                            transport=httpx.MockTransport(handler))
    await conn.query("hi")
    await conn.aclose()
    assert seen["auth"] == "Bearer t" and b'"question":"hi"' in seen["body"].replace(b" ", b"")


QUESTIONS = [
    "How do I start tomato seeds?", "What goes in a worm bin?", "How deep should raised beds be?",
    "What carbon to nitrogen ratio should a compost pile have?", "How do I prune suckers on tomatoes?",
    "What is the Kratky method?",
]


async def test_query_many_caps_in_flight_requests(kb_path):
    slow = create_app(kb_path, latency=0.1)
    qs = QUESTIONS * 20
    async with connector_for(slow) as conn:
        start = time.monotonic()
        out = await conn.query_many(qs, concurrency=50)
        elapsed = time.monotonic() - start
    assert all(r.error is None for r in out) and [r.question for r in out] == qs
    assert slow.state.traffic.max_in_flight == 50
    assert elapsed < len(qs) * 0.1 / 2  # well under serial latency; embedding is the rest

    async with connector_for(slow) as conn:
        await conn.query_many(QUESTIONS * 3, concurrency=4)
    assert slow.state.traffic.max_in_flight == 50  # unchanged by the smaller run
    slow.state.traffic.max_in_flight = 0
    async with connector_for(slow) as conn:
        await conn.query_many(QUESTIONS * 3, concurrency=4)
    assert slow.state.traffic.max_in_flight == 4


async def test_query_many_against_rate_limited_server(kb_path):
    limited = create_app(kb_path, latency=0.02, rate_limit=30, rate_window=0.25)
    qs = QUESTIONS * 25
    async with connector_for(limited, backoff_base=0.01) as conn:
        start = time.monotonic()
        out = await conn.query_many(qs, concurrency=50)
        elapsed = time.monotonic() - start
    traffic = limited.state.traffic
    assert all(r.error is None for r in out), [r.error for r in out if r.error][:3]
    assert [r.question for r in out] == qs and all(len(r.chunks) == 5 for r in out)
    assert traffic.rejected > 0 and conn.n_rate_limited == traffic.rejected
    assert traffic.served == len(qs) and traffic.max_served_per_window <= 30
    assert elapsed >= (len(qs) / 30 - 1) * 0.25


async def test_rate_limit_pauses_other_workers():
    sent = []
    t0 = time.monotonic()

    async def handler(request: httpx.Request) -> httpx.Response:
        sent.append(time.monotonic() - t0)
        if len(sent) == 1:
            await asyncio.sleep(0.02)  # let the second request get in flight first
            return httpx.Response(429, headers={"Retry-After": "0.3"})
        await asyncio.sleep(0.05)
        return httpx.Response(200, json={"chunks": ["c"], "answer": "a"})

    conn = HTTPRAGConnector("http://x/query", transport=httpx.MockTransport(handler))
    out = await conn.query_many(["a", "b", "c", "d"], concurrency=2)
    await conn.aclose()
    assert all(r.error is None for r in out)
    # sent[1] was already in flight when the 429 came back; everything after it waited
    assert len(sent) == 5 and sent[1] < 0.1 and min(sent[2:]) >= 0.3


async def test_rate_limit_budget_is_separate_and_exhaustion_is_recorded():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(429, headers={"Retry-After": "0"})

    conn = HTTPRAGConnector("http://x/query", transport=httpx.MockTransport(handler),
                            max_retries=0, max_rate_limit_retries=3)
    [resp] = await conn.query_many(["q"])
    await conn.aclose()
    assert len(calls) == 4 and "429" in resp.error


@pytest.mark.parametrize("header, expected", [
    ("7", 7.0), ("1.5", 1.5), ("-3", 0.0), ("600", 60.0), ("soon", None), (30, 30.0), (-30, 0.0),
])
def test_retry_after_parsing(header, expected):
    if isinstance(header, int):  # HTTP-date form, `header` seconds from now
        header = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=header), usegmt=True)
    got = RetryingClient()._retry_after(httpx.Response(429, headers={"Retry-After": header}))
    assert got is None if expected is None else got == pytest.approx(expected, abs=1.5)
