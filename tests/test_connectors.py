import httpx
import pytest

from src.connectors import HTTPRAGConnector, RAGResponse
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
