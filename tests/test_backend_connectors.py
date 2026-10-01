"""QdrantRAGConnector and LangChainRAGConnector against real backends built
over the synthetic KB: an in-process Qdrant (and a server one when QDRANT_URL
is set), and LangChain chains over an InMemoryVectorStore."""

import os
import time
from operator import itemgetter

import httpx
import numpy as np
import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.fake import FakeListLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore
from qdrant_client import AsyncQdrantClient

from src.connectors import HTTPRAGConnector, LangChainRAGConnector, QdrantRAGConnector
from src.connectors.mock_rag_server import create_app
from src.connectors.qdrant import index_chunks
from src.data import synthetic_kb_builder
from src.data.kb_loader import load_chunks
from src.embeddings import embed

COMPOST_Q = "What carbon to nitrogen ratio should a compost pile have?"
QUESTIONS = [COMPOST_Q, "How do I prune suckers on tomatoes?", "What is the Kratky method?",
             "How are capital gains on Bitcoin taxed?"]


@pytest.fixture(scope="module")
def chunks(tmp_path_factory):
    out = tmp_path_factory.mktemp("kb")
    synthetic_kb_builder.build(out)
    return out, load_chunks(out)


def _fixed_embed(texts):
    return np.ones((len(texts), 384), dtype=np.float32)


async def _memory_qdrant(chunks, collection="kb"):
    client = AsyncQdrantClient(location=":memory:")
    await index_chunks(client, collection, chunks)
    return client


class TestQdrant:
    async def test_retrieves_like_the_mock_server(self, chunks):
        kb_path, kb_chunks = chunks
        async with QdrantRAGConnector(client=await _memory_qdrant(kb_chunks), collection="kb") as conn:
            got = await conn.query_many(QUESTIONS)
        # same embedder + cosine over the same chunks as the mock server's brute-force index
        async with HTTPRAGConnector("http://mock/query", transport=httpx.ASGITransport(app=create_app(kb_path))) as ref:
            want = await ref.query_many(QUESTIONS)
        for g, w in zip(got, want):
            assert g.error is None and g.answer == ""
            assert g.chunk_ids == w.chunk_ids
            assert [c.score for c in g.chunks] == pytest.approx([c.score for c in w.chunks], abs=1e-4)
        assert got[0].chunks[0].metadata["doc_id"].startswith("compost-")

    async def test_top_k_and_single_query(self, chunks):
        async with QdrantRAGConnector(client=await _memory_qdrant(chunks[1]), collection="kb", top_k=2) as conn:
            resp = await conn.query("How do I prune suckers on tomatoes?")
        assert len(resp.chunks) == 2 and all(c.metadata["doc_id"].startswith("tomato-") for c in resp.chunks)

    async def test_reindexing_overwrites(self, chunks):
        client = await _memory_qdrant(chunks[1])
        await index_chunks(client, "kb", chunks[1])
        assert (await client.count("kb")).count == len(chunks[1])

    async def test_langchain_qdrant_payloads_and_named_vectors(self, chunks):
        from qdrant_client import models

        # langchain-qdrant writes {"page_content", "metadata"}; other indexers use their own text key
        client = AsyncQdrantClient(location=":memory:")
        await client.create_collection("lc", vectors_config={"dense": models.VectorParams(
            size=384, distance=models.Distance.COSINE)})
        sample = chunks[1][:20]
        vecs = embed([c.text for c in sample])
        await client.upsert("lc", points=[
            models.PointStruct(id=i, vector={"dense": v.tolist()},
                               payload={"page_content": c.text, "metadata": {"doc_id": c.doc_id}})
            for i, (c, v) in enumerate(zip(sample, vecs))
        ])
        async with QdrantRAGConnector(client=client, collection="lc", vector_name="dense", top_k=3) as conn:
            resp = await conn.query(sample[4].text)
        assert resp.chunks[0].text == sample[4].text and resp.chunks[0].chunk_id == "4"
        assert resp.chunks[0].metadata == {"doc_id": sample[4].doc_id}

    async def test_custom_text_key(self, chunks):
        from qdrant_client import models

        client = AsyncQdrantClient(location=":memory:")
        await client.create_collection("b", vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE))
        c = chunks[1][0]
        await client.upsert("b", points=[models.PointStruct(id=1, vector=embed([c.text])[0].tolist(),
                                                            payload={"body": c.text, "source": "x"})])
        async with QdrantRAGConnector(client=client, collection="b", text_key="body") as conn:
            resp = await conn.query("anything")
        assert resp.chunk_texts == [c.text] and resp.chunks[0].metadata == {"source": "x"}

    async def test_llm_endpoint_generates_the_answer_with_retries(self, chunks):
        seen = []

        def llm(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            if len(seen) == 1:
                return httpx.Response(429, headers={"Retry-After": "0"})
            assert b'"contexts"' in request.content
            return httpx.Response(200, json={"answer": "use 30:1"})

        async with QdrantRAGConnector(client=await _memory_qdrant(chunks[1]), collection="kb",
                                      llm_endpoint="http://llm/generate",
                                      transport=httpx.MockTransport(llm)) as conn:
            resp = await conn.query(COMPOST_Q)
        assert resp.answer == "use 30:1" and len(seen) == 2 and conn.llm.n_rate_limited == 1

    async def test_rate_limited_rest_client_waits_out_retry_after(self):
        # a real REST AsyncQdrantClient; middleware stands in for Qdrant Cloud answering 429s
        sent = []
        t0 = time.monotonic()
        point = {"id": 7, "version": 0, "score": 0.9, "payload": {"chunk_id": "c7", "doc_id": "d", "text": "hi"}}

        async def cloud(request, call_next):
            sent.append(time.monotonic() - t0)
            if len(sent) == 1:
                return httpx.Response(429, headers={"Retry-After": "1"}, json={"status": {"error": "slow down"}})
            if len(sent) == 2:
                return httpx.Response(429, json={"status": {"error": "slow down"}})  # no Retry-After
            return httpx.Response(200, json={"result": {"points": [point]}, "status": "ok", "time": 0.0})

        client = AsyncQdrantClient(url="http://qdrant.test:6333", check_compatibility=False)
        client.http.client.add_middleware(cloud)
        async with QdrantRAGConnector(client=client, collection="kb", embed_fn=_fixed_embed,
                                      backoff_base=0.05) as conn:
            [resp] = await conn.query_many(["q"])
        assert resp.error is None and resp.chunk_ids == ["c7"] and resp.chunk_texts == ["hi"]
        assert conn.n_rate_limited == 2 and len(sent) == 3
        assert sent[1] >= 1.0  # honored Retry-After
        assert sent[2] - sent[1] >= 0.05  # backoff when there wasn't one

    async def test_rate_limit_exhaustion_is_a_per_probe_error(self):
        async def always_limited(request, call_next):
            return httpx.Response(429, headers={"Retry-After": "0"}, json={"status": {"error": "quota"}})

        client = AsyncQdrantClient(url="http://qdrant.test:6333", check_compatibility=False)
        client.http.client.add_middleware(always_limited)
        async with QdrantRAGConnector(client=client, collection="kb", embed_fn=_fixed_embed,
                                      max_rate_limit_retries=2) as conn:
            [resp] = await conn.query_many(["q"])
        assert "ResourceExhausted" in resp.error and conn.n_rate_limited == 2

    async def test_missing_collection_is_a_per_probe_error(self, chunks):
        async with QdrantRAGConnector(client=AsyncQdrantClient(location=":memory:"), collection="nope") as conn:
            [resp] = await conn.query_many(["q"])
        assert resp.error and resp.chunks == []

    def test_needs_url_or_client(self):
        with pytest.raises(ValueError):
            QdrantRAGConnector(collection="kb")


@pytest.mark.qdrant
@pytest.mark.skipif(not os.environ.get("QDRANT_URL"), reason="QDRANT_URL not set")
async def test_qdrant_server(chunks):
    url, collection = os.environ["QDRANT_URL"], "umbra_test_kb"
    client = AsyncQdrantClient(url=url)
    try:
        await index_chunks(client, collection, chunks[1], recreate=True)
        async with QdrantRAGConnector(url, collection) as conn:
            out = await conn.query_many(QUESTIONS * 10, concurrency=20)
        assert all(r.error is None and len(r.chunks) == 5 for r in out)
        assert out[0].chunks[0].metadata["doc_id"].startswith("compost-")
    finally:
        await client.delete_collection(collection)
        await client.close()


class MiniLM(Embeddings):
    def embed_documents(self, texts):
        return embed(texts).tolist()

    def embed_query(self, text):
        return embed([text])[0].tolist()


@pytest.fixture(scope="module")
def retriever(chunks):
    store = InMemoryVectorStore(MiniLM())
    store.add_documents([Document(page_content=c.text, id=c.chunk_id, metadata={"doc_id": c.doc_id})
                         for c in chunks[1]])
    return store.as_retriever(search_kwargs={"k": 4})


def retrieval_qa(retriever):
    """Same input/output contract as RetrievalQA(return_source_documents=True),
    which langchain 1.x only ships in langchain-classic."""
    prompt = PromptTemplate.from_template("Context:\n{context}\n\nQuestion: {query}\nAnswer:")
    llm = FakeListLLM(responses=["Aim for about 30 parts carbon to 1 part nitrogen."])
    answer = (RunnableLambda(lambda x: {"context": "\n\n".join(d.page_content for d in x["source_documents"]),
                                        "query": x["query"]})
              | prompt | llm | StrOutputParser())
    return RunnablePassthrough.assign(source_documents=itemgetter("query") | retriever).assign(result=answer)


class TestLangChain:
    async def test_retrieval_qa_shape(self, retriever):
        async with LangChainRAGConnector(retrieval_qa(retriever)) as conn:
            out = await conn.query_many(QUESTIONS, concurrency=4)
        compost = out[0]
        assert compost.error is None and compost.answer.startswith("Aim for about 30")
        assert len(compost.chunks) == 4 and compost.chunks[0].metadata["doc_id"].startswith("compost-")
        assert compost.chunks[0].chunk_id.startswith("compost-")  # Document.id carried through
        assert [r.question for r in out] == QUESTIONS

    async def test_create_retrieval_chain_shape(self, retriever):
        chain = RunnablePassthrough.assign(context=itemgetter("input") | retriever).assign(
            answer=RunnableLambda(lambda x: f"{len(x['context'])} docs"))
        async with LangChainRAGConnector(chain, input_key="input") as conn:
            resp = await conn.query("How do I prune suckers on tomatoes?")
        assert resp.answer == "4 docs" and resp.chunks[0].metadata["doc_id"].startswith("tomato-")

    async def test_bare_retriever(self, retriever):
        async with LangChainRAGConnector(retriever) as conn:
            resp = await conn.query(COMPOST_Q)
        assert resp.answer == "" and len(resp.chunks) == 4

    async def test_legacy_chain_input_keys(self):
        class LegacyChain:
            input_keys = ["question"]

            async def ainvoke(self, inputs):
                assert set(inputs) == {"question"}
                return {"answer": "a", "source_documents": [{"page_content": "x", "metadata": {"source": "s"}}]}

        async with LangChainRAGConnector(LegacyChain()) as conn:
            resp = await conn.query("q")
        assert resp.chunk_texts == ["x"] and resp.answer == "a" and resp.chunks[0].metadata == {"source": "s"}

    async def test_no_source_documents_is_a_per_probe_error(self, retriever):
        chain = RunnableLambda(lambda x: {"result": "an answer with no sources"})
        async with LangChainRAGConnector(chain) as conn:
            [resp] = await conn.query_many(["q"])
        assert "return_source_documents" in resp.error
