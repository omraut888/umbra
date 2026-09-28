"""Wraps a LangChain chain or retriever (spec §11).

Accepts the output shapes LangChain RAG setups actually produce:
    RetrievalQA / legacy chains   {"query"} -> {"result", "source_documents"}
    create_retrieval_chain        {"input"} -> {"answer", "context"}
    a bare retriever              str -> [Document, ...]  (no answer)
Documents can be langchain_core Documents or plain dicts/strings.
"""

from __future__ import annotations

from typing import Any, List, Optional

from src.connectors.base import _ANSWER_KEYS, RAGConnector, RAGResponse, RetrievedChunk

_DOC_LIST_KEYS = ("source_documents", "context", "documents", "docs")


class LangChainRAGConnector(RAGConnector):
    def __init__(self, chain: Any, input_key: Optional[str] = None):
        from langchain_core.retrievers import BaseRetriever

        self.chain = chain
        self.is_retriever = isinstance(chain, BaseRetriever)
        self.input_key = input_key or _guess_input_key(chain)

    async def query(self, question: str) -> RAGResponse:
        if self.is_retriever:
            return RAGResponse(question=question, chunks=_to_chunks(await self.chain.ainvoke(question)))
        result = await self.chain.ainvoke({self.input_key: question})
        docs = next((result[k] for k in _DOC_LIST_KEYS if isinstance(result.get(k), list)), None)
        if docs is None:
            raise ValueError(f"chain output has no document list (tried {_DOC_LIST_KEYS}): {list(result)}"
                             " - RetrievalQA needs return_source_documents=True")
        answer = next((result[k] for k in _ANSWER_KEYS if isinstance(result.get(k), str)), "")
        return RAGResponse(question=question, chunks=_to_chunks(docs), answer=answer)


def _guess_input_key(chain: Any) -> str:
    keys = getattr(chain, "input_keys", None)  # legacy Chain objects declare theirs
    if keys and len(keys) == 1:
        return keys[0]
    return "query"


def _to_chunks(docs: List[Any]) -> List[RetrievedChunk]:
    chunks = []
    for i, d in enumerate(docs):
        if hasattr(d, "page_content"):
            meta = dict(d.metadata or {})
            chunk_id = getattr(d, "id", None) or meta.get("chunk_id") or meta.get("id")
            score = meta.get("score", meta.get("relevance_score"))
            chunks.append(RetrievedChunk(text=d.page_content, chunk_id=str(chunk_id) if chunk_id else str(i),
                                         score=float(score) if score is not None else None, metadata=meta))
        else:
            chunks.append(RetrievedChunk.from_obj(d, i))
    return chunks
