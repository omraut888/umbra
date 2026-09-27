"""Document recommendations for coverage gaps (spec §7).

Two sources of candidates per cluster:

  kb_internal  passages already in the KB that match the gap but that retrieval
               isn't reaching, usually because they're buried in a chunk about
               something else (Strategy A)
  external     web search results for queries Haiku writes from the cluster's
               questions (Strategy B)

Every candidate is ranked by simulated gain: put its text into each probe's
retrieved set (replacing the lowest-ranked chunk), re-score the probes with
the same three signals, and take the change in the cluster's mean score. That
assumes the retriever would actually surface the new text, so it's an upper
bound, but it ranks internal and external candidates on one scale, and it's
the same quantity the report's estimated_improvement is built from.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Protocol, Sequence

import numpy as np

from src.audit import ProbeOutcome
from src.data.kb_loader import Chunk
from src.embeddings import embed
from src.probe_generation.taxonomy import AsyncComplete
from src.scoring.scorer import CoverageScorer

log = logging.getLogger(__name__)

# A KB passage is flagged as underused when it matches the cluster
# (MIN_SIMILARITY), matches it clearly better than the chunk it sits in
# (MIN_MARGIN, i.e. it's buried), is missing from most of the cluster's
# retrievals, and the cross-encoder says it actually answers at least one of
# the cluster's questions. On the synthetic KB the hydroponics passage sits at
# 0.517 vs 0.371 for its chunk, which comes back for 14% of the cluster's
# probes, and it answers 3 of 28. Similarity alone let through generic
# passages ("It starts with prevention: healthy soil...") that answer nothing.
MIN_SIMILARITY = 0.45
MIN_MARGIN = 0.10
MAX_RETRIEVAL_RATE = 0.5
WINDOW_SENTENCES = 2

SEARCH_TERMS_PROMPT = """This is a knowledge gap in a RAG system. Users are asking questions like:
{queries}

Generate {k} specific search queries that would find documents to fill this gap.
Focus on authoritative sources: official docs, academic papers, industry reports.
Return only the search queries, one per line."""


@dataclass
class Candidate:
    kind: str
    title: str
    snippet: str
    similarity: float
    url: Optional[str] = None
    doc_id: Optional[str] = None
    chunk_id: Optional[str] = None
    search_query: Optional[str] = None
    retrieval_rate: Optional[float] = None
    expected_gain: float = 0.0
    answered_probes: Optional[int] = None


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str


class WebSearchProvider(Protocol):
    async def search(self, query: str, n: int) -> List[SearchResult]: ...


def answered_count(queries: Sequence[str], text: str) -> int:
    from scipy.special import expit

    from src.embeddings import MODEL_LOCK, get_cross_encoder

    model = get_cross_encoder()
    with MODEL_LOCK:
        logits = model.predict([(q, text) for q in queries], show_progress_bar=False)
    return int((expit(logits) >= 0.5).sum())


def _norm(text: str) -> str:
    return " ".join(text.split()).lower()


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n else v


class KBInternalRecommender:
    def __init__(self, chunks: Sequence[Chunk]):
        self.chunks = list(chunks)
        self.windows: List[tuple] = []  # (chunk index, text)
        for ci, c in enumerate(self.chunks):
            sentences = [s for s in re.split(r"(?<=[.!?])\s+", c.text) if s.strip()]
            for i in range(max(1, len(sentences) - WINDOW_SENTENCES + 1)):
                self.windows.append((ci, " ".join(sentences[i:i + WINDOW_SENTENCES])))
        self.window_embs = embed([w[1] for w in self.windows])
        self.chunk_embs = embed([c.text for c in self.chunks])

    def candidates(self, probes: Sequence[ProbeOutcome], centroid: np.ndarray, limit: int = 3) -> List[Candidate]:
        centroid = _unit(centroid)
        win_sims = self.window_embs @ centroid
        chunk_sims = self.chunk_embs @ centroid
        contexts = [_norm(" ".join(o.response.chunk_texts)) for o in probes]
        queries = [o.probe.query for o in probes]
        best_per_chunk = {}
        for wi in np.argsort(-win_sims):
            sim = float(win_sims[wi])
            if sim < MIN_SIMILARITY:
                break
            ci, text = self.windows[wi]
            if ci in best_per_chunk or sim - float(chunk_sims[ci]) < MIN_MARGIN:
                continue
            rate = sum(_norm(text) in ctx for ctx in contexts) / len(contexts)
            if rate >= MAX_RETRIEVAL_RATE:
                continue
            answered = answered_count(queries, text)
            if answered == 0:
                continue
            chunk = self.chunks[ci]
            best_per_chunk[ci] = Candidate(
                kind="kb_internal", title=f"{chunk.doc_id} (chunk {chunk.chunk_id})", snippet=text, similarity=sim,
                doc_id=chunk.doc_id, chunk_id=chunk.chunk_id, retrieval_rate=rate, answered_probes=answered,
            )
            if len(best_per_chunk) == limit:
                break
        return list(best_per_chunk.values())


async def external_candidates(
    probes: Sequence[ProbeOutcome],
    representative: Sequence[str],
    centroid: np.ndarray,
    complete: AsyncComplete,
    provider: WebSearchProvider,
    n_queries: int = 5,
    per_query: int = 5,
    limit: int = 5,
) -> List[Candidate]:
    text = await complete(SEARCH_TERMS_PROMPT.format(queries="\n".join(f"- {q}" for q in representative), k=n_queries))
    queries = [q.strip().strip('"').lstrip("-*0123456789.) ").strip() for q in text.splitlines()
               if not q.lstrip().startswith("#")]
    queries = [q for q in queries if len(q) > 3][:n_queries]
    batches = await asyncio.gather(*(provider.search(q, per_query) for q in queries))
    seen, results = set(), []
    for q, batch in zip(queries, batches):
        for r in batch:
            if r.url not in seen:
                seen.add(r.url)
                results.append((q, r))
    if not results:
        return []
    embs = embed([f"{r.title}. {r.snippet}" for _, r in results])
    sims = embs @ _unit(centroid)
    order = np.argsort(-sims)[: limit * 2]  # keep a few spare for the gain ranking to choose from
    return [
        Candidate(kind="external", title=results[i][1].title, snippet=results[i][1].snippet, url=results[i][1].url,
                  similarity=float(sims[i]), search_query=results[i][0])
        for i in order
    ]


def simulate(scorer: CoverageScorer, probes: Sequence[ProbeOutcome], text: str) -> List[float]:
    """New coverage scores for `probes` if `text` replaced each one's lowest-ranked chunk."""
    chunk_lists = [(o.response.chunk_texts[:-1] if o.response.chunk_texts else []) + [text] for o in probes]
    scores, _ = scorer.score_batch([o.probe.query for o in probes], chunk_lists)
    return [s.score for s in scores]


def rank_by_gain(scorer: CoverageScorer, probes: Sequence[ProbeOutcome], candidates: List[Candidate],
                 limit: int = 5) -> List[Candidate]:
    before = float(np.mean([o.score.score for o in probes]))
    for c in candidates:
        c.expected_gain = float(np.mean(simulate(scorer, probes, c.snippet))) - before
    # a candidate that doesn't raise the simulated score isn't worth recommending
    kept = sorted((c for c in candidates if c.expected_gain > 0), key=lambda c: (-c.expected_gain, -c.similarity))
    return kept[:limit]
