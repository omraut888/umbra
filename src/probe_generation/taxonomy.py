"""Strategy 1: taxonomy-guided probe generation (spec §3).

1. Extract a topic taxonomy from the KB chunks with BERTopic.
2. Ask Claude Haiku for factual questions per topic, varying question type
   (factual lookup, comparison, causal, procedural, definitional).
3. Deduplicate near-identical probes (cosine >= 0.95, spec §14).

Because the taxonomy is derived from the KB, this strategy only probes topics
that exist in the KB. Topics with zero documents are the job of the
adversarial/boundary strategy (Phase 2).
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from src.data.kb_loader import Chunk
from src.embeddings import embed, get_embedder

log = logging.getLogger(__name__)

CLAUDE_MODEL = "claude-haiku-4-5"  # spec §3: claude_haiku
DEDUP_THRESHOLD = 0.95
MAX_TOPICS = 50  # spec §3 nr_topics=50; §14: enforce max 50 named clusters
QUESTION_TYPES = ("factual lookup", "comparison", "causal", "procedural", "definitional")


@dataclass
class Topic:
    topic_id: int
    name: str  # BERTopic name, e.g. "2_pile_browns_nitrogen_compost"
    keywords: List[str]
    size: int
    chunk_ids: List[str]
    doc_ids: List[str]
    representative_texts: List[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return ", ".join(self.keywords[:6])

    def as_dict(self) -> dict:
        return {
            "topic_id": self.topic_id,
            "name": self.name,
            "keywords": self.keywords,
            "size": self.size,
            "chunk_ids": self.chunk_ids,
            "doc_ids": sorted(set(self.doc_ids)),
        }


@dataclass
class Probe:
    query: str
    strategy: str = "taxonomy"
    topic: Optional[str] = None


def extract_taxonomy(
    chunks: Sequence[Chunk],
    min_topic_size: Optional[int] = None,
    max_topics: int = MAX_TOPICS,
    random_state: int = 42,
) -> List[Topic]:
    """Fit BERTopic over KB chunks and return non-outlier topics.

    UMAP/HDBSCAN are configured explicitly because BERTopic's defaults
    (n_neighbors=15, min_cluster_size=10) are tuned for thousands of documents;
    on a ~100-chunk KB they collapse everything into one or two topics.
    Outlier chunks (topic -1) are reassigned to their nearest topic by
    embedding similarity so that no KB content goes unprobed.
    """
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    texts = [c.text for c in chunks]
    n = len(texts)
    if n < 10:
        raise ValueError(f"need at least 10 chunks to extract a taxonomy, got {n}")
    if min_topic_size is None:
        # ~2% of the KB, at least 3 chunks.
        min_topic_size = max(3, round(0.02 * n))

    embeddings = embed(texts)
    model = BERTopic(
        embedding_model=get_embedder(),
        umap_model=UMAP(
            n_neighbors=min(15, max(2, n // 10)),
            n_components=5,
            min_dist=0.0,
            metric="cosine",
            init="spectral",
            random_state=random_state,
        ),
        hdbscan_model=HDBSCAN(
            min_cluster_size=min_topic_size,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
        ),
        vectorizer_model=CountVectorizer(stop_words="english", ngram_range=(1, 2)),
        nr_topics=None,
    )
    topics, _ = model.fit_transform(texts, embeddings)

    if len(set(topics) - {-1}) > max_topics:
        model.reduce_topics(texts, nr_topics=max_topics)
        topics = model.topics_

    if -1 in topics and len(set(topics)) > 1:
        topics = model.reduce_outliers(texts, topics, strategy="embeddings", embeddings=embeddings)
        model.update_topics(texts, topics=topics, vectorizer_model=CountVectorizer(stop_words="english", ngram_range=(1, 2)))

    info = model.get_topic_info().set_index("Topic")
    result: List[Topic] = []
    for topic_id in sorted(set(topics) - {-1}):
        members = [i for i, t in enumerate(topics) if t == topic_id]
        keywords = [w for w, _ in (model.get_topic(topic_id) or [])][:10]
        # Representative texts: members closest to the topic centroid.
        member_embs = embeddings[members]
        centroid = member_embs.mean(axis=0)
        order = np.argsort(-(member_embs @ centroid))
        result.append(
            Topic(
                topic_id=int(topic_id),
                name=str(info.loc[topic_id, "Name"]),
                keywords=keywords,
                size=len(members),
                chunk_ids=[chunks[i].chunk_id for i in members],
                doc_ids=[chunks[i].doc_id for i in members],
                representative_texts=[texts[members[j]] for j in order[:3]],
            )
        )
    log.info("extracted %d topics from %d chunks", len(result), n)
    return result


PROMPT_TEMPLATE = """Generate {k} factual questions that test knowledge of this topic: {label}

Example passages from the knowledge base on this topic:
{examples}

These questions should be answerable from a knowledge base on this topic, but must not be copied from the passages; ask what a real user of this knowledge base would ask.
Vary the question types: factual lookup, comparison, causal, procedural, definitional.
{avoid}
Return only the questions, one per line. No numbering, no explanation."""

_LIST_PREFIX = re.compile(r"^\s*(?:[-*•]|\d+[.)]|Q\d*[:.])\s*")


def parse_questions(text: str) -> List[str]:
    out = []
    for line in text.splitlines():
        q = _LIST_PREFIX.sub("", line).strip().strip('"').strip()
        if len(q) >= 10 and q.endswith("?"):
            out.append(q)
    return out


def build_prompt(topic: Topic, k: int, already: Sequence[str]) -> str:
    examples = "\n".join(f"- {t[:400]}" for t in topic.representative_texts)
    avoid = ""
    if already:
        recent = "\n".join(f"- {q}" for q in list(already)[-30:])
        avoid = f"\nDo not repeat or paraphrase any of these existing questions:\n{recent}\n"
    return PROMPT_TEMPLATE.format(k=k, label=topic.label, examples=examples, avoid=avoid)


AsyncComplete = Callable[[str], "asyncio.Future[str]"]


def anthropic_completer(model: str = CLAUDE_MODEL, max_tokens: int = 1024) -> AsyncComplete:
    import anthropic

    client = anthropic.AsyncAnthropic(max_retries=5)

    async def complete(prompt: str) -> str:
        response = await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in response.content if b.type == "text")

    return complete


async def generate_probes_for_topic(
    topic: Topic,
    n: int,
    complete: AsyncComplete,
    questions_per_call: int = 5,
    max_calls: Optional[int] = None,
    existing: Sequence[str] = (),
) -> List[Probe]:
    # Repeating the same prompt at temperature 1 gives lots of repeats, so each
    # call is shown the latest questions and told not to reuse them.
    max_calls = max_calls or math.ceil(n / questions_per_call) * 3
    prior = set(existing)
    seen: Dict[str, None] = {}
    for _ in range(max_calls):
        if len(seen) >= n:
            break
        k = min(questions_per_call, n - len(seen))
        text = await complete(build_prompt(topic, k, [*existing, *seen]))
        for q in parse_questions(text):
            if q not in prior:
                seen.setdefault(q, None)
    return [Probe(query=q, topic=topic.name) for q in list(seen)[:n]]


def allocate(n_total: int, topics: Sequence[Topic]) -> List[int]:
    # spec §3: N/topics per topic, not proportional to topic size
    base, extra = divmod(n_total, len(topics))
    return [base + (1 if i < extra else 0) for i in range(len(topics))]


def dedup_probes(probes: Sequence[Probe], threshold: float = DEDUP_THRESHOLD) -> List[Probe]:
    if not probes:
        return []
    embs = embed([p.query for p in probes])
    kept: List[int] = []
    for i in range(len(probes)):
        if kept and float(np.max(embs[kept] @ embs[i])) >= threshold:
            continue
        kept.append(i)
    return [probes[i] for i in kept]


async def taxonomy_guided_generation(
    topics: Sequence[Topic],
    n: int,
    complete: Optional[AsyncComplete] = None,
    concurrency: int = 8,
    questions_per_call: int = 5,
    oversample: float = 1.15,
    max_rounds: int = 3,
) -> List[Probe]:
    # Dedup runs over the whole set after each round, so a topic can lose probes
    # to near-duplicates; later rounds top those topics back up.
    if not topics:
        raise ValueError("no topics to generate probes for")
    complete = complete or anthropic_completer()
    sem = asyncio.Semaphore(concurrency)
    targets = {t.name: k for t, k in zip(topics, allocate(n, topics))}
    kept: List[Probe] = []

    async def run(topic: Topic, count: int, existing: List[str]) -> List[Probe]:
        async with sem:
            return await generate_probes_for_topic(
                topic, count, complete, questions_per_call, existing=existing
            )

    for round_no in range(max_rounds):
        counts = Counter(p.topic for p in kept)
        deficits = {t.name: targets[t.name] - counts[t.name] for t in topics if counts[t.name] < targets[t.name]}
        if not deficits:
            break
        factor = oversample if round_no == 0 else 1.0
        jobs = [
            run(t, math.ceil(deficits[t.name] * factor), [p.query for p in kept if p.topic == t.name])
            for t in topics if t.name in deficits
        ]
        fresh = [p for batch in await asyncio.gather(*jobs) for p in batch]
        kept = dedup_probes(kept + fresh)  # kept probes come first, so they survive dedup

    by_topic: Dict[str, List[Probe]] = {}
    for p in kept:
        by_topic.setdefault(p.topic, []).append(p)
    final = [p for t in topics for p in by_topic.get(t.name, [])[: targets[t.name]]]

    counts = Counter(p.topic for p in final)
    short = {name: (counts[name], k) for name, k in targets.items() if counts[name] < k}
    if short:
        log.warning("topics short of their probe allocation after %d rounds: %s", max_rounds, short)
    return final
