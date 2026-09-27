"""Strategy 4: KB-blind probes, generated from topic descriptions only.

The other three strategies all start from KB content (its topics, its
neighborhood, its chunks), so they can't ask about a topic the KB never
mentions. This one never sees the KB. It takes what the KB is *supposed* to
cover, either as an explicit topic list or as a one-line domain description
that the model expands into topics, and asks questions about that.

It only finds gaps inside the scope it's given. A gardening domain description
will never produce a question about crypto taxes; that kind of gap needs real
user queries.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

from src.probe_generation.adversarial import parse_topics
from src.probe_generation.taxonomy import AsyncComplete, Probe, parse_questions

QUESTIONS_PROMPT = """Write {k} different questions that a user might ask a knowledge base about this topic:

{description}

Mix simple questions with detailed, specific ones, and vary the type (factual lookup, comparison, causal, procedural, definitional).
{avoid}
Return only the questions, one per line. No numbering, no explanation."""

DOMAIN_TOPICS_PROMPT = """A knowledge base is meant to cover this domain:

{domain}

List {k} distinct topics its users would expect it to answer questions about. Give each as a short name, a colon, then a one-sentence description.
Return only the list, one topic per line."""


@dataclass(frozen=True)
class TopicSpec:
    name: str
    description: str


def load_topics_file(path: str | Path) -> List[TopicSpec]:
    """Accepts a list of {"name", "description"}, a {name: description} dict, or
    a ground_truth.json from the synthetic KB builder."""
    data = json.loads(Path(path).read_text())
    if isinstance(data, list):
        return [TopicSpec(d["name"], d["description"]) for d in data]
    if "topics" in data:  # synthetic KB ground truth
        merged = {**data["topics"], **data.get("background_topics", {})}
        return [TopicSpec(name, t["description"]) for name, t in merged.items()]
    return [TopicSpec(name, desc) for name, desc in data.items()]


async def enumerate_domain_topics(domain: str, complete: AsyncComplete, n_topics: int = 15) -> List[TopicSpec]:
    text = await complete(DOMAIN_TOPICS_PROMPT.format(domain=domain, k=n_topics))
    specs = []
    for line in text.splitlines():
        if ":" not in line:
            continue
        name, desc = line.split(":", 1)
        name = (parse_topics(name) or [""])[0]
        if name and desc.strip():
            specs.append(TopicSpec(name, f"{name}: {desc.strip()}"))
    return specs[:n_topics]


async def _questions_for(topic: TopicSpec, n: int, complete: AsyncComplete) -> List[Probe]:
    seen: Dict[str, None] = {}
    for _ in range(max(3, n)):  # cap on calls in case the model keeps repeating itself
        if len(seen) >= n:
            break
        avoid = ""
        if seen:
            avoid = "\nDo not repeat any of these:\n" + "\n".join(f"- {q}" for q in list(seen)[-30:]) + "\n"
        text = await complete(QUESTIONS_PROMPT.format(k=min(10, n - len(seen)), description=topic.description, avoid=avoid))
        for q in parse_questions(text):
            seen.setdefault(q, None)
    return [Probe(query=q, strategy="kb_blind", topic=topic.name) for q in list(seen)[:n]]


async def kb_blind_generation(
    topics: Sequence[TopicSpec],
    per_topic: int,
    complete: AsyncComplete,
    concurrency: int = 8,
) -> List[Probe]:
    sem = asyncio.Semaphore(concurrency)

    async def one(t: TopicSpec) -> List[Probe]:
        async with sem:
            return await _questions_for(t, per_topic, complete)

    return [p for batch in await asyncio.gather(*(one(t) for t in topics)) for p in batch]
