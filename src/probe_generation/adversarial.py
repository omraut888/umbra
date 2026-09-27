"""Strategy 2: HyDE-style boundary probes (spec §3).

HyDE is normally used to improve retrieval: write a hypothetical answer
document and search with that. Here it's turned around. Ask for topics that
sit next to the KB but aren't in it, write a hypothetical document for each,
then ask for questions that document would answer. The questions end up
specific enough to need that missing document, rather than generic questions
about the neighboring topic, which the KB might half-answer.
"""

from __future__ import annotations

import asyncio
import random
import re
from typing import Dict, List, Sequence

from src.data.kb_loader import Chunk
from src.probe_generation.taxonomy import AsyncComplete, Probe, parse_questions

ADJACENT_TOPICS_PROMPT = """Given these document excerpts from a knowledge base:
{excerpts}

Generate {k} related topics that are NOT covered by these excerpts but are closely adjacent to them. These are the "boundary topics": things a user might naturally ask about after reading this content.
{avoid}
Return only topic names, one per line."""

HYPOTHETICAL_DOC_PROMPT = """Write a short factual reference passage (about 120 words) about: {topic}
Write it like an entry in a knowledge base on that subject. Plain prose, no headings."""

QUESTIONS_PROMPT = """Here is a passage about {topic}:

{document}

Generate {k} specific questions that this passage answers. Return only the questions, one per line."""

_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")


def parse_topics(text: str) -> List[str]:
    topics = []
    for line in text.splitlines():
        if line.lstrip().startswith("#"):  # Haiku sometimes adds a markdown heading
            continue
        t = _BULLET.sub("", line).strip().strip('"').strip("*").strip()
        if 3 <= len(t) <= 80 and not t.endswith(":"):
            topics.append(t)
    return topics


async def adversarial_boundary_generation(
    chunks: Sequence[Chunk],
    n: int,
    complete: AsyncComplete,
    topics_per_round: int = 20,
    questions_per_topic: int = 3,
    excerpts_per_round: int = 10,
    concurrency: int = 8,
    seed: int = 42,
    max_rounds: int = 10,
) -> List[Probe]:
    rng = random.Random(seed)
    sem = asyncio.Semaphore(concurrency)
    seen_topics: Dict[str, str] = {}  # lowercased -> original
    probes: Dict[str, Probe] = {}

    async def probe_topic(topic: str) -> List[Probe]:
        async with sem:
            doc = await complete(HYPOTHETICAL_DOC_PROMPT.format(topic=topic))
            text = await complete(QUESTIONS_PROMPT.format(topic=topic, document=doc.strip(), k=questions_per_topic))
        return [Probe(query=q, strategy="adversarial", topic=topic) for q in parse_questions(text)]

    for _ in range(max_rounds):
        if len(probes) >= n:
            break
        sample = rng.sample(list(chunks), min(excerpts_per_round, len(chunks)))
        avoid = ""
        if seen_topics:
            avoid = "\nDo not repeat any of these topics:\n" + "\n".join(f"- {t}" for t in list(seen_topics.values())[-40:]) + "\n"
        needed_topics = min(topics_per_round, -(-(n - len(probes)) // questions_per_topic))
        text = await complete(ADJACENT_TOPICS_PROMPT.format(
            excerpts="\n\n".join(c.text for c in sample), k=needed_topics, avoid=avoid))
        fresh = [t for t in parse_topics(text) if t.lower() not in seen_topics]
        for t in fresh:
            seen_topics[t.lower()] = t
        for batch in await asyncio.gather(*(probe_topic(t) for t in fresh)):
            for p in batch:
                probes.setdefault(p.query, p)

    return list(probes.values())[:n]
