"""Strategy 3: counterfactual near-miss probes (spec §3).

Questions that look answerable from a real KB chunk but need one fact the
chunk doesn't contain. These test whether the system admits it doesn't know
or fills the gap with something plausible.
"""

from __future__ import annotations

import asyncio
import math
import random
from typing import List, Sequence

from src.data.kb_loader import Chunk
from src.probe_generation.taxonomy import AsyncComplete, Probe, parse_questions

COUNTERFACTUAL_PROMPT = """Given this document excerpt:
"{chunk}"

Generate {k} questions that seem related to this content but require information NOT contained in it to answer correctly.
The questions should be specific and factual.
Return only the questions, one per line."""


async def counterfactual_generation(
    chunks: Sequence[Chunk],
    n: int,
    complete: AsyncComplete,
    questions_per_chunk: int = 2,
    concurrency: int = 8,
    seed: int = 42,
) -> List[Probe]:
    rng = random.Random(seed)
    n_chunks = math.ceil(n / questions_per_chunk)
    # more probes requested than chunks x 2: cycle through the KB again
    picked = []
    while len(picked) < n_chunks:
        picked.extend(rng.sample(list(chunks), min(len(chunks), n_chunks - len(picked))))

    sem = asyncio.Semaphore(concurrency)

    async def one(chunk: Chunk) -> List[Probe]:
        async with sem:
            text = await complete(COUNTERFACTUAL_PROMPT.format(chunk=chunk.text, k=questions_per_chunk))
        return [Probe(query=q, strategy="counterfactual", topic=chunk.chunk_id) for q in parse_questions(text)]

    probes = {}
    for batch in await asyncio.gather(*(one(c) for c in picked)):
        for p in batch:
            probes.setdefault(p.query, p)
    return list(probes.values())[:n]
