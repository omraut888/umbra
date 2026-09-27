from __future__ import annotations

import asyncio
from typing import List, Sequence

from src.clustering.zones import ClusterCoverage

NAMING_PROMPT = """These questions were grouped together because they are about the same topic:
{queries}

Name this topic cluster in 3-5 words. Reply with the name only."""

NOISE_NAME = "noise (unclustered)"


def _clean(name: str) -> str:
    name = name.strip().splitlines()[0] if name.strip() else ""
    return name.strip(" \"'*`.")[:100]  # cluster_name is VARCHAR(100)


async def name_clusters(clusters: Sequence[ClusterCoverage], complete, concurrency: int = 8) -> List[ClusterCoverage]:
    sem = asyncio.Semaphore(concurrency)

    async def one(c: ClusterCoverage) -> None:
        if c.is_noise:
            c.name = NOISE_NAME
            return
        async with sem:
            text = await complete(NAMING_PROMPT.format(queries="\n".join(f"- {q}" for q in c.representative_queries)))
        c.name = _clean(text) or f"cluster {c.cluster_id}"

    await asyncio.gather(*(one(c) for c in clusters))
    return list(clusters)
