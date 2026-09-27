"""Run any subset of the probe strategies and merge the results.

Each probe keeps its `strategy` tag through scoring and clustering, so an audit
can be sliced by strategy afterwards (which strategy found which dark zone).
"""

from __future__ import annotations

import asyncio
from typing import Dict, List, Optional, Sequence, Tuple

from src.data.kb_loader import Chunk
from src.probe_generation.adversarial import adversarial_boundary_generation
from src.probe_generation.counterfactual import counterfactual_generation
from src.probe_generation.kb_blind import TopicSpec, kb_blind_generation
from src.probe_generation.taxonomy import (
    AsyncComplete,
    Probe,
    Topic,
    dedup_probes,
    extract_taxonomy,
    taxonomy_guided_generation,
)

# spec §3 probe set composition for the KB-anchored strategies. The 10%
# "userpattern" share needs production query logs, so it isn't generated; the
# remaining shares are renormalized.
STRATEGY_SHARES = {"taxonomy": 0.40, "adversarial": 0.30, "counterfactual": 0.20}
# kb_blind isn't part of the share split. It's sized per topic instead: a topic
# needs at least min_cluster_size probes (15) to come out as its own cluster,
# and a 10% share of 1000 spread over 15 topics would leave every missing topic
# scattered in noise.
ALL_STRATEGIES = (*STRATEGY_SHARES, "kb_blind")


def split_budget(n: int, strategies: Sequence[str]) -> Dict[str, int]:
    unknown = set(strategies) - set(ALL_STRATEGIES)
    if unknown:
        raise ValueError(f"unknown strategies {sorted(unknown)}; choose from {ALL_STRATEGIES}")
    strategies = [s for s in strategies if s in STRATEGY_SHARES]
    if not strategies:
        return {}
    total = sum(STRATEGY_SHARES[s] for s in strategies)
    budget = {s: int(n * STRATEGY_SHARES[s] / total) for s in strategies}
    # hand the rounding remainder to the first strategy so the total is exact
    budget[strategies[0]] += n - sum(budget.values())
    return budget


async def generate_probe_set(
    chunks: Sequence[Chunk],
    n: int,
    strategies: Sequence[str],
    complete: AsyncComplete,
    extra_probes: Sequence[Probe] = (),
    kb_blind_topics: Sequence[TopicSpec] = (),
    kb_blind_per_topic: int = 30,
) -> Tuple[List[Probe], Optional[List[Topic]]]:
    """Returns (deduplicated probes, taxonomy topics if taxonomy ran)."""
    budget = split_budget(n, strategies)
    if "kb_blind" in strategies and not kb_blind_topics:
        raise ValueError("kb_blind needs topic descriptions (a topics file or a domain description)")
    topics = extract_taxonomy(chunks) if "taxonomy" in budget else None

    jobs = {}
    if "taxonomy" in budget:
        jobs["taxonomy"] = taxonomy_guided_generation(topics, budget["taxonomy"], complete)
    if "adversarial" in budget:
        jobs["adversarial"] = adversarial_boundary_generation(chunks, budget["adversarial"], complete)
    if "counterfactual" in budget:
        jobs["counterfactual"] = counterfactual_generation(chunks, budget["counterfactual"], complete)

    if "kb_blind" in strategies:
        jobs["kb_blind"] = kb_blind_generation(kb_blind_topics, kb_blind_per_topic, complete)

    results = await asyncio.gather(*jobs.values())
    combined = [p for batch in results for p in batch] + list(extra_probes)
    # dedup across strategies too (spec §14, cosine >= 0.95): two strategies
    # asking the same question would double-weight that spot in the map
    return dedup_probes(combined), topics
