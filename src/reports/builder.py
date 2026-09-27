"""Assemble a GapReport from an audit's probes and clusters."""

from __future__ import annotations

import logging
import random
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence

import numpy as np

from src.audit import ProbeOutcome
from src.clustering.naming import NOISE_NAME
from src.clustering.zones import (
    DEFAULT_THRESHOLDS,
    SEVERITY_SIZE_CAP,
    ClusterCoverage,
    ZoneThresholds,
    classify_zone,
    severity,
)
from src.data.kb_loader import Chunk
from src.probe_generation.taxonomy import AsyncComplete
from src.reports.recommend import (
    KBInternalRecommender,
    WebSearchProvider,
    external_candidates,
    rank_by_gain,
    simulate,
)
from src.reports.schema import ClusterReport, DocumentRec, GapReport, ImprovementEstimate, Thresholds
from src.scoring.scorer import CoverageScorer

log = logging.getLogger(__name__)

SAMPLE_QUERIES = 10
RECS_PER_CLUSTER = 5


def is_unanswered(o: ProbeOutcome, hp_band_high: float = 0.65) -> bool:
    s = o.score
    if s.hp is not None:
        return s.hp >= 0.5
    # HP skipped: the cheap signals already put the probe clearly high or low
    return s.preliminary <= hp_band_high


def _sample_queries(members: Sequence[ProbeOutcome], rng: random.Random) -> List[str]:
    # half the lowest-scoring probes (the ones worth checking), half random
    worst = sorted(members, key=lambda o: o.score.score)[: SAMPLE_QUERIES // 2]
    rest = [o for o in members if o not in worst]
    picked = worst + rng.sample(rest, min(len(rest), SAMPLE_QUERIES - len(worst)))
    return [o.probe.query for o in picked]


def select_targets(clusters: Sequence[ClusterCoverage], top_k: int) -> Dict[int, str]:
    """Clusters that get recommendations, and why: every DARK and THIN cluster,
    plus the top_k most severe overall (catches depth gaps in ADEQUATE clusters).
    The noise bucket is skipped: it's leftovers from many topics, not one gap."""
    real = [c for c in clusters if not c.is_noise]
    reasons = {c.cluster_id: c.zone for c in real if c.zone in ("DARK", "THIN")}
    for c in sorted(real, key=lambda c: -c.severity)[:top_k]:
        reasons.setdefault(c.cluster_id, f"top {top_k} by severity")
    return reasons


async def build_report(
    outcomes: Sequence[ProbeOutcome],
    clusters: Sequence[ClusterCoverage],
    kb_chunks: Sequence[Chunk],
    *,
    kb_fingerprint: Optional[str] = None,
    thresholds: ZoneThresholds = DEFAULT_THRESHOLDS,
    thresholds_source: str = "calibrated default",
    scorer: Optional[CoverageScorer] = None,
    complete: Optional[AsyncComplete] = None,
    search: Optional[WebSearchProvider] = None,
    top_k: int = 10,
    coverage_map_uri: Optional[str] = None,
    config: Optional[dict] = None,
    seed: int = 42,
) -> GapReport:
    scorer = scorer or CoverageScorer()
    scored = [o for o in outcomes if o.score is not None and o.cluster_id is not None]
    members: Dict[int, List[ProbeOutcome]] = defaultdict(list)
    for o in scored:
        members[o.cluster_id].append(o)

    # zones and severity are recomputed with the current rules, so an older
    # audit reads the same way a new one would
    for c in clusters:
        c.zone = classify_zone(c.mean_cs, thresholds)
        c.severity = severity(c.mean_cs, c.std_cs, c.query_count)
    ordered = sorted(clusters, key=lambda c: -c.severity)
    targets = select_targets(ordered, top_k)

    kb_rec = KBInternalRecommender(kb_chunks)
    recs: Dict[int, List[DocumentRec]] = {}
    for c in ordered:
        if c.cluster_id not in targets:
            continue
        ms = members[c.cluster_id]
        candidates = kb_rec.candidates(ms, c.centroid_emb)
        if complete is not None and search is not None:
            try:
                candidates += await external_candidates(ms, c.representative_queries, c.centroid_emb, complete, search)
            except Exception as exc:  # one failed search shouldn't lose the whole report
                log.warning("web search failed for cluster %s: %s", c.cluster_id, exc)
        ranked = rank_by_gain(scorer, ms, candidates, RECS_PER_CLUSTER)
        recs[c.cluster_id] = [
            DocumentRec(
                kind=x.kind, cluster_id=c.cluster_id, rank=i + 1, title=x.title, url=x.url, doc_id=x.doc_id,
                chunk_id=x.chunk_id, snippet=x.snippet, similarity=x.similarity, expected_gain=x.expected_gain,
                search_query=x.search_query, retrieval_rate=x.retrieval_rate, answered_probes=x.answered_probes,
                action=(f"Give this passage its own chunk or document. It matches this gap but sits inside "
                        f"{x.chunk_id}, which is mostly about something else and was retrieved for "
                        f"{x.retrieval_rate:.0%} of this cluster's probes."
                        if x.kind == "kb_internal" else "Add this document (or its relevant section) to the KB."),
            )
            for i, x in enumerate(ranked)
        ]

    rng = random.Random(seed)
    reports = []
    for rank, c in enumerate(ordered, start=1):
        ms = members[c.cluster_id]
        reports.append(ClusterReport(
            cluster_id=c.cluster_id, name=c.name or (NOISE_NAME if c.is_noise else f"cluster {c.cluster_id}"),
            zone=c.zone, mean_cs=c.mean_cs, std_cs=c.std_cs, severity=c.severity, severity_rank=rank,
            query_count=c.query_count,
            unanswered_share=sum(is_unanswered(o) for o in ms) / len(ms) if ms else 0.0,
            purity=c.purity, strategy_mix=c.strategy_mix, centroid_x=c.centroid_x, centroid_y=c.centroid_y,
            representative_queries=c.representative_queries, sample_queries=_sample_queries(ms, rng) if ms else [],
            recommendation_reason=targets.get(c.cluster_id), recommendations=recs.get(c.cluster_id, []),
        ))

    all_recs = [r for c in reports for r in c.recommendations]
    detail = estimate_improvement(scorer, scored, reports, members)
    breakdown: Dict[str, Counter] = defaultdict(Counter)
    zone_of = {c.cluster_id: c.zone for c in ordered}
    for o in scored:
        breakdown[o.probe.strategy][zone_of[o.cluster_id]] += 1

    return GapReport(
        report_id=uuid.uuid4(),
        kb_fingerprint=kb_fingerprint,
        probe_count=len(outcomes),
        cluster_count=sum(not c.is_noise for c in clusters),
        overall_coverage_score=float(np.mean([o.score.score for o in scored])),
        dark_zones=[r for r in reports if r.zone == "DARK"],
        thin_zones=[r for r in reports if r.zone == "THIN"],
        clusters=reports,
        recommendations=all_recs,
        coverage_map_uri=coverage_map_uri,
        estimated_improvement=detail.overall_after,
        estimated_improvement_detail=detail,
        thresholds=Thresholds(dark_below=thresholds.dark_below, adequate_above=thresholds.adequate_above,
                              source=thresholds_source),
        severity_size_cap=SEVERITY_SIZE_CAP,
        strategy_breakdown={s: dict(z) for s, z in breakdown.items()},
        config=config or {},
        generated_at=datetime.now(timezone.utc),
    )


def estimate_improvement(scorer, scored, reports: Sequence[ClusterReport], members) -> ImprovementEstimate:
    """Apply the best recommendation of each of the three most severe clusters
    that have one, and recompute the overall score."""
    before = {o.probe_id: o.score.score for o in scored}
    after = dict(before)
    applied = []
    for c in reports:  # already severity-sorted
        if len(applied) == 3:
            break
        if not c.recommendations:
            continue
        ms = members[c.cluster_id]
        for o, s in zip(ms, simulate(scorer, ms, c.recommendations[0].snippet)):
            after[o.probe_id] = s
        applied.append(c.cluster_id)
    return ImprovementEstimate(
        overall_before=float(np.mean(list(before.values()))) if before else 0.0,
        overall_after=float(np.mean(list(after.values()))) if after else 0.0,
        recommendations_applied=len(applied),
        clusters_affected=applied,
        method="Each applied recommendation's text replaces the lowest-ranked retrieved chunk for every probe in "
               "its cluster, and those probes are re-scored. This assumes the retriever would return the new "
               "text, so it's an optimistic estimate.",
    )
