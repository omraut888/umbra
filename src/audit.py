"""Audit pipeline: probes -> RAG endpoint -> coverage scores -> clusters -> CSV/Postgres."""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import uuid
from dataclasses import dataclass, field
from collections import Counter
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from src.clustering.hdbscan_clusterer import NOISE, identify_clusters
from src.clustering.umap_projector import project_for_clustering, project_to_2d
from src.clustering.zones import (
    DEFAULT_THRESHOLDS,
    ClusterCoverage,
    PurityRule,
    ZoneThresholds,
    classify_zone,
    compute_cluster_coverage,
    label_purity,
)
from src.connectors.base import RAGConnector, RAGResponse
from src.probe_generation.taxonomy import Probe
from src.scoring.composite import CoverageScore
from src.scoring.scorer import CoverageScorer, ScorerConfig

log = logging.getLogger(__name__)

CSV_FIELDS = [
    "probe_id", "query", "generation_strategy", "probe_topic",
    "coverage_score", "rc_score", "se_score", "hp_score", "hp_computed", "preliminary_score",
    "cluster_id", "cluster_prob", "umap_x", "umap_y", "umap_10d",
    "n_chunks", "retrieved_chunk_ids", "answer", "error",
]

CLUSTER_CSV_FIELDS = [
    "cluster_id", "name", "zone", "query_count", "mean_cs", "std_cs", "severity", "purity",
    "strategy_mix", "representative_queries",
]


@dataclass
class ProbeOutcome:
    probe_id: uuid.UUID
    probe: Probe
    response: RAGResponse
    score: Optional[CoverageScore]  # None when the RAG query failed
    query_embedding: Optional[np.ndarray] = field(default=None, repr=False)
    cluster_id: Optional[int] = None
    cluster_prob: Optional[float] = None
    umap_x: Optional[float] = None
    umap_y: Optional[float] = None
    umap_10d: Optional[List[float]] = field(default=None, repr=False)

    def csv_row(self) -> dict:
        s = self.score
        return {
            "probe_id": str(self.probe_id),
            "query": self.probe.query,
            "generation_strategy": self.probe.strategy,
            "probe_topic": self.probe.topic or "",
            "coverage_score": _fmt(s.score if s else None),
            "rc_score": _fmt(s.rc if s else None),
            "se_score": _fmt(s.se if s else None),
            "hp_score": _fmt(s.hp if s else None),
            "hp_computed": s.hp_computed if s else "",
            "preliminary_score": _fmt(s.preliminary if s else None),
            "cluster_id": "" if self.cluster_id is None else self.cluster_id,
            "cluster_prob": _fmt(self.cluster_prob),
            "umap_x": _fmt(self.umap_x),
            "umap_y": _fmt(self.umap_y),
            "umap_10d": json.dumps([round(v, 4) for v in self.umap_10d]) if self.umap_10d else "",
            "n_chunks": len(self.response.chunks),
            "retrieved_chunk_ids": json.dumps(self.response.chunk_ids),
            "answer": self.response.answer,
            "error": self.response.error or "",
        }


def _fmt(x: Optional[float]) -> str:
    return "" if x is None else f"{x:.4f}"


async def run_probes(
    probes: Sequence[Probe],
    connector: RAGConnector,
    scorer: CoverageScorer,
    concurrency: int = 50,
) -> List[ProbeOutcome]:
    responses = await connector.query_many([p.query for p in probes], concurrency=concurrency)
    ok = [i for i, r in enumerate(responses) if r.error is None]
    failed = len(probes) - len(ok)
    if failed:
        log.warning("%d/%d probe queries failed; they are reported with an error and no score", failed, len(probes))

    # Scoring is CPU/GPU-bound; keep it off the event loop.
    scores, query_embs = await asyncio.to_thread(
        scorer.score_batch, [probes[i].query for i in ok], [responses[i].chunk_texts for i in ok]
    )
    by_index = {i: (s, e) for i, s, e in zip(ok, scores, query_embs)}
    outcomes = []
    for i, (probe, resp) in enumerate(zip(probes, responses)):
        score, emb = by_index.get(i, (None, None))
        outcomes.append(ProbeOutcome(uuid.uuid4(), probe, resp, score, emb))
    return outcomes


def write_csv(outcomes: Sequence[ProbeOutcome], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(o.csv_row() for o in outcomes)


def cluster_outcomes(
    outcomes: Sequence[ProbeOutcome],
    min_cluster_size: int = 20,
    min_samples: int = 5,
    thresholds: ZoneThresholds = DEFAULT_THRESHOLDS,
    purity_rule: Optional[PurityRule] = None,
) -> List[ClusterCoverage]:
    """Project, cluster and zone the scored probes. Fills cluster/UMAP fields in place."""
    scored = [o for o in outcomes if o.score is not None]
    embs = np.vstack([o.query_embedding for o in scored])
    coords_10d = project_for_clustering(embs)
    coords_2d = project_to_2d(embs)
    clustering = identify_clusters(coords_10d, min_cluster_size=min_cluster_size, min_samples=min_samples)

    for o, label, prob, c10, c2 in zip(scored, clustering.labels, clustering.probabilities, coords_10d, coords_2d):
        o.cluster_id = int(label)
        o.cluster_prob = float(prob) if label != NOISE else 0.0
        o.umap_10d = [float(v) for v in c10]
        o.umap_x, o.umap_y = float(c2[0]), float(c2[1])

    clusters = compute_cluster_coverage(
        clustering.labels, [o.score.score for o in scored], embs, coords_2d, [o.probe.query for o in scored],
        thresholds)
    strategies = {}
    for o in scored:
        strategies.setdefault(o.cluster_id, Counter())[o.probe.strategy] += 1
    blind_labels = {}
    for o in scored:
        blind_labels.setdefault(o.cluster_id, []).append(o.probe.topic if o.probe.strategy == "kb_blind" else None)
    for c in clusters:
        c.strategy_mix = dict(strategies[c.cluster_id].most_common())
        c.purity = label_purity(blind_labels[c.cluster_id])
        if purity_rule is not None and not c.is_noise:
            c.zone = classify_zone(c.mean_cs, thresholds, c.purity, purity_rule)
    log.info("%d clusters + noise from %d probes", clustering.n_clusters, len(scored))
    return clusters


def write_cluster_csv(clusters: Sequence[ClusterCoverage], path: str | Path) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CLUSTER_CSV_FIELDS)
        writer.writeheader()
        for c in clusters:
            writer.writerow({
                "cluster_id": c.cluster_id, "name": c.name or "", "zone": c.zone, "query_count": c.query_count,
                "mean_cs": f"{c.mean_cs:.4f}", "std_cs": f"{c.std_cs:.4f}", "severity": f"{c.severity:.4f}",
                "purity": "" if c.purity is None else f"{c.purity:.4f}",
                "strategy_mix": json.dumps(c.strategy_mix),
                "representative_queries": json.dumps(c.representative_queries),
            })


def write_responses(outcomes: Sequence[ProbeOutcome], path: str | Path) -> None:
    # retrieved chunk texts, which the report needs for KB-internal recommendations;
    # kept out of the CSV because five chunks per probe make it unreadable
    with Path(path).open("w", encoding="utf-8") as f:
        for o in outcomes:
            f.write(json.dumps({"probe_id": str(o.probe_id), "chunk_ids": o.response.chunk_ids,
                                "chunk_texts": o.response.chunk_texts}) + "\n")


def overall_score(outcomes: Sequence[ProbeOutcome]) -> Optional[float]:
    scores = [o.score.score for o in outcomes if o.score]
    return float(np.mean(scores)) if scores else None


def persist(
    dsn: str,
    outcomes: Sequence[ProbeOutcome],
    *,
    kb_fingerprint: str,
    endpoint_url: str,
    kb_path: str,
    config: ScorerConfig,
    extra_config: Optional[dict] = None,
    clusters: Sequence[ClusterCoverage] = (),
) -> uuid.UUID:
    from src.db.models import AuditRun, ClusterSummary, ProbeResult
    from src.db.store import save_audit

    run = AuditRun(
        report_id=uuid.uuid4(),
        kb_fingerprint=kb_fingerprint,
        probe_count=len(outcomes),
        cluster_count=sum(not c.is_noise for c in clusters) if clusters else None,
        overall_score=overall_score(outcomes),
        config={**config.as_dict(), **(extra_config or {})},
        endpoint_url=endpoint_url,
        kb_path=kb_path,
    )
    rows = []
    for o in outcomes:
        s = o.score
        rows.append(ProbeResult(
            probe_id=o.probe_id,
            query_text=o.probe.query,
            query_embedding=o.query_embedding.tolist() if o.query_embedding is not None else None,
            coverage_score=s.score if s else None,
            rc_score=s.rc if s else None,
            se_score=s.se if s else None,
            hp_score=s.hp if s else None,
            hp_computed=s.hp_computed if s else None,
            preliminary_score=s.preliminary if s else None,
            generation_strategy=o.probe.strategy,
            probe_topic=o.probe.topic,
            cluster_id=o.cluster_id,
            umap_x=o.umap_x,
            umap_y=o.umap_y,
            umap_10d=o.umap_10d,
            retrieved_chunk_ids=o.response.chunk_ids,
            answer=o.response.answer,
            error=o.response.error,
        ))
    summaries = [
        ClusterSummary(
            cluster_id=c.cluster_id, cluster_name=c.name, zone=c.zone, mean_cs=c.mean_cs, std_cs=c.std_cs,
            severity=c.severity, query_count=c.query_count, centroid_emb=c.centroid_emb.tolist(),
            centroid_x=c.centroid_x, centroid_y=c.centroid_y, strategy_mix=c.strategy_mix,
            representative_queries=c.representative_queries,
        )
        for c in clusters
    ]
    return save_audit(dsn, run, rows, summaries)
