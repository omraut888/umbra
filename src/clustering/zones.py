"""Cluster-level coverage, zone tiers and severity (spec §6)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from src.clustering.hdbscan_clusterer import NOISE

DARK_BELOW = 0.30
ADEQUATE_ABOVE = 0.60


def classify_zone(mean_cs: float) -> str:
    if mean_cs < DARK_BELOW:
        return "DARK"
    if mean_cs <= ADEQUATE_ABOVE:
        return "THIN"
    return "ADEQUATE"


def severity(mean_cs: float, std_cs: float, size: int) -> float:
    # (1 - mean) * log(1 + size) * (1 - std): big, consistently failing clusters
    # come first. 500 probes all at 0.1 outrank 10 probes averaging 0.05.
    return (1.0 - mean_cs) * math.log1p(size) * (1.0 - std_cs)


@dataclass
class ClusterCoverage:
    cluster_id: int
    zone: str
    mean_cs: float
    std_cs: float
    query_count: int
    severity: float
    centroid_emb: np.ndarray = field(repr=False)
    centroid_x: float = 0.0
    centroid_y: float = 0.0
    representative_queries: List[str] = field(default_factory=list)
    name: Optional[str] = None
    strategy_mix: Dict[str, int] = field(default_factory=dict)

    @property
    def is_noise(self) -> bool:
        return self.cluster_id == NOISE


def representative_indices(embeddings: np.ndarray, k: int = 3) -> List[int]:
    """Indices of the k rows closest (cosine) to the centroid, nearest first."""
    centroid = embeddings.mean(axis=0)
    norms = np.linalg.norm(embeddings, axis=1) * (np.linalg.norm(centroid) or 1.0)
    sims = embeddings @ centroid / np.where(norms == 0, 1.0, norms)
    return np.argsort(-sims)[:k].tolist()


def compute_cluster_coverage(
    labels: np.ndarray,
    coverage_scores: Sequence[float],
    query_embeddings: np.ndarray,
    coords_2d: np.ndarray,
    queries: Sequence[str],
) -> List[ClusterCoverage]:
    """One ClusterCoverage per cluster (noise bucket included), most severe first.

    The noise bucket gets a zone from its own mean like any other cluster. The
    spec says to treat noise as dark by default, but noise here is "didn't
    fit a dense region", which in practice is a mix of covered and uncovered
    probes; its per-probe scores are still there for anyone who wants them.
    """
    scores = np.asarray(coverage_scores, dtype=float)
    results = []
    for cid in sorted(set(labels.tolist())):
        idx = np.flatnonzero(labels == cid)
        cs = scores[idx]
        mean_cs, std_cs = float(cs.mean()), float(cs.std())
        reps = [queries[idx[i]] for i in representative_indices(query_embeddings[idx])]
        results.append(ClusterCoverage(
            cluster_id=int(cid),
            zone=classify_zone(mean_cs),
            mean_cs=mean_cs,
            std_cs=std_cs,
            query_count=len(idx),
            severity=severity(mean_cs, std_cs, len(idx)),
            centroid_emb=query_embeddings[idx].mean(axis=0),
            centroid_x=float(coords_2d[idx, 0].mean()),
            centroid_y=float(coords_2d[idx, 1].mean()),
            representative_queries=reps,
        ))
    return sorted(results, key=lambda c: -c.severity)
