"""Cluster-level coverage, zone tiers and severity (spec §6)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from src.clustering.hdbscan_clusterer import NOISE



@dataclass(frozen=True)
class ZoneThresholds:
    dark_below: float
    adequate_above: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.dark_below <= self.adequate_above <= 1.0:
            raise ValueError(f"need 0 <= dark_below <= adequate_above <= 1, got {self}")

    @classmethod
    def parse(cls, spec: str) -> "ZoneThresholds":
        dark, adequate = (float(x) for x in spec.split(","))
        return cls(dark, adequate)


SPEC_THRESHOLDS = ZoneThresholds(0.30, 0.60)
# Fit on the gap-injection benchmark (3 seeds, 46 labeled clusters) with MiniLM
# embeddings and dispersion SE; see docs/findings.md section 4. With MiniLM a
# well-answered probe tops out around 0.70, so the spec's 0.60 line put 25 of 28
# known-present clusters in THIN. Re-fit these if the embedding model changes.
DEFAULT_THRESHOLDS = ZoneThresholds(0.324, 0.400)


@dataclass(frozen=True)
class PurityRule:
    """Decides DARK vs THIN for clusters the score already says aren't adequate."""

    threshold: float
    dark_if_above: bool  # which side is DARK is fit on the benchmark, not assumed

    def is_dark(self, purity: float) -> bool:
        return purity >= self.threshold if self.dark_if_above else purity < self.threshold


def classify_zone(
    mean_cs: float,
    thresholds: ZoneThresholds = DEFAULT_THRESHOLDS,
    purity: Optional[float] = None,
    purity_rule: Optional[PurityRule] = None,
) -> str:
    if mean_cs > thresholds.adequate_above:
        return "ADEQUATE"
    if purity_rule is not None and purity is not None:
        return "DARK" if purity_rule.is_dark(purity) else "THIN"
    return "DARK" if mean_cs < thresholds.dark_below else "THIN"


def label_purity(labels: Sequence[Optional[str]]) -> Optional[float]:
    """Share of a cluster's probes that carry its most common topic label.

    The denominator is every probe in the cluster, labeled or not. In an audit
    the labels are kb_blind topic names (from the operator's topic list); the
    benchmark can also trace counterfactual probes to a topic through their
    source chunk. None when nothing in the cluster is labeled.
    """
    named = [x for x in labels if x]
    if not named:
        return None
    return max(named.count(x) for x in set(named)) / len(labels)


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
    purity: Optional[float] = None

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
    thresholds: ZoneThresholds = DEFAULT_THRESHOLDS,
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
            zone=classify_zone(mean_cs, thresholds),
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
