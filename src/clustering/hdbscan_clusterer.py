from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Optional

import numpy as np

NOISE = -1
MAX_NAMED_CLUSTERS = 50
MIN_CLUSTER_SHARE = 0.005  # spec §14: fold clusters under 0.5% of probes into noise


@dataclass
class Clustering:
    labels: np.ndarray  # final ids after capping/merging; -1 = noise bucket
    raw_labels: np.ndarray  # straight from HDBSCAN
    probabilities: np.ndarray  # HDBSCAN membership strength for the assigned cluster
    soft_membership: Optional[np.ndarray]  # (n_probes, n_raw_clusters), from prediction_data
    clusterer: object

    @property
    def n_clusters(self) -> int:
        return len(set(self.labels.tolist()) - {NOISE})


def identify_clusters(
    embeddings_10d: np.ndarray,
    min_cluster_size: int = 20,
    min_samples: int = 5,
    max_clusters: int = MAX_NAMED_CLUSTERS,
    min_share: float = MIN_CLUSTER_SHARE,
) -> Clustering:
    import hdbscan

    n = len(embeddings_10d)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",  # on UMAP output; cosine was already handled by UMAP
        cluster_selection_method="eom",
        prediction_data=True,
    )
    raw = clusterer.fit_predict(embeddings_10d)

    sizes = Counter(raw[raw != NOISE].tolist())
    floor = math.ceil(min_share * n)
    keep = [c for c, s in sizes.most_common() if s >= floor][:max_clusters]
    # Re-number surviving clusters by size so ids are stable for a given probe
    # set, instead of whatever order HDBSCAN's condensed tree produced.
    remap = {old: new for new, old in enumerate(keep)}
    labels = np.array([remap.get(c, NOISE) for c in raw.tolist()], dtype=int)

    soft = None
    if sizes:
        soft = hdbscan.all_points_membership_vectors(clusterer)
        soft = soft.reshape(n, -1)

    return Clustering(labels=labels, raw_labels=raw, probabilities=clusterer.probabilities_,
                      soft_membership=soft, clusterer=clusterer)
