"""Two UMAP projections of the probe embeddings, used for different things.

Clustering happens on the 10D projection. The 2D projection is only for the
coverage map. Never run HDBSCAN on the 2D output: squeezing 384 dims into 2
makes UMAP trade away distance fidelity for a readable picture. It tears some
neighborhoods apart and glues unrelated ones together at the seams, so density
clusters found in 2D are partly layout artifacts. 10D keeps enough of the local
structure for HDBSCAN's density estimate to mean something. Both coordinate
sets get stored per probe, so the map can be redrawn later without
re-projecting (which could reshuffle the layout).
"""

from __future__ import annotations

import numpy as np

RANDOM_STATE = 42


def _umap(n_components: int, n_points: int, **kwargs):
    import umap

    return umap.UMAP(
        n_components=n_components,
        # 15 is the spec's value for 5k-50k probes; small test sets need fewer
        n_neighbors=min(15, n_points - 1),
        metric="cosine",
        init="spectral",
        random_state=RANDOM_STATE,
        **kwargs,
    )


def project_for_clustering(query_embeddings: np.ndarray) -> np.ndarray:
    # min_dist=0 packs neighbors tightly, which is what a density clusterer
    # wants; it would make an unreadable scatter plot, which is fine here.
    reducer = _umap(10, len(query_embeddings), min_dist=0.0)
    return reducer.fit_transform(query_embeddings)


def project_to_2d(query_embeddings: np.ndarray) -> np.ndarray:
    reducer = _umap(2, len(query_embeddings), min_dist=0.1, spread=1.0, n_epochs=200)
    return reducer.fit_transform(query_embeddings)
