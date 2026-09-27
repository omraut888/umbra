"""The three coverage signals (spec §4). All return floats in [0, 1].

    RC  max cosine(query, chunk)               higher = better coverage
    SE  spread of the retrieved chunks         higher = worse coverage
    HP  1 - sigmoid(max cross-encoder logit)   higher = worse coverage
"""

from __future__ import annotations

import math
from typing import Callable, List, Sequence

import numpy as np
from scipy.spatial.distance import pdist
from scipy.special import expit
from scipy.stats import entropy

from src.embeddings import MODEL_LOCK, embed, get_cross_encoder

# "dispersion" is the default: see docs/findings.md for why the spec's
# histogram-entropy formula ("spec") gets uniformly irrelevant retrievals wrong.
SE_METHODS = ("dispersion", "spec")
DEFAULT_SE_METHOD = "dispersion"

SE_BINS = 10
# The spec calls np.histogram without a range, so bins span the data's own
# min..max. That makes the entropy scale-invariant -- 10 distances spread over
# 0.30-0.35 look the same as 0.1-0.9 -- and SE comes out ~0.54-0.78 for every
# test case. A fixed [0, 1] range is the minimum needed for "spec" to carry any
# signal. Cosine distances between MiniLM embeddings stay in [0, 1] in practice.
SE_RANGE = (0.0, 1.0)


def retrieval_confidence(
    query_embedding: Sequence[float] | np.ndarray,
    retrieved_chunk_embeddings: Sequence[Sequence[float]] | np.ndarray,
) -> float:
    chunks = np.asarray(retrieved_chunk_embeddings, dtype=np.float64)
    if chunks.size == 0:
        return 0.0
    chunks = np.atleast_2d(chunks)
    q = np.asarray(query_embedding, dtype=np.float64).ravel()
    q_norm = np.linalg.norm(q)
    if q_norm == 0:
        return 0.0
    with np.errstate(invalid="ignore", divide="ignore"):
        sims = (chunks @ q) / (np.linalg.norm(chunks, axis=1) * q_norm)
    sims = np.nan_to_num(sims, nan=0.0)
    # negative cosine is clipped so RC stays usable as a [0, 1] weight
    return float(np.clip(sims.max(), 0.0, 1.0))


def spec_histogram_entropy(embs: np.ndarray) -> float:
    """The original spec §4 formula: pairwise cosine distances -> 10-bin
    histogram -> Shannon entropy, divided by ln(10) so it fits in [0, 1].

    It measures how varied the distances are, not how large. Chunks that are all
    unrelated to each other sit at uniformly large distances, land in one bin,
    and score ~0 -- the same as near-duplicates.
    """
    distances = np.clip(pdist(embs, metric="cosine"), *SE_RANGE)
    distances = np.nan_to_num(distances, nan=SE_RANGE[1])
    hist, _ = np.histogram(distances, bins=SE_BINS, range=SE_RANGE, density=True)
    hist = hist + 1e-10
    return float(entropy(hist / hist.sum())) / math.log(SE_BINS)


def mean_pairwise_distance(embs: np.ndarray) -> float:
    return float(np.nan_to_num(pdist(embs, metric="cosine"), nan=1.0).mean())


def semantic_entropy_from_embeddings(
    chunk_embeddings: Sequence[Sequence[float]] | np.ndarray,
    method: str = DEFAULT_SE_METHOD,
) -> float:
    if method not in SE_METHODS:
        raise ValueError(f"unknown SE method {method!r}; expected one of {SE_METHODS}")
    embs = np.asarray(chunk_embeddings, dtype=np.float64)
    if embs.ndim != 2 or embs.shape[0] < 2:
        return 0.0  # spec: fewer than 2 chunks -> 0
    value = mean_pairwise_distance(embs) if method == "dispersion" else spec_histogram_entropy(embs)
    return float(np.clip(value, 0.0, 1.0))


def semantic_entropy(
    retrieved_chunks: List[str],
    embed_fn: Callable[[List[str]], np.ndarray] = embed,
    method: str = DEFAULT_SE_METHOD,
) -> float:
    if len(retrieved_chunks) < 2:
        return 0.0
    return semantic_entropy_from_embeddings(embed_fn(list(retrieved_chunks)), method=method)


def hallucination_probability(query: str, retrieved_chunks: List[str], cross_encoder=None) -> float:
    if not retrieved_chunks:
        return 1.0
    model = cross_encoder if cross_encoder is not None else get_cross_encoder()
    with MODEL_LOCK:
        logits = model.predict([(query, c) for c in retrieved_chunks], show_progress_bar=False)
    # ms-marco outputs raw logits; sigmoid maps the best one to a relevance probability
    return 1.0 - float(expit(np.max(logits)))
