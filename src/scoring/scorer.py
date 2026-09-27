"""Batch scoring: every query and distinct chunk text is embedded once."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

from src.embeddings import embed
from src.scoring.composite import DEFAULT_HP_BAND, CoverageScore, ScoringWeights, composite_coverage_score
from src.scoring.signals import (
    DEFAULT_SE_METHOD,
    hallucination_probability,
    retrieval_confidence,
    semantic_entropy_from_embeddings,
)


@dataclass
class ScorerConfig:
    weights: ScoringWeights = field(default_factory=ScoringWeights)
    hp_band: Tuple[float, float] = DEFAULT_HP_BAND
    se_method: str = DEFAULT_SE_METHOD

    def as_dict(self) -> dict:
        return {
            "alpha": self.weights.alpha,
            "beta": self.weights.beta,
            "gamma": self.weights.gamma,
            "hp_band": list(self.hp_band),
            "se_method": self.se_method,
        }


class CoverageScorer:
    def __init__(self, config: ScorerConfig | None = None, cross_encoder=None):
        self.config = config or ScorerConfig()
        self._cross_encoder = cross_encoder

    def score_batch(
        self, queries: Sequence[str], chunk_lists: Sequence[Sequence[str]]
    ) -> Tuple[List[CoverageScore], np.ndarray]:
        # query embeddings are handed back so the audit can store them in pgvector
        if len(queries) != len(chunk_lists):
            raise ValueError("queries and chunk_lists must have the same length")
        query_embs = embed(list(queries))

        unique_texts = sorted({t for chunks in chunk_lists for t in chunks})
        text_embs = embed(unique_texts)
        index: Dict[str, int] = {t: i for i, t in enumerate(unique_texts)}

        cfg = self.config
        results: List[CoverageScore] = []
        for query, q_emb, chunks in zip(queries, query_embs, chunk_lists):
            chunks = list(chunks)
            if not chunks:
                # otherwise the spec's SE=0 for <2 chunks hands an empty retrieval a free β
                results.append(CoverageScore(score=0.0, rc=0.0, se=0.0, hp=1.0, preliminary=0.0))
                continue
            c_embs = text_embs[[index[t] for t in chunks]]
            rc = retrieval_confidence(q_emb, c_embs)
            se = semantic_entropy_from_embeddings(c_embs, method=cfg.se_method)
            results.append(
                composite_coverage_score(
                    rc,
                    se,
                    lambda q=query, c=chunks: hallucination_probability(q, c, self._cross_encoder),
                    weights=cfg.weights,
                    hp_band=cfg.hp_band,
                )
            )
        return results, query_embs
