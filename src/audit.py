"""Phase 1 audit pipeline: probes -> RAG endpoint -> coverage scores -> CSV/Postgres."""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from src.connectors.base import RAGConnector, RAGResponse
from src.probe_generation.taxonomy import Probe
from src.scoring.composite import CoverageScore
from src.scoring.scorer import CoverageScorer, ScorerConfig

log = logging.getLogger(__name__)

CSV_FIELDS = [
    "probe_id", "query", "generation_strategy", "probe_topic",
    "coverage_score", "rc_score", "se_score", "hp_score", "hp_computed", "preliminary_score",
    "n_chunks", "retrieved_chunk_ids", "answer", "error",
]


@dataclass
class ProbeOutcome:
    probe_id: uuid.UUID
    probe: Probe
    response: RAGResponse
    score: Optional[CoverageScore]  # None when the RAG query failed
    query_embedding: Optional[np.ndarray] = field(default=None, repr=False)

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
) -> uuid.UUID:
    from src.db.models import AuditRun, ProbeResult
    from src.db.store import save_audit

    run = AuditRun(
        report_id=uuid.uuid4(),
        kb_fingerprint=kb_fingerprint,
        probe_count=len(outcomes),
        cluster_count=None,  # Phase 2
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
            retrieved_chunk_ids=o.response.chunk_ids,
            answer=o.response.answer,
            error=o.response.error,
        ))
    return save_audit(dsn, run, rows)
