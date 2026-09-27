"""Rebuild an audit's probes and clusters from its output files.

An audit writes <name>.csv (probes), <name>.clusters.csv and, since Phase 3,
<name>.responses.jsonl with the retrieved chunk texts. Older runs have no
responses file; for those, retrieved chunk ids are resolved against the KB,
which works when the RAG system chunked the KB with Umbra's own loader (the
mock server does). Unresolvable ids come back as missing text, and the
KB-internal recommender just sees less retrieval.
"""

from __future__ import annotations

import csv
import json
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from src.audit import ProbeOutcome
from src.clustering.zones import ClusterCoverage
from src.connectors.base import RAGResponse, RetrievedChunk
from src.data.kb_loader import Chunk
from src.embeddings import embed
from src.probe_generation.taxonomy import Probe
from src.scoring.composite import CoverageScore

log = logging.getLogger(__name__)


@dataclass
class LoadedAudit:
    outcomes: List[ProbeOutcome]
    clusters: List[ClusterCoverage]
    path: Path


def _float(x: str) -> Optional[float]:
    return float(x) if x not in ("", None) else None


def _mix(mix: Dict[str, int]) -> Dict[str, int]:
    if "ground_truth" in mix:
        mix["kb_blind"] = mix.get("kb_blind", 0) + mix.pop("ground_truth")
    return mix


def load_audit(probes_csv: str | Path, kb_chunks: List[Chunk]) -> LoadedAudit:
    probes_csv = Path(probes_csv)
    outcomes = load_probes(probes_csv, kb_chunks)
    return LoadedAudit(outcomes, _load_clusters(probes_csv, outcomes), probes_csv)


def load_probes(probes_csv: str | Path, kb_chunks: List[Chunk]) -> List[ProbeOutcome]:
    probes_csv = Path(probes_csv)
    rows = list(csv.DictReader(open(probes_csv, newline="", encoding="utf-8")))
    responses_path = probes_csv.with_suffix(".responses.jsonl")
    texts_by_probe: Dict[str, List[str]] = {}
    if responses_path.exists():
        for line in responses_path.read_text().splitlines():
            r = json.loads(line)
            texts_by_probe[r["probe_id"]] = r["chunk_texts"]
    chunk_text = {c.chunk_id: c.text for c in kb_chunks}
    unresolved = 0

    scored_rows = [r for r in rows if r["coverage_score"]]
    embs = embed([r["query"] for r in scored_rows])
    emb_of = {r["probe_id"]: e for r, e in zip(scored_rows, embs)}

    outcomes = []
    for r in rows:
        ids = json.loads(r["retrieved_chunk_ids"] or "[]")
        if r["probe_id"] in texts_by_probe:
            texts = texts_by_probe[r["probe_id"]]
        else:
            texts = [chunk_text.get(i, "") for i in ids]
            unresolved += sum(i not in chunk_text for i in ids)
        response = RAGResponse(
            question=r["query"],
            chunks=[RetrievedChunk(text=t, chunk_id=i) for t, i in zip(texts, ids)],
            answer=r["answer"],
            error=r["error"] or None,
        )
        score = None
        if r["coverage_score"]:
            score = CoverageScore(
                score=float(r["coverage_score"]), rc=float(r["rc_score"]), se=float(r["se_score"]),
                hp=_float(r["hp_score"]), preliminary=float(r["preliminary_score"]),
            )
        # runs from before kb_blind was a strategy tagged those probes "ground_truth"
        strategy = "kb_blind" if r["generation_strategy"] == "ground_truth" else r["generation_strategy"]
        o = ProbeOutcome(uuid.UUID(r["probe_id"]), Probe(r["query"], strategy, r["probe_topic"] or None),
                         response, score, emb_of.get(r["probe_id"]))
        o.cluster_id = int(r["cluster_id"]) if r.get("cluster_id") not in ("", None) else None
        o.umap_x, o.umap_y = _float(r.get("umap_x", "")), _float(r.get("umap_y", ""))
        outcomes.append(o)
    if unresolved:
        log.warning("%d retrieved chunk ids didn't match any KB chunk; treated as empty text", unresolved)
    return outcomes


def _load_clusters(probes_csv: Path, outcomes: List[ProbeOutcome]) -> List[ClusterCoverage]:
    clusters = []
    clusters_csv = probes_csv.with_suffix(".clusters.csv")
    for c in csv.DictReader(open(clusters_csv, newline="", encoding="utf-8")):
        cid = int(c["cluster_id"])
        members = [o for o in outcomes if o.cluster_id == cid and o.query_embedding is not None]
        xs = [o.umap_x for o in members if o.umap_x is not None]
        ys = [o.umap_y for o in members if o.umap_y is not None]
        clusters.append(ClusterCoverage(
            cluster_id=cid, zone=c["zone"], mean_cs=float(c["mean_cs"]), std_cs=float(c["std_cs"]),
            query_count=int(c["query_count"]), severity=float(c["severity"]),
            centroid_emb=np.mean([o.query_embedding for o in members], axis=0),
            centroid_x=float(np.mean(xs)) if xs else 0.0, centroid_y=float(np.mean(ys)) if ys else 0.0,
            representative_queries=json.loads(c["representative_queries"]), name=c["name"] or None,
            strategy_mix=_mix(json.loads(c["strategy_mix"])), purity=_float(c.get("purity", "")),
        ))
    return clusters
