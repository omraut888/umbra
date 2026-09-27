"""GapReport, the final output of an audit (spec §7), as pydantic models.

`python -m src.reports.schema` writes the JSON Schema to docs/gap_report.schema.json.

Beyond the spec's fields:
  clusters                 every cluster, ADEQUATE ones included, sorted by severity.
                           The tier answers "is this topic in the KB"; depth gaps
                           inside covered topics only show up through severity
                           (docs/findings.md sections 4 and 6), so they can't be
                           filtered out of the report.
  thresholds / severity_size_cap / config   what produced the tiers and ranking
  estimated_improvement_detail              how estimated_improvement was simulated
  strategy_breakdown       which probe strategies landed in which zones
"""

from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

SCHEMA_PATH = Path(__file__).resolve().parents[2] / "docs" / "gap_report.schema.json"

Zone = Literal["DARK", "THIN", "ADEQUATE"]


class DocumentRec(BaseModel):
    kind: Literal["kb_internal", "external"]
    cluster_id: int
    rank: int = Field(description="1 = best recommendation for this cluster")
    title: str
    url: Optional[str] = Field(None, description="external recommendations")
    doc_id: Optional[str] = Field(None, description="kb_internal recommendations: the KB document holding the passage")
    chunk_id: Optional[str] = None
    snippet: str = Field(description="the passage (kb_internal) or search snippet (external)")
    action: str = Field(description="what to do with it, in plain words")
    similarity: float = Field(description="cosine similarity of the snippet to the cluster's query centroid")
    expected_gain: float = Field(description="simulated change in the cluster's mean coverage score if the "
                                             "retriever returned this text for every probe in the cluster")
    search_query: Optional[str] = None
    retrieval_rate: Optional[float] = Field(None, description="kb_internal: share of the cluster's probes whose "
                                                              "retrieved context included this passage")
    answered_probes: Optional[int] = Field(None, description="kb_internal: how many of the cluster's probes the "
                                                             "cross-encoder says this passage answers")


class ClusterReport(BaseModel):
    cluster_id: int = Field(description="-1 is the HDBSCAN noise bucket")
    name: str
    zone: Zone
    mean_cs: float
    std_cs: float
    severity: float
    severity_rank: int
    query_count: int
    unanswered_share: float = Field(description="share of probes the cross-encoder found no answer for "
                                                "(HP >= 0.5, or preliminary score below the HP band)")
    purity: Optional[float] = None
    strategy_mix: Dict[str, int]
    centroid_x: float
    centroid_y: float
    representative_queries: List[str]
    sample_queries: List[str] = Field(description="spec §14: examples to sanity-check before acting on the zone")
    recommendation_reason: Optional[str] = Field(None, description="why this cluster got recommendations, "
                                                                    "or null if it didn't")
    recommendations: List[DocumentRec] = []


class Thresholds(BaseModel):
    dark_below: float
    adequate_above: float
    source: str


class ImprovementEstimate(BaseModel):
    overall_before: float
    overall_after: float
    recommendations_applied: int
    clusters_affected: List[int]
    method: str


class GapReport(BaseModel):
    report_id: uuid.UUID
    kb_fingerprint: Optional[str]
    probe_count: int
    cluster_count: int = Field(description="clusters excluding the noise bucket")
    overall_coverage_score: float
    dark_zones: List[ClusterReport] = Field(description="zone DARK, severity descending")
    thin_zones: List[ClusterReport] = Field(description="zone THIN, severity descending")
    clusters: List[ClusterReport] = Field(description="every cluster including ADEQUATE and noise, severity descending")
    recommendations: List[DocumentRec] = Field(description="all recommendations, highest-severity gap first")
    coverage_map_uri: Optional[str] = None
    estimated_improvement: float = Field(description="predicted overall_coverage_score after the top-3 "
                                                     "recommendations (simulated, see detail)")
    estimated_improvement_detail: ImprovementEstimate
    thresholds: Thresholds
    severity_size_cap: Optional[int]
    strategy_breakdown: Dict[str, Dict[str, int]] = Field(description="strategy -> zone -> probe count")
    config: Dict[str, object] = {}
    generated_at: datetime


def write_schema(path: Path = SCHEMA_PATH) -> None:
    path.write_text(json.dumps(GapReport.model_json_schema(), indent=2) + "\n")


if __name__ == "__main__":
    write_schema(Path(sys.argv[1]) if len(sys.argv) > 1 else SCHEMA_PATH)
    print(f"wrote {SCHEMA_PATH if len(sys.argv) == 1 else sys.argv[1]}")
