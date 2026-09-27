"""SQLAlchemy models for audit storage (spec §8).

audit_runs, probe_results (one row per probe: query, scores, cluster, UMAP
coordinates) and cluster_summaries. kb_health_history comes with monitoring.

Columns beyond the spec, all needed to reproduce an audit:
    audit_runs.endpoint_url, audit_runs.kb_path   what was audited
    probe_results.hp_computed      whether HP was computed (preliminary score in the HP band)
    probe_results.preliminary_score  the RC+SE-only score used for the HP gate
    probe_results.probe_topic      taxonomy topic the probe was generated from
    probe_results.answer, .error   the RAG system's answer, or the query failure
    probe_results.umap_10d         the clustering coordinates (umap_x/y are display only)
    cluster_summaries.centroid_x/y, .strategy_mix, .representative_queries
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import List, Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from src.embeddings import EMBEDDING_DIM


class Base(DeclarativeBase):
    pass


class AuditRun(Base):
    __tablename__ = "audit_runs"

    report_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    kb_fingerprint: Mapped[Optional[str]] = mapped_column(String(64))
    probe_count: Mapped[Optional[int]] = mapped_column(Integer)
    cluster_count: Mapped[Optional[int]] = mapped_column(Integer)
    overall_score: Mapped[Optional[float]] = mapped_column(Float)
    config: Mapped[Optional[dict]] = mapped_column(JSONB)  # weights α,β,γ, thresholds, SE method, ...
    endpoint_url: Mapped[Optional[str]] = mapped_column(Text)
    kb_path: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    probes: Mapped[List["ProbeResult"]] = relationship(back_populates="audit_run", cascade="all, delete-orphan")
    clusters: Mapped[List["ClusterSummary"]] = relationship(back_populates="audit_run", cascade="all, delete-orphan")


class ProbeResult(Base):
    __tablename__ = "probe_results"

    probe_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    report_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("audit_runs.report_id", ondelete="CASCADE"), index=True
    )
    query_text: Mapped[str] = mapped_column(Text)
    query_embedding = mapped_column(Vector(EMBEDDING_DIM), nullable=True)
    umap_x: Mapped[Optional[float]] = mapped_column(Float)
    umap_y: Mapped[Optional[float]] = mapped_column(Float)
    umap_10d = mapped_column(Vector(10), nullable=True)
    cluster_id: Mapped[Optional[int]] = mapped_column(Integer)
    coverage_score: Mapped[Optional[float]] = mapped_column(Float)
    rc_score: Mapped[Optional[float]] = mapped_column(Float)
    se_score: Mapped[Optional[float]] = mapped_column(Float)
    hp_score: Mapped[Optional[float]] = mapped_column(Float)
    generation_strategy: Mapped[Optional[str]] = mapped_column(String(30))  # taxonomy/adversarial/counterfactual/userpattern
    retrieved_chunk_ids: Mapped[Optional[list]] = mapped_column(JSONB)
    hp_computed: Mapped[Optional[bool]] = mapped_column(Boolean)
    preliminary_score: Mapped[Optional[float]] = mapped_column(Float)
    probe_topic: Mapped[Optional[str]] = mapped_column(Text)
    answer: Mapped[Optional[str]] = mapped_column(Text)
    error: Mapped[Optional[str]] = mapped_column(Text)

    audit_run: Mapped[AuditRun] = relationship(back_populates="probes")


class ClusterSummary(Base):
    __tablename__ = "cluster_summaries"

    cluster_id: Mapped[int] = mapped_column(Integer, primary_key=True)  # -1 = noise bucket
    report_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("audit_runs.report_id", ondelete="CASCADE"), primary_key=True
    )
    cluster_name: Mapped[Optional[str]] = mapped_column(String(100))
    zone: Mapped[Optional[str]] = mapped_column(String(10))
    mean_cs: Mapped[Optional[float]] = mapped_column(Float)
    std_cs: Mapped[Optional[float]] = mapped_column(Float)
    severity: Mapped[Optional[float]] = mapped_column(Float)
    query_count: Mapped[Optional[int]] = mapped_column(Integer)
    centroid_emb = mapped_column(Vector(EMBEDDING_DIM), nullable=True)
    centroid_x: Mapped[Optional[float]] = mapped_column(Float)
    centroid_y: Mapped[Optional[float]] = mapped_column(Float)
    strategy_mix: Mapped[Optional[dict]] = mapped_column(JSONB)
    representative_queries: Mapped[Optional[list]] = mapped_column(JSONB)

    audit_run: Mapped[AuditRun] = relationship(back_populates="clusters")
