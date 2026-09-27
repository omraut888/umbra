"""SQLAlchemy models for audit storage (spec §8).

Phase 1 tables: audit_runs and probe_results (one row per probe, holding the
probe query and its scores). umap_x / umap_y / cluster_id are created now but
stay NULL until Phase 2 clustering fills them in. cluster_summaries and
kb_health_history belong to Phases 2 and 4.

Columns beyond the spec, all needed to reproduce a Phase 1 audit:
    audit_runs.endpoint_url, audit_runs.kb_path   what was audited
    probe_results.hp_computed      whether HP was computed (preliminary score in the HP band)
    probe_results.preliminary_score  the RC+SE-only score used for the HP gate
    probe_results.probe_topic      taxonomy topic the probe was generated from
    probe_results.answer, .error   the RAG system's answer, or the query failure
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
