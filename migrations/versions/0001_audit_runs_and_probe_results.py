"""audit_runs and probe_results (Phase 1)

Revision ID: 0001
Revises:
Create Date: 2026-09-26
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "audit_runs",
        sa.Column("report_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("kb_fingerprint", sa.String(64)),
        sa.Column("probe_count", sa.Integer),
        sa.Column("cluster_count", sa.Integer),
        sa.Column("overall_score", sa.Float),
        sa.Column("config", postgresql.JSONB),
        sa.Column("endpoint_url", sa.Text),
        sa.Column("kb_path", sa.Text),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    op.create_table(
        "probe_results",
        sa.Column("probe_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "report_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("audit_runs.report_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("query_text", sa.Text, nullable=False),
        sa.Column("query_embedding", Vector(384)),
        sa.Column("umap_x", sa.Float),
        sa.Column("umap_y", sa.Float),
        sa.Column("cluster_id", sa.Integer),
        sa.Column("coverage_score", sa.Float),
        sa.Column("rc_score", sa.Float),
        sa.Column("se_score", sa.Float),
        sa.Column("hp_score", sa.Float),
        sa.Column("generation_strategy", sa.String(30)),
        sa.Column("retrieved_chunk_ids", postgresql.JSONB),
        sa.Column("hp_computed", sa.Boolean),
        sa.Column("preliminary_score", sa.Float),
        sa.Column("probe_topic", sa.Text),
        sa.Column("answer", sa.Text),
        sa.Column("error", sa.Text),
    )
    op.create_index("ix_probe_results_report_id", "probe_results", ["report_id"])


def downgrade() -> None:
    op.drop_index("ix_probe_results_report_id", table_name="probe_results")
    op.drop_table("probe_results")
    op.drop_table("audit_runs")
    # The vector extension is left installed: other schemas may depend on it.
