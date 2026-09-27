"""cluster_summaries and umap_10d

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-26
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("probe_results", sa.Column("umap_10d", Vector(10)))
    op.create_table(
        "cluster_summaries",
        sa.Column("cluster_id", sa.Integer, primary_key=True),
        sa.Column(
            "report_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("audit_runs.report_id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("cluster_name", sa.String(100)),
        sa.Column("zone", sa.String(10)),
        sa.Column("mean_cs", sa.Float),
        sa.Column("std_cs", sa.Float),
        sa.Column("severity", sa.Float),
        sa.Column("query_count", sa.Integer),
        sa.Column("centroid_emb", Vector(384)),
        sa.Column("centroid_x", sa.Float),
        sa.Column("centroid_y", sa.Float),
        sa.Column("strategy_mix", postgresql.JSONB),
        sa.Column("representative_queries", postgresql.JSONB),
    )


def downgrade() -> None:
    op.drop_table("cluster_summaries")
    op.drop_column("probe_results", "umap_10d")
