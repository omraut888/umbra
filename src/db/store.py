"""Persist audit results to Postgres."""

from __future__ import annotations

import uuid
from typing import Iterable

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from src.db.models import AuditRun, ClusterSummary, ProbeResult


def save_audit(
    dsn: str, run: AuditRun, probes: Iterable[ProbeResult], clusters: Iterable[ClusterSummary] = ()
) -> uuid.UUID:
    if run.report_id is None:
        run.report_id = uuid.uuid4()
    report_id = run.report_id  # read before commit expires the instance
    engine = create_engine(dsn)
    try:
        with Session(engine) as session, session.begin():
            run.probes = list(probes)
            run.clusters = list(clusters)
            session.add(run)
        return report_id
    finally:
        engine.dispose()
