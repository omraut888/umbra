import asyncio
import json
import uuid

import pandas as pd
import pytest

from src.dashboard.figures import THEMES, coverage_map, severity_bars, strategy_breakdown


def clusters():
    return [
        {"cluster_id": 1, "name": "crypto", "zone": "DARK", "severity": 2.7, "query_count": 40, "mean_cs": 0.23,
         "unanswered_share": 1.0, "centroid_x": 0.0, "centroid_y": 0.0},
        {"cluster_id": 2, "name": "pest control", "zone": "ADEQUATE", "severity": 1.9, "query_count": 47, "mean_cs": 0.44,
         "unanswered_share": 0.62, "centroid_x": 1.0, "centroid_y": 1.0},
        {"cluster_id": -1, "name": "noise (unclustered)", "zone": "ADEQUATE", "severity": 1.8, "query_count": 10,
         "mean_cs": 0.45, "unanswered_share": 0.6, "centroid_x": 0.5, "centroid_y": 0.5},
    ]


def points():
    rows = []
    for cid, zone in ((1, "DARK"), (2, "ADEQUATE"), (-1, "ADEQUATE")):
        for i in range(3):
            rows.append({"x": cid + i * 0.1, "y": i, "coverage_score": 0.3, "query": f"q{cid}{i}?", "cluster_id": cid,
                         "cluster_name": str(cid), "zone": zone, "strategy": "kb_blind"})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_map_colors_by_zone_with_shapes_and_labels_only_dark_thin_or_selected(theme):
    fig = coverage_map(points(), clusters(), theme)
    by_name = {t.name: t for t in fig.data}
    assert by_name["dark"].marker.color == THEMES[theme]["zone"]["DARK"] and by_name["dark"].marker.symbol == "diamond"
    assert by_name["noise (unclustered)"].marker.color == THEMES[theme]["noise"]
    assert [a.text for a in fig.layout.annotations] == ["crypto"]
    fig = coverage_map(points(), clusters(), theme, selected=2)
    assert [a.text for a in fig.layout.annotations] == ["crypto", "pest control"]


def test_map_score_view_uses_one_sequential_trace():
    fig = coverage_map(points(), clusters(), color_by="score")
    probes = [t for t in fig.data if t.name == "probes"]
    assert len(probes) == 1 and probes[0].marker.colorscale is not None


def test_severity_bars_keep_adequate_clusters_and_skip_noise():
    fig = severity_bars(clusters())
    names = [n for t in fig.data for n in t.y]
    assert "pest control" in names and "noise (unclustered)" not in names
    assert list(fig.layout.yaxis.categoryarray) == ["pest control", "crypto"]  # most severe on top


def test_strategy_breakdown_shares_sum_to_one():
    fig = strategy_breakdown({"kb_blind": {"DARK": 3, "ADEQUATE": 1}, "taxonomy": {"ADEQUATE": 4}})
    for i, strategy in enumerate(fig.data[0].y):
        assert sum(t.x[i] for t in fig.data) == pytest.approx(1.0)
    assert list(fig.data[0].y)[-1] == "kb_blind"  # most dark share drawn on top


def test_app_builds_from_a_report(tmp_path):
    from src.audit import ProbeOutcome, write_cluster_csv, write_csv
    from src.clustering.zones import ClusterCoverage
    from src.connectors.base import RAGResponse
    from src.dashboard.app import create_app, detail_panel
    from src.data import synthetic_kb_builder
    from src.data.kb_loader import load_chunks
    from src.probe_generation.taxonomy import Probe
    from src.reports.builder import build_report
    from src.reports.load import load_audit
    from src.scoring.composite import CoverageScore

    import numpy as np

    outs = []
    for i in range(6):
        o = ProbeOutcome(uuid.uuid4(), Probe(f"question {i}?", "kb_blind", "t"), RAGResponse(f"question {i}?", []),
                         CoverageScore(0.2 if i < 3 else 0.6, 0.2, 0.3, None, 0.2), np.eye(384)[i % 2])
        o.cluster_id, o.umap_x, o.umap_y, o.umap_10d = (0 if i < 3 else 1), float(i), float(i), [0.0] * 10
        outs.append(o)
    cl = [ClusterCoverage(k, "DARK", 0.2 if k == 0 else 0.6, 0.0, 3, 1.0, np.eye(384)[k], name=f"c{k}",
                          representative_queries=["question 0?"], strategy_mix={"kb_blind": 3}) for k in (0, 1)]
    audit_csv = tmp_path / "a.csv"
    write_csv(outs, audit_csv)
    write_cluster_csv(cl, audit_csv.with_suffix(".clusters.csv"))
    kb = tmp_path / "kb"
    synthetic_kb_builder.build(kb)
    chunks = load_chunks(kb)
    loaded = load_audit(audit_csv, chunks)
    report = asyncio.run(build_report(loaded.outcomes, loaded.clusters, chunks, config={"audit": str(audit_csv)}))
    report_path = tmp_path / "r.json"
    report_path.write_text(report.model_dump_json())

    app = create_app(report_path)
    assert app.layout is not None
    panel = detail_panel(report, 0)
    assert "c0" in json.dumps(panel.to_plotly_json(), default=str)
