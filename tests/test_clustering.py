import json
import os
import math
import uuid
from pathlib import Path

import numpy as np
import pytest

from src.audit import ProbeOutcome, cluster_outcomes
from src.clustering.hdbscan_clusterer import NOISE, identify_clusters
from src.clustering.naming import NOISE_NAME, name_clusters
from src.clustering.umap_projector import project_for_clustering, project_to_2d
from src.clustering.zones import (
    SPEC_THRESHOLDS,
    ZoneThresholds,
    classify_zone,
    compute_cluster_coverage,
    representative_indices,
    severity,
)
from src.connectors.base import RAGResponse
from src.embeddings import embed
from src.probe_generation.taxonomy import Probe
from src.scoring.composite import CoverageScore

VALIDATION_PROBES = Path(__file__).resolve().parents[1] / "data/validation/validation_probes.jsonl"


def blobs(sizes, dim=384, spread=0.05, seed=0):
    rng = np.random.default_rng(seed)
    points, labels = [], []
    for k, n in enumerate(sizes):
        center = rng.normal(size=dim)
        pts = center + spread * np.linalg.norm(center) * rng.normal(size=(n, dim)) / math.sqrt(dim)
        points.append(pts)
        labels += [k] * n
    x = np.vstack(points)
    return x / np.linalg.norm(x, axis=1, keepdims=True), np.array(labels)


@pytest.fixture(scope="module")
def three_blobs():
    return blobs([80, 60, 40])


def test_projections_have_the_right_shape_and_are_deterministic(three_blobs):
    x, _ = three_blobs
    a10, b10 = project_for_clustering(x), project_for_clustering(x)
    a2 = project_to_2d(x)
    assert a10.shape == (180, 10) and a2.shape == (180, 2)
    np.testing.assert_allclose(a10, b10)


def test_hdbscan_recovers_blobs_and_numbers_by_size(three_blobs):
    x, truth = three_blobs
    c = identify_clusters(project_for_clustering(x))
    assert c.n_clusters == 3
    # cluster 0 is the largest blob, 2 the smallest
    for k in range(3):
        members = c.labels[truth == k]
        assert (members == k).mean() > 0.9
    assert c.soft_membership is not None and c.soft_membership.shape[0] == 180


def test_cluster_cap_sends_the_smallest_clusters_to_noise(three_blobs):
    x, truth = three_blobs
    c = identify_clusters(project_for_clustering(x), max_clusters=2)
    assert c.n_clusters == 2
    assert (c.labels[truth == 2] == NOISE).all()


def test_clusters_under_min_share_are_merged_into_noise(three_blobs):
    x, truth = three_blobs
    # 40/180 = 22% < 25% floor, so the smallest blob is folded into noise
    c = identify_clusters(project_for_clustering(x), min_share=0.25)
    assert c.n_clusters == 2 and (c.labels[truth == 2] == NOISE).all()


@pytest.mark.parametrize("mean,zone", [(0.0, "DARK"), (0.299, "DARK"), (0.30, "THIN"), (0.60, "THIN"), (0.601, "ADEQUATE")])
def test_spec_zone_thresholds(mean, zone):
    assert classify_zone(mean, SPEC_THRESHOLDS) == zone


@pytest.mark.parametrize("mean,zone", [(0.30, "DARK"), (0.333, "DARK"), (0.334, "THIN"), (0.408, "THIN"), (0.45, "ADEQUATE")])
def test_calibrated_default_thresholds(mean, zone):
    assert classify_zone(mean) == zone


def test_thresholds_must_be_ordered():
    with pytest.raises(ValueError):
        ZoneThresholds(0.5, 0.4)


def test_severity_formula_and_ordering():
    assert severity(0.1, 0.0, 500, size_cap=None) == pytest.approx(0.9 * math.log(501))
    assert severity(0.1, 0.0, 500) == pytest.approx(0.9 * math.log(51))
    # spec §6 example: 500 probes at 0.1 are more urgent than 10 averaging 0.05
    assert severity(0.1, 0.02, 500) > severity(0.05, 0.02, 10)
    # same mean and size: the more consistent cluster ranks higher
    assert severity(0.2, 0.05, 50) > severity(0.2, 0.3, 50)


def test_representative_indices_are_nearest_to_centroid():
    embs = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.8, 0.2]])
    assert representative_indices(embs, k=2) == [3, 1]


def test_compute_cluster_coverage():
    labels = np.array([0, 0, 0, 1, 1, NOISE])
    scores = [0.1, 0.2, 0.15, 0.8, 0.7, 0.5]
    embs = np.eye(6)
    coords = np.arange(12, dtype=float).reshape(6, 2)
    out = compute_cluster_coverage(labels, scores, embs, coords, [f"q{i}" for i in range(6)], SPEC_THRESHOLDS)
    by_id = {c.cluster_id: c for c in out}
    assert by_id[0].zone == "DARK" and by_id[1].zone == "ADEQUATE" and by_id[NOISE].zone == "THIN"
    assert by_id[0].query_count == 3 and by_id[0].mean_cs == pytest.approx(0.15)
    assert by_id[0].centroid_x == pytest.approx(2.0)
    assert out[0].cluster_id == 0  # most severe first
    assert [c.severity for c in out] == sorted((c.severity for c in out), reverse=True)


async def test_name_clusters_uses_representatives_and_skips_noise():
    labels = np.array([0, 0, NOISE])
    clusters = compute_cluster_coverage(labels, [0.2, 0.3, 0.9], np.eye(3), np.zeros((3, 2)), ["a?", "b?", "c?"])
    prompts = []

    async def fake(prompt):
        prompts.append(prompt)
        return '"Compost Pile Heat".\nextra line'

    await name_clusters(clusters, fake)
    names = {c.cluster_id: c.name for c in clusters}
    assert names == {0: "Compost Pile Heat", NOISE: NOISE_NAME}
    assert len(prompts) == 1 and "- a?" in prompts[0] and "c?" not in prompts[0]


async def test_name_clusters_without_llm_uses_central_query():
    labels = np.array([0, 0, NOISE])
    clusters = compute_cluster_coverage(labels, [0.2, 0.3, 0.9], np.eye(3), np.zeros((3, 2)), ["a?", "b" * 200, "c?"])
    await name_clusters(clusters, None)
    names = {c.cluster_id: c.name for c in clusters}
    assert names[NOISE] == NOISE_NAME
    assert names[0].startswith("cluster 0: ") and len(names[0]) <= 100


def test_cluster_outcomes_on_real_probe_embeddings():
    rows = [json.loads(line) for line in VALIDATION_PROBES.read_text().splitlines()]
    tier_score = {"composting": 0.65, "tomato_growing": 0.65, "hydroponics": 0.35,
                  "mushroom_cultivation": 0.35, "cryptocurrency_taxation": 0.2, "orbital_mechanics": 0.2}
    embs = embed([r["query"] for r in rows])
    outcomes = [
        ProbeOutcome(uuid.uuid4(), Probe(r["query"], "ground_truth", r["topic"]), RAGResponse(r["query"], []),
                     CoverageScore(tier_score[r["topic"]], 0, 0, None, 0), e)
        for r, e in zip(rows, embs)
    ]
    outcomes.append(ProbeOutcome(uuid.uuid4(), Probe("failed?"), RAGResponse("failed?", [], error="boom"), None))

    clusters = cluster_outcomes(outcomes)
    assert outcomes[-1].cluster_id is None  # failed query: not clustered
    scored = outcomes[:-1]
    assert all(o.cluster_id is not None and len(o.umap_10d) == 10 and o.umap_x is not None for o in scored)
    assert sum(c.query_count for c in clusters) == len(scored)
    assert all(c.strategy_mix == {"ground_truth": c.query_count} for c in clusters)
    # The out-of-domain topics should each get a cluster of their own. (In-domain
    # ones can legitimately merge: tomato and hydroponics questions end up together.)
    for absent in ("cryptocurrency_taxation", "orbital_mechanics"):
        ids = {o.cluster_id for o in scored if o.probe.topic == absent}
        assert len(ids) == 1 and NOISE not in ids
        members = [o.probe.topic for o in scored if o.cluster_id in ids]
        assert set(members) == {absent}


@pytest.mark.postgres
@pytest.mark.skipif(not os.environ.get("POSTGRES_DSN"), reason="POSTGRES_DSN not set")
def test_clusters_and_umap_coordinates_persist():
    from sqlalchemy import create_engine, text

    from src.audit import persist
    from src.scoring.scorer import ScorerConfig

    embs = embed([f"question {i} about {w}" for i, w in enumerate(["compost", "tomato", "bitcoin"] * 20)])
    outcomes = [
        ProbeOutcome(uuid.uuid4(), Probe(f"q{i}?"), RAGResponse(f"q{i}?", []), CoverageScore(0.1 * (i % 3), 0, 0, None, 0), e)
        for i, e in enumerate(embs)
    ]
    clusters = cluster_outcomes(outcomes, min_cluster_size=5)
    for c in clusters:
        c.name = f"cluster {c.cluster_id}"
    report_id = persist(os.environ["POSTGRES_DSN"], outcomes, kb_fingerprint="x", endpoint_url="e", kb_path="k",
                        config=ScorerConfig(), clusters=clusters)

    engine = create_engine(os.environ["POSTGRES_DSN"])
    try:
        with engine.connect() as conn:
            run = conn.execute(text("SELECT cluster_count FROM audit_runs WHERE report_id = :r"), {"r": report_id}).one()
            rows = conn.execute(text(
                "SELECT cluster_id, query_count, vector_dims(centroid_emb), strategy_mix FROM cluster_summaries "
                "WHERE report_id = :r"), {"r": report_id}).all()
            dims = conn.execute(text(
                "SELECT DISTINCT vector_dims(umap_10d) FROM probe_results WHERE report_id = :r"), {"r": report_id}).all()
            conn.execute(text("DELETE FROM audit_runs WHERE report_id = :r"), {"r": report_id})
            conn.commit()
    finally:
        engine.dispose()

    assert run.cluster_count == sum(not c.is_noise for c in clusters)
    assert sorted(r.cluster_id for r in rows) == sorted(c.cluster_id for c in clusters)
    assert sum(r.query_count for r in rows) == 60 and all(r[2] == 384 for r in rows)
    assert dims == [(10,)]


def test_label_purity_uses_all_probes_as_denominator():
    from src.clustering.zones import label_purity

    assert label_purity(["a", "a", "b", None]) == 0.5
    assert label_purity([None, None]) is None


def test_purity_rule_only_splits_dark_from_thin():
    from src.clustering.zones import PurityRule

    th = ZoneThresholds(0.324, 0.400)
    rule = PurityRule(0.6, dark_if_above=True)
    assert classify_zone(0.45, th, purity=0.9, purity_rule=rule) == "ADEQUATE"  # score decides adequate
    assert classify_zone(0.38, th, purity=0.9, purity_rule=rule) == "DARK"  # score alone would say THIN
    assert classify_zone(0.30, th, purity=0.2, purity_rule=rule) == "THIN"  # score alone would say DARK
    assert classify_zone(0.30, th, purity=None, purity_rule=rule) == "DARK"  # no labels: fall back to score
    assert PurityRule(0.6, dark_if_above=False).is_dark(0.5)


def test_severity_cap_keeps_big_covered_clusters_below_small_depth_gaps():
    # the phase 2 numbers: compost pile temperature vs fungal disease / pest control
    compost = severity(0.579, 0.128, 134)
    assert severity(0.432, 0.125, 28) > compost
    assert severity(0.444, 0.131, 47) > compost
    # uncapped, the spec formula gets this wrong
    assert severity(0.432, 0.125, 28, size_cap=None) < severity(0.579, 0.128, 134, size_cap=None)
