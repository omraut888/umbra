from collections import Counter

from src.benchmark.gap_injection import N_ABSENT, N_THIN, UNITS, inject
from src.data.synthetic_kb_builder import ALL_DOCS


def test_injection_splits_units_into_tiers():
    inj = inject(0)
    assert sorted(inj.tiers) == sorted(UNITS)
    assert Counter(inj.tiers.values()) == {"absent": N_ABSENT, "thin": N_THIN, "present": len(UNITS) - N_ABSENT - N_THIN}


def test_absent_and_thin_docs_are_removed_and_thin_passage_is_buried_once():
    inj = inject(3)
    kept_topics = {d.topic for d in inj.docs}
    for unit, tier in inj.tiers.items():
        assert (unit in kept_topics) == (tier == "present")
    for unit, info in inj.thin_passages.items():
        hits = [d for d in inj.docs if info["passage"] in d.body]
        assert [d.doc_id for d in hits] == [info["host"]]
        assert inj.tiers[hits[0].topic] == "present"
        assert len(info["passage"].split(". ")) <= 2


def test_injection_is_deterministic_per_seed_and_varies_across_seeds():
    assert inject(1).tiers == inject(1).tiers
    assert len({tuple(sorted(inject(s).tiers.items())) for s in range(5)}) == 5


def test_non_unit_docs_are_untouched():
    # the builder's other docs (none today) and every present doc keep their text,
    # apart from the hosts that received a thin passage
    inj = inject(2)
    hosts = {i["host"] for i in inj.thin_passages.values()}
    original = {d.doc_id: d.body for d in ALL_DOCS}
    for d in inj.docs:
        if d.doc_id not in hosts:
            assert d.body == original[d.doc_id]


def test_fit_threshold_lands_in_the_gap_between_classes():
    from src.benchmark.calibrate import fit_threshold, prf

    scores = [0.10, 0.20, 0.25, 0.40, 0.45, 0.60]
    absent = [True, True, True, False, False, False]
    t = fit_threshold(scores, absent, below=True)
    assert 0.25 < t < 0.40 and prf(scores, absent, t, below=True) == (1.0, 1.0, 1.0)


def test_fit_threshold_trades_off_when_classes_overlap():
    from src.benchmark.calibrate import fit_threshold, prf

    scores = [0.10, 0.30, 0.35, 0.32, 0.50, 0.60]
    present = [False, False, True, False, True, True]
    t = fit_threshold(scores, present, below=False)
    p, r, f1 = prf(scores, present, t, below=False)
    assert f1 == max(prf(scores, present, c, below=False)[2] for c in (0.2, 0.31, 0.325, 0.34, 0.4, 0.55))


def test_fit_orders_thresholds():
    from src.benchmark.calibrate import fit

    clusters = [{"mean_cs": s, "tier": t} for s, t in
                [(0.2, "absent"), (0.25, "absent"), (0.35, "thin"), (0.4, "thin"), (0.6, "present"), (0.65, "present")]]
    th = fit(clusters)
    assert 0.25 < th.dark_below < 0.35 and 0.4 < th.adequate_above < 0.6


def test_fit_purity_rule_learns_direction():
    from src.benchmark.calibrate import fit_purity_rule
    from src.clustering.zones import ZoneThresholds

    th = ZoneThresholds(0.324, 0.400)
    clusters = [{"mean_cs": 0.33, "tier": "absent", "p": x} for x in (0.2, 0.25, 0.3)] + \
               [{"mean_cs": 0.33, "tier": "thin", "p": x} for x in (0.7, 0.8, 0.9)] + \
               [{"mean_cs": 0.6, "tier": "present", "p": 0.1}]  # adequate by score: ignored
    rule = fit_purity_rule(clusters, th, "p")
    assert rule.dark_if_above is False and 0.3 < rule.threshold < 0.7


def test_inject_respects_configured_tier_counts():
    import pytest as _pytest

    inj = inject(0, n_absent=2, n_thin=1)
    assert Counter(inj.tiers.values()) == {"absent": 2, "thin": 1, "present": len(UNITS) - 3}
    with _pytest.raises(ValueError):
        inject(0, n_absent=10, n_thin=4)


async def test_run_benchmark_skips_finished_seeds_and_flags_config_changes(tmp_path):
    import json

    from src.benchmark.gap_injection import BenchmarkConfig, run_benchmark

    seed_dir = tmp_path / "seed_7"
    seed_dir.mkdir()
    (seed_dir / "result.json").write_text(json.dumps({"seed": 7, "config": {"n_probes": 1}}))
    messages = []

    async def never(prompt):
        raise AssertionError("finished seeds must not call the model")

    out = await run_benchmark([7], tmp_path, BenchmarkConfig(), complete=never, echo=messages.append)
    assert out[0]["seed"] == 7 and "different config" in messages[0]


def test_spec10_check_and_thresholds_file(tmp_path):
    import json

    from src.benchmark.calibrate import spec10_check, write_thresholds
    from src.cli import _thresholds

    s = {"calibrated": [0.33, 0.41], "n_seeds": 2, "n_labeled": 9,
         "comparison": {"calibrated": {"spec10_per_seed": {0: {"precision": 0.9, "recall": 0.8, "fpr": 0.2},
                                                           1: {"precision": 0.7, "recall": 0.6, "fpr": 0.0}}}}}
    check = spec10_check(s)
    assert check["precision"]["verdict"] == "target" and check["recall"]["verdict"] == "below minimum"
    assert check["fpr"]["verdict"] == "target"
    path = tmp_path / "t.json"
    write_thresholds(s, path)
    th, source = _thresholds("0.3,0.6", path)
    assert (th.dark_below, th.adequate_above) == (0.33, 0.41) and "2 gap-injection seeds" in source
    assert json.loads(path.read_text())["fit_on"].startswith("2 ")
