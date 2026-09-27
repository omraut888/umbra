import math

import numpy as np
import pytest

from src.embeddings import embed, embed_one
from src.scoring import (
    ScoringWeights,
    composite_coverage_score,
    hallucination_probability,
    retrieval_confidence,
    semantic_entropy,
    semantic_entropy_from_embeddings,
)
from src.scoring.scorer import CoverageScorer, ScorerConfig
from tests.scoring_cases import CASES

GOOD = [c for c in CASES if c.label == "good"]
BAD = [c for c in CASES if c.label == "bad"]
BORDERLINE = [c for c in CASES if c.label == "borderline"]


class TestRetrievalConfidence:
    def test_identical_vector_is_one(self):
        v = np.array([0.3, 0.4, 0.5])
        assert retrieval_confidence(v, [v]) == pytest.approx(1.0)

    def test_takes_max_over_chunks(self):
        q = np.array([1.0, 0.0])
        chunks = [[0.0, 1.0], [1.0, 1.0], [1.0, 0.1]]
        assert retrieval_confidence(q, chunks) == pytest.approx(1 / math.sqrt(1.01))

    def test_orthogonal_is_zero_and_negative_clipped(self):
        q = np.array([1.0, 0.0])
        assert retrieval_confidence(q, [[0.0, 1.0]]) == pytest.approx(0.0)
        assert retrieval_confidence(q, [[-1.0, 0.0]]) == 0.0

    def test_unnormalized_inputs(self):
        assert retrieval_confidence([2.0, 0.0], [[5.0, 0.0]]) == pytest.approx(1.0)

    def test_empty_and_zero_vectors(self):
        assert retrieval_confidence([1.0, 0.0], []) == 0.0
        assert retrieval_confidence([0.0, 0.0], [[1.0, 0.0]]) == 0.0
        assert retrieval_confidence([1.0, 0.0], [[0.0, 0.0], [1.0, 0.0]]) == pytest.approx(1.0)


class TestSemanticEntropy:
    @pytest.mark.parametrize("method", ["dispersion", "spec"])
    def test_fewer_than_two_chunks_is_zero(self, method):
        assert semantic_entropy_from_embeddings(np.zeros((0, 3)), method=method) == 0.0
        assert semantic_entropy_from_embeddings(np.ones((1, 3)), method=method) == 0.0
        assert semantic_entropy([], method=method) == 0.0
        assert semantic_entropy(["only one"], method=method) == 0.0

    @pytest.mark.parametrize("method", ["dispersion", "spec"])
    def test_identical_chunks_have_zero_entropy(self, method):
        embs = np.tile([0.2, 0.5, 0.1], (5, 1))
        assert semantic_entropy_from_embeddings(embs, method=method) == pytest.approx(0.0, abs=1e-6)

    def test_spec_is_normalized_entropy_of_distance_histogram(self):
        # 3 orthonormal vectors + their normalized sum: distances 1.0 (x3) and
        # 1 - 1/sqrt(3) = 0.4226 (x3) -> two equally-filled bins -> H = ln 2.
        e = np.eye(3)
        embs = np.vstack([e, e.sum(axis=0) / math.sqrt(3)])
        assert semantic_entropy_from_embeddings(embs, method="spec") == pytest.approx(
            math.log(2) / math.log(10), abs=1e-6
        )

    def test_dispersion_is_mean_pairwise_distance(self):
        assert semantic_entropy_from_embeddings(np.eye(4), method="dispersion") == pytest.approx(1.0)

    def test_unknown_method_rejected(self):
        with pytest.raises(ValueError):
            semantic_entropy_from_embeddings(np.eye(3), method="nope")

    @pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
    @pytest.mark.parametrize("method", ["dispersion", "spec"])
    def test_in_unit_range(self, case, method):
        se = semantic_entropy(case.chunks, method=method)
        assert 0.0 <= se <= 1.0

    def test_dispersion_flags_one_relevant_among_noise(self):
        """Unrelated chunks are more diverse than on-topic chunks."""
        noisy = [c for c in BORDERLINE if "among_noise" in c.name]
        focused_max = max(semantic_entropy(c.chunks, method="dispersion") for c in GOOD)
        for c in noisy:
            assert semantic_entropy(c.chunks, method="dispersion") > focused_max

    def test_spec_formula_rates_noise_as_focused(self):
        # The spec formula's failure (docs/findings.md): mutually unrelated chunks
        # get *lower* entropy than on-topic ones. Pinned so the comparison stays honest.
        noisy = [c for c in BORDERLINE if "among_noise" in c.name]
        focused_min = min(semantic_entropy(c.chunks, method="spec") for c in GOOD)
        for c in noisy:
            assert semantic_entropy(c.chunks, method="spec") < focused_min


class TestHallucinationProbability:
    def test_empty_retrieval_is_certain_hallucination(self):
        assert hallucination_probability("anything", []) == 1.0

    def test_uses_max_logit_through_sigmoid(self):
        class FakeCE:
            def predict(self, pairs, **_):
                return np.array([-3.0, 2.0, 0.5])[: len(pairs)]

        hp = hallucination_probability("q", ["a", "b", "c"], cross_encoder=FakeCE())
        assert hp == pytest.approx(1 - 1 / (1 + math.exp(-2.0)))

    @pytest.mark.parametrize("case", GOOD, ids=lambda c: c.name)
    def test_low_for_good(self, case):
        assert hallucination_probability(case.query, case.chunks) < 0.1

    @pytest.mark.parametrize("case", [c for c in BAD if c.chunks], ids=lambda c: c.name)
    def test_high_for_bad(self, case):
        assert hallucination_probability(case.query, case.chunks) > 0.9


class TestComposite:
    def test_full_formula_when_hp_in_band(self):
        w = ScoringWeights()
        rc, se, hp = 0.5, 0.5, 0.2
        prelim = (0.4 * 0.5 + 0.35 * 0.5) / 0.75
        assert 0.35 <= prelim <= 0.65
        res = composite_coverage_score(rc, se, hp, w)
        assert res.hp_computed
        assert res.score == pytest.approx(0.4 * 0.5 + 0.35 * 0.5 + 0.25 * 0.8)
        assert res.preliminary == pytest.approx(prelim)

    @pytest.mark.parametrize("rc,se", [(0.95, 0.1), (0.05, 0.95)])
    def test_hp_skipped_outside_band(self, rc, se):
        calls = []

        def hp():
            calls.append(1)
            return 0.5

        res = composite_coverage_score(rc, se, hp)
        assert calls == []
        assert not res.hp_computed
        assert res.score == pytest.approx(res.preliminary)
        assert not 0.35 <= res.preliminary <= 0.65

    def test_lazy_hp_called_exactly_once_in_band(self):
        calls = []

        def hp():
            calls.append(1)
            return 0.3

        res = composite_coverage_score(0.5, 0.5, hp)
        assert calls == [1] and res.hp == 0.3

    def test_band_edges_inclusive(self):
        # prelim = (0.4*rc + 0.35*(1-se)) / 0.75 ; rc=0.35, se=0.65 -> exactly 0.35
        assert composite_coverage_score(0.35, 0.65, 0.0).hp_computed
        assert composite_coverage_score(0.65, 0.35, 0.0).hp_computed

    def test_none_hp_returns_preliminary(self):
        res = composite_coverage_score(0.5, 0.5, None)
        assert res.hp is None and res.score == pytest.approx(res.preliminary)

    def test_custom_weights_and_band(self):
        w = ScoringWeights(0.5, 0.25, 0.25)
        res = composite_coverage_score(0.9, 0.1, 0.0, w, hp_band=(0.0, 1.0))
        assert res.score == pytest.approx(0.5 * 0.9 + 0.25 * 0.9 + 0.25 * 1.0)

    @pytest.mark.parametrize("weights", [(0.5, 0.5, 0.5), (-0.1, 0.6, 0.5), (0.0, 0.0, 1.0)])
    def test_invalid_weights(self, weights):
        with pytest.raises(ValueError):
            ScoringWeights(*weights)

    def test_parse_weights(self):
        assert ScoringWeights.parse("0.5,0.3,0.2") == ScoringWeights(0.5, 0.3, 0.2)

    @pytest.mark.parametrize("bad", [(-0.1, 0.5, 0.5), (0.5, 1.1, 0.5), (0.5, 0.5, 1.5)])
    def test_out_of_range_signals(self, bad):
        with pytest.raises(ValueError):
            composite_coverage_score(*bad, hp_band=(0.0, 1.0))

    def test_score_bounded(self):
        for rc in np.linspace(0, 1, 5):
            for se in np.linspace(0, 1, 5):
                for hp in np.linspace(0, 1, 5):
                    assert 0.0 <= composite_coverage_score(rc, se, hp).score <= 1.0


EXPECTED_ZONE = {
    "good": lambda s: s > 0.60,
    "bad": lambda s: s < 0.30,
    "borderline": lambda s: 0.30 <= s <= 0.60,
}

# Where the spec SE formula lands outside the expected zone (docs/findings.md):
# unrelated chunks get low entropy, so off-topic retrievals earn most of β·(1 − SE).
SPEC_SE_MISSES = {
    "bad_quantum_vs_baking",
    "bad_tax_vs_football",
    "bad_kubernetes_vs_poetry",
    "borderline_one_relevant_among_noise_insulin",
    "borderline_one_relevant_among_noise_wifi",
}


@pytest.fixture(scope="module")
def scores_by_method():
    out = {}
    for method in ("dispersion", "spec"):
        scorer = CoverageScorer(ScorerConfig(se_method=method))
        results, _ = scorer.score_batch([c.query for c in CASES], [c.chunks for c in CASES])
        out[method] = {c.name: r for c, r in zip(CASES, results)}
    return out


def _zone_params():
    for method in ("dispersion", "spec"):
        for case in CASES:
            marks = []
            if method == "spec" and case.name in SPEC_SE_MISSES:
                marks.append(pytest.mark.xfail(strict=True, reason="spec SE formula, see docs/findings.md"))
            yield pytest.param(method, case, marks=marks, id=f"{method}-{case.name}")


@pytest.mark.parametrize("method,case", list(_zone_params()))
def test_case_lands_in_expected_zone(scores_by_method, method, case):
    score = scores_by_method[method][case.name].score
    assert EXPECTED_ZONE[case.label](score), f"{case.label} case scored {score:.3f}"


@pytest.mark.parametrize("method", ["dispersion", "spec"])
def test_every_good_case_outscores_every_bad_case(scores_by_method, method):
    s = scores_by_method[method]
    assert min(s[c.name].score for c in GOOD) > max(s[c.name].score for c in BAD)


def test_empty_retrieval_scores_zero(scores_by_method):
    for method in scores_by_method:
        assert scores_by_method[method]["bad_empty_retrieval"].score == 0.0


def test_rc_separates_good_from_bad():
    for good in GOOD:
        rc_good = retrieval_confidence(embed_one(good.query), embed(good.chunks))
        for bad in BAD:
            rc_bad = retrieval_confidence(embed_one(bad.query), embed(bad.chunks))
            assert rc_good > rc_bad
