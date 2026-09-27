import json
import uuid
from types import SimpleNamespace

import numpy as np
import pytest

from src.audit import ProbeOutcome
from src.clustering.zones import ClusterCoverage
from src.connectors.base import RAGResponse, RetrievedChunk
from src.data import synthetic_kb_builder
from src.data.kb_loader import load_chunks
from src.probe_generation.taxonomy import Probe
from src.reports.builder import build_report, select_targets
from src.reports.recommend import KBInternalRecommender, SearchResult, external_candidates
from src.reports.schema import GapReport
from src.scoring.scorer import CoverageScorer

HYDRO_QUESTIONS = [
    "What is the Kratky method of hydroponics?",
    "Can I grow lettuce hydroponically without a pump?",
    "How do net cups work in a passive hydroponic jar?",
    "What do you need to grow basil in a nutrient solution?",
    "Is hydroponics possible without soil for herbs?",
]
CRYPTO_QUESTIONS = [
    "How are capital gains on Bitcoin taxed?",
    "Do I owe tax when I swap one token for another?",
    "Which form reports cryptocurrency sales?",
    "Is staking income taxable?",
    "How do I calculate cost basis for crypto?",
]


@pytest.fixture(scope="module")
def chunks(tmp_path_factory):
    out = tmp_path_factory.mktemp("kb")
    synthetic_kb_builder.build(out)
    return load_chunks(out)


def outcomes_for(questions, retrieved, scorer, cluster_id):
    by_id = {c.chunk_id: c.text for c in retrieved}
    ids = list(by_id)
    scores, embs = scorer.score_batch(questions, [list(by_id.values())] * len(questions))
    out = []
    for q, s, e in zip(questions, scores, embs):
        o = ProbeOutcome(uuid.uuid4(), Probe(q, "kb_blind", "t"),
                         RAGResponse(q, [RetrievedChunk(by_id[i], i) for i in ids]), s, e)
        o.cluster_id, o.umap_x, o.umap_y = cluster_id, 0.0, 0.0
        out.append(o)
    return out


def cluster_for(cid, outs, name):
    embs = np.vstack([o.query_embedding for o in outs])
    scores = [o.score.score for o in outs]
    return ClusterCoverage(cluster_id=cid, zone="THIN", mean_cs=float(np.mean(scores)), std_cs=float(np.std(scores)),
                           query_count=len(outs), severity=0.0, centroid_emb=embs.mean(axis=0),
                           representative_queries=[o.probe.query for o in outs[:3]], name=name,
                           strategy_mix={"kb_blind": len(outs)})


@pytest.fixture(scope="module")
def audit(chunks):
    scorer = CoverageScorer()
    # retrieval that never reaches the buried hydroponics passage
    unrelated = [c for c in chunks if c.doc_id.startswith("compost-")][:5]
    hydro = outcomes_for(HYDRO_QUESTIONS, unrelated, scorer, 0)
    crypto = outcomes_for(CRYPTO_QUESTIONS, unrelated, scorer, 1)
    return hydro + crypto, [cluster_for(0, hydro, "hydroponics"), cluster_for(1, crypto, "crypto taxes")]


def test_kb_internal_finds_the_buried_hydroponics_passage(chunks, audit):
    outcomes, clusters = audit
    rec = KBInternalRecommender(chunks)
    found = rec.candidates([o for o in outcomes if o.cluster_id == 0], clusters[0].centroid_emb)
    assert found and found[0].chunk_id == "seeds-01-indoor#1" and "Kratky" in found[0].snippet
    assert found[0].retrieval_rate == 0.0 and found[0].answered_probes >= 1


def test_kb_internal_recommends_nothing_for_an_absent_topic(chunks, audit):
    outcomes, clusters = audit
    rec = KBInternalRecommender(chunks)
    assert rec.candidates([o for o in outcomes if o.cluster_id == 1], clusters[1].centroid_emb) == []


def test_kb_internal_skips_passages_that_are_already_retrieved(chunks):
    scorer = CoverageScorer()
    host = next(c for c in chunks if c.chunk_id == "seeds-01-indoor#1")
    outs = outcomes_for(HYDRO_QUESTIONS, [host], scorer, 0)
    c = cluster_for(0, outs, "hydroponics")
    assert KBInternalRecommender(chunks).candidates(outs, c.centroid_emb) == []


class FakeSearch:
    def __init__(self):
        self.queries = []

    async def search(self, query, n):
        self.queries.append(query)
        return [
            SearchResult("Kratky method guide", f"https://example.org/kratky?q={len(self.queries)}",
                         "The Kratky method grows lettuce in a jar of nutrient solution with no pump."),
            SearchResult("Stock prices today", "https://example.org/stocks", "Markets closed higher."),
        ][:n]


async def test_external_candidates_rank_by_similarity_and_dedupe(audit):
    outcomes, clusters = audit
    search = FakeSearch()

    async def fake_complete(prompt):
        assert "What is the Kratky method" in prompt
        return "# Search queries\n1. kratky hydroponics\n2. passive hydroponics lettuce\n"

    found = await external_candidates([o for o in outcomes if o.cluster_id == 0], clusters[0].representative_queries,
                                      clusters[0].centroid_emb, fake_complete, search)
    assert search.queries == ["kratky hydroponics", "passive hydroponics lettuce"]
    assert found[0].title == "Kratky method guide" and found[0].search_query == "kratky hydroponics"
    assert len({c.url for c in found}) == len(found)


async def test_build_report_end_to_end(chunks, audit):
    outcomes, clusters = audit

    async def fake_complete(prompt):
        return "kratky hydroponics"

    report = await build_report(outcomes, clusters, chunks, complete=fake_complete, search=FakeSearch(), top_k=10)
    GapReport.model_validate_json(report.model_dump_json())  # round-trips through the schema

    assert [c.severity_rank for c in report.clusters] == [1, 2]
    assert [c.severity for c in report.clusters] == sorted((c.severity for c in report.clusters), reverse=True)
    hydro = next(c for c in report.clusters if c.cluster_id == 0)
    kinds = {r.kind for r in hydro.recommendations}
    assert "kb_internal" in kinds and "external" in kinds
    assert all(r.expected_gain > 0 for r in report.recommendations)
    assert [r.rank for r in hydro.recommendations] == list(range(1, len(hydro.recommendations) + 1))
    d = report.estimated_improvement_detail
    assert report.estimated_improvement == d.overall_after >= d.overall_before
    assert 0 in d.clusters_affected
    assert set(report.strategy_breakdown) == {"kb_blind"}
    assert len(hydro.sample_queries) == len(HYDRO_QUESTIONS)


async def test_a_failing_search_backend_doesnt_lose_the_report(chunks, audit):
    outcomes, clusters = audit

    class Broken:
        async def search(self, query, n):
            raise RuntimeError("rate limited")

    async def fake_complete(prompt):
        return "q"

    report = await build_report(outcomes, clusters, chunks, complete=fake_complete, search=Broken())
    hydro = next(c for c in report.clusters if c.cluster_id == 0)
    assert {r.kind for r in hydro.recommendations} == {"kb_internal"}


def test_select_targets_includes_severe_adequate_clusters_and_skips_noise():
    def c(cid, zone, sev):
        return ClusterCoverage(cid, zone, 0.5, 0.1, 30, sev, np.zeros(3))

    clusters = [c(0, "ADEQUATE", 1.9), c(1, "DARK", 1.0), c(2, "ADEQUATE", 0.5), c(-1, "ADEQUATE", 3.0)]
    assert select_targets(clusters, top_k=1) == {1: "DARK", 0: "top 1 by severity"}


async def test_anthropic_search_parses_results_and_citations(monkeypatch):
    from src.reports import search as search_mod

    response = SimpleNamespace(content=[
        SimpleNamespace(type="server_tool_use"),
        SimpleNamespace(type="web_search_tool_result", content=[
            SimpleNamespace(url="https://a.org", title="A"), SimpleNamespace(url="https://b.org", title="B")]),
        SimpleNamespace(type="text", text="See A.", citations=[
            SimpleNamespace(type="web_search_result_location", url="https://b.org", title="B", cited_text="B says x.")]),
    ])

    class FakeMessages:
        async def create(self, **kwargs):
            assert kwargs["tools"][0]["type"] == "web_search_20250305"
            return response

    provider = search_mod.AnthropicWebSearch.__new__(search_mod.AnthropicWebSearch)
    provider.client, provider.model = SimpleNamespace(messages=FakeMessages()), "m"
    results = await provider.search("q", 5)
    assert [r.url for r in results] == ["https://b.org", "https://a.org"]  # quoted results first
    assert results[0].snippet == "B says x." and results[1].snippet == "A"


def test_schema_file_is_current():
    from src.reports.schema import SCHEMA_PATH

    assert json.loads(SCHEMA_PATH.read_text()) == GapReport.model_json_schema()
