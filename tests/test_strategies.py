import itertools
import json

import pytest

from src.data.kb_loader import Chunk
from src.probe_generation.adversarial import adversarial_boundary_generation, parse_topics
from src.probe_generation.counterfactual import counterfactual_generation
from src.probe_generation.strategies import generate_probe_set, split_budget

CHUNKS = [Chunk(f"doc{i}#0", f"doc{i}", f"Chunk number {i} about compost piles and browns.") for i in range(12)]

WORDS = ["apple", "river", "violin", "granite", "comet", "falcon", "tundra", "lantern", "quartz", "harbor",
         "cactus", "meteor", "saddle", "glacier", "orchid", "canyon", "piston", "walrus", "bamboo", "magnet"]


def distinct_question(i: int) -> str:
    # different words each time so these don't collapse in embedding dedup
    a, b, c = WORDS[i % 20], WORDS[(i * 7 + 3) % 20], WORDS[(i * 13 + 5) % 20]
    return f"How does the {a} relate to the {b} and {c} in case {i}?"


def test_split_budget_renormalizes_spec_shares():
    assert split_budget(900, ["taxonomy", "adversarial", "counterfactual"]) == {
        "taxonomy": 400, "adversarial": 300, "counterfactual": 200}
    assert split_budget(100, ["adversarial", "counterfactual"]) == {"adversarial": 60, "counterfactual": 40}
    assert sum(split_budget(1001, ["taxonomy", "adversarial", "counterfactual"]).values()) == 1001
    assert split_budget(50, ["counterfactual"]) == {"counterfactual": 50}


def test_split_budget_rejects_unknown():
    with pytest.raises(ValueError):
        split_budget(10, ["taxonomy", "userpattern"])


def test_parse_topics():
    assert parse_topics("# Boundary Topics\n1. Beekeeping\n- **Soil microbiology**\nHere are topics:\n\nx\n"
                        "## Also\nGreenhouse heating") == ["Beekeeping", "Soil microbiology", "Greenhouse heating"]


async def test_adversarial_runs_hyde_pipeline():
    counter = itertools.count()
    calls = {"topics": 0, "doc": 0, "questions": 0}

    async def fake(prompt: str) -> str:
        if "boundary topics" in prompt:
            calls["topics"] += 1
            return "\n".join(f"Adjacent topic {next(counter)}" for _ in range(4))
        if prompt.startswith("Write a short factual reference passage"):
            calls["doc"] += 1
            return "A hypothetical passage."
        calls["questions"] += 1
        assert "A hypothetical passage." in prompt  # questions come from the hypothetical doc
        return "\n".join(distinct_question(next(counter)) for _ in range(3))

    probes = await adversarial_boundary_generation(CHUNKS, 10, fake, topics_per_round=4)
    assert len(probes) == 10
    assert all(p.strategy == "adversarial" and p.topic.startswith("Adjacent topic") for p in probes)
    assert calls["doc"] == calls["questions"] >= 4


async def test_adversarial_stops_when_model_keeps_repeating_topics():
    async def fake(prompt: str) -> str:
        if "boundary topics" in prompt:
            return "Beekeeping"
        if prompt.startswith("Write"):
            return "doc"
        return "What is beekeeping?"

    probes = await adversarial_boundary_generation(CHUNKS, 50, fake, max_rounds=3)
    assert [p.query for p in probes] == ["What is beekeeping?"]


async def test_counterfactual_asks_two_per_chunk_and_tags_source_chunk():
    counter = itertools.count()
    seen_chunks = []

    async def fake(prompt: str) -> str:
        seen_chunks.append(prompt)
        return "\n".join(distinct_question(next(counter)) for _ in range(2))

    probes = await counterfactual_generation(CHUNKS, 30, fake)
    assert len(probes) == 30 and len(seen_chunks) == 15  # 12 chunks, then cycles
    assert all(p.strategy == "counterfactual" and p.topic.endswith("#0") for p in probes)


async def test_generate_probe_set_dedups_across_strategies():
    async def fake(prompt: str) -> str:
        if "boundary topics" in prompt:
            return "Topic A\nTopic B"
        if prompt.startswith("Write"):
            return "doc"
        # both strategies ask the same question -> must survive only once
        return "What is the ideal carbon to nitrogen ratio?\nHow wet should compost be?"

    extra = [pytest.importorskip("src.probe_generation.taxonomy").Probe("How wet should compost be?", "ground_truth")]
    probes, topics = await generate_probe_set(CHUNKS, 10, ["adversarial", "counterfactual"], fake, extra)
    assert topics is None
    assert sorted(p.query for p in probes) == ["How wet should compost be?", "What is the ideal carbon to nitrogen ratio?"]


def test_load_topics_file_formats(tmp_path):
    from src.probe_generation.kb_blind import TopicSpec, load_topics_file

    listy = tmp_path / "a.json"
    listy.write_text(json.dumps([{"name": "soil", "description": "Soil pH."}]))
    mapping = tmp_path / "b.json"
    mapping.write_text(json.dumps({"soil": "Soil pH."}))
    gt = tmp_path / "c.json"
    gt.write_text(json.dumps({"topics": {"hydroponics": {"tier": "thin", "description": "No soil."}},
                              "background_topics": {"soil": {"description": "Soil pH.", "documents": []}}}))
    assert load_topics_file(listy) == load_topics_file(mapping) == [TopicSpec("soil", "Soil pH.")]
    assert load_topics_file(gt) == [TopicSpec("hydroponics", "No soil."), TopicSpec("soil", "Soil pH.")]


async def test_enumerate_domain_topics_parses_name_colon_description():
    from src.probe_generation.kb_blind import enumerate_domain_topics

    async def fake(prompt):
        assert "home gardening" in prompt
        return "Here you go:\n1. **Composting**: Turning scraps into compost.\n- Irrigation: Watering systems.\nno colon here"

    topics = await enumerate_domain_topics("home gardening", fake)
    assert [t.name for t in topics] == ["Composting", "Irrigation"]
    assert topics[1].description == "Irrigation: Watering systems."


async def test_kb_blind_never_sees_kb_and_labels_by_topic():
    from src.probe_generation.kb_blind import TopicSpec, kb_blind_generation

    counter = itertools.count()
    prompts = []

    async def fake(prompt):
        prompts.append(prompt)
        return "\n".join(distinct_question(next(counter)) for _ in range(4))

    topics = [TopicSpec("soil", "Soil pH and texture."), TopicSpec("tax", "Crypto taxes.")]
    probes = await kb_blind_generation(topics, 6, fake)
    assert [p.topic for p in probes] == ["soil"] * 6 + ["tax"] * 6
    assert all(p.strategy == "kb_blind" for p in probes)
    assert not any("Chunk number" in p for p in prompts)


async def test_kb_blind_requires_topics():
    with pytest.raises(ValueError, match="kb_blind needs"):
        await generate_probe_set(CHUNKS, 10, ["kb_blind"], None)


def test_split_budget_ignores_kb_blind():
    assert split_budget(100, ["kb_blind", "counterfactual"]) == {"counterfactual": 100}
    assert split_budget(100, ["kb_blind"]) == {}
