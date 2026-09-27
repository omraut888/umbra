import itertools
import json

import pytest

from src.data import synthetic_kb_builder
from src.data.kb_loader import load_chunks
from src.probe_generation.taxonomy import (
    Probe,
    Topic,
    allocate,
    build_prompt,
    dedup_probes,
    extract_taxonomy,
    generate_probes_for_topic,
    parse_questions,
    taxonomy_guided_generation,
)


def make_topic(name="0_compost_pile", keywords=("compost", "pile")) -> Topic:
    return Topic(topic_id=0, name=name, keywords=list(keywords), size=3, chunk_ids=["a#0"], doc_ids=["a"],
                 representative_texts=["Compost needs browns and greens."])


def test_parse_questions_strips_numbering_and_junk():
    text = """Here are your questions:
1. What is compost?
2) How hot should a pile get?
- Why does compost smell?
"Which worms are used in vermicomposting?"
Not a question
Q3: How often should I turn the pile?"""
    assert parse_questions(text) == [
        "What is compost?",
        "How hot should a pile get?",
        "Why does compost smell?",
        "Which worms are used in vermicomposting?",
        "How often should I turn the pile?",
    ]


def test_allocate_is_even_and_exact():
    topics = [make_topic(f"t{i}") for i in range(3)]
    assert allocate(10, topics) == [4, 3, 3]
    assert sum(allocate(1000, topics)) == 1000


def test_build_prompt_includes_avoid_list():
    p = build_prompt(make_topic(), 5, ["What is compost?"])
    assert "compost, pile" in p and "What is compost?" in p and "Generate 5" in p
    assert "Do not repeat" not in build_prompt(make_topic(), 5, [])


def test_dedup_drops_near_duplicates_only():
    probes = [Probe("How do I make compost at home?"), Probe("How do I make compost at home ?"),
              Probe("What is the capital of France?")]
    kept = dedup_probes(probes)
    assert [p.query for p in kept] == ["How do I make compost at home?", "What is the capital of France?"]


async def test_generate_probes_for_topic_loops_until_n_distinct():
    counter = itertools.count()
    prompts = []

    async def fake_complete(prompt: str) -> str:
        prompts.append(prompt)
        # Each call returns 3 questions, one of which repeats the previous call's.
        i = next(counter)
        return "\n".join([f"Question {i}-a?", f"Question {i}-b?", f"Question {max(i - 1, 0)}-a?"])

    probes = await generate_probes_for_topic(make_topic(), 7, fake_complete, questions_per_call=3)
    assert len(probes) == 7 and len({p.query for p in probes}) == 7
    assert all(p.topic == "0_compost_pile" and p.strategy == "taxonomy" for p in probes)
    assert "Question 0-a?" in prompts[1]  # later calls see earlier questions


async def test_generate_stops_after_max_calls_when_model_repeats_itself():
    async def stuck(prompt: str) -> str:
        return "The same question?"

    probes = await generate_probes_for_topic(make_topic(), 5, stuck, max_calls=4)
    assert [p.query for p in probes] == ["The same question?"]


DISTINCT_QUESTIONS = iter([
    "What temperature kills weed seeds?", "Which worms work in a bin?", "Why does my pile smell of ammonia?",
    "How long does leaf mold take?", "Can I add citrus peels?", "What is the ideal moisture level?",
    "Is sawdust safe to add?", "How big should a hot pile be?", "When is compost finished?",
    "What causes blossom end rot?", "How deep should I plant transplants?", "Which varieties resist wilt?",
    "When do I harden off seedlings?", "How much water per week?", "Why are flowers dropping?",
    "How do I ripen green fruit indoors?", "What spacing avoids blight?", "How tall do indeterminates get?",
    "Should I prune bush types?", "What eats leaves overnight?", "How do I store the harvest?",
])


async def test_taxonomy_guided_generation_respects_allocation():
    topics = [make_topic("0_compost", ("compost",)), make_topic("1_tomato", ("tomato",))]

    async def fake_complete(prompt: str) -> str:
        return "\n".join(next(DISTINCT_QUESTIONS) for _ in range(2))

    probes = await taxonomy_guided_generation(topics, 9, fake_complete, questions_per_call=2)
    assert len(probes) == 9
    assert sum(p.topic == "0_compost" for p in probes) == 5
    assert sum(p.topic == "1_tomato" for p in probes) == 4


async def test_taxonomy_generation_tops_up_topics_that_lost_probes_to_dedup():
    topics = [make_topic("0_compost", ("compost",))]
    batches = iter([
        # Round 1: 4 questions, 2 of which are near-duplicates of the other 2.
        "How do I make compost at home?\nHow do I make compost at home ?\n"
        "What temperature kills weed seeds?\nWhat temperature kills weed seeds ?",
        # Round 2 (top-up for the 2 lost to dedup).
        "Which worms work in a bin?\nCan I add citrus peels?",
    ])

    async def fake_complete(prompt: str) -> str:
        return next(batches)

    probes = await taxonomy_guided_generation(topics, 4, fake_complete, questions_per_call=4, oversample=1.0)
    assert [p.query for p in probes] == [
        "How do I make compost at home?", "What temperature kills weed seeds?",
        "Which worms work in a bin?", "Can I add citrus peels?",
    ]


def test_extract_taxonomy_on_synthetic_kb(tmp_path):
    synthetic_kb_builder.build(tmp_path)
    chunks = load_chunks(tmp_path)
    doc_topic = json.loads((tmp_path / "ground_truth.json").read_text())["document_topics"]
    topics = extract_taxonomy(chunks)

    assert 3 <= len(topics) <= 50
    # Outliers are reassigned: every chunk belongs to exactly one topic.
    assigned = [cid for t in topics for cid in t.chunk_ids]
    assert sorted(assigned) == sorted(c.chunk_id for c in chunks)
    # The two full-coverage topics each dominate at least one extracted topic.
    for gt_topic in ("composting", "tomato_growing"):
        purities = [sum(doc_topic[d] == gt_topic for d in t.doc_ids) / t.size for t in topics]
        assert max(purities) >= 0.8, gt_topic


def test_extract_taxonomy_rejects_tiny_kb():
    from src.data.kb_loader import Chunk

    with pytest.raises(ValueError):
        extract_taxonomy([Chunk(f"d#{i}", "d", f"text {i}") for i in range(5)])
