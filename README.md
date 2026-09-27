# Umbra

Umbra finds the questions a RAG system can't answer before your users do. It
generates probe questions, sends them through the system's normal query API,
scores how well each one is covered, and clusters the failures into named
"dark zones": topics where the system is likely to make things up.

## The problem

RAG systems don't fail loudly when a question falls outside what the knowledge
base covers. They retrieve whatever is closest, and the model writes a fluent
answer on top of it. The mock server in this repo does exactly that: ask its
gardening KB how to set up a deep water culture hydroponic system and you get
a confident sentence about staking tomato plants.

The usual evaluation tools (RAGAS, TruLens, DeepEval) grade answers to
questions you bring. That's useful, but it only covers questions you already
thought to ask. The gaps that hurt are the ones nobody wrote a test case for.
Umbra treats coverage as a search problem: generate a lot of questions from
different angles, score all of them, and look for regions of question-space
where scores are consistently low.

## Other tools in this space

**Answer-quality evaluators: RAGAS, TruLens, DeepEval.** These score how well
a system answers the questions you give them (faithfulness, relevance,
context precision). They don't generate the probing questions, so they can't
tell you what's missing. Umbra is complementary: it produces the questions
worth evaluating.

**Tools that do look at coverage or corpus quality:**

- **GapView** (academic, OpenReview 2025) is the closest peer. It maps KB
  coverage with cosine similarity and a 2D MDS projection. The main
  differences: it relies on a single similarity signal, whereas Umbra combines
  retrieval similarity, chunk spread and a cross-encoder answer check, because
  similarity alone can't tell "on-topic" from "actually answers it" (see the
  borderline cases in `tests/scoring_cases.py`). And it has no gap-injection
  benchmark, so there's no measured precision/recall for what it flags.
- **semantic-coverage** does UMAP plus centroid-distance gap reports. Also a
  single similarity signal, and one way of generating queries. Umbra's
  strategy tags exist because different strategies find different kinds of
  gaps (the ablation in [docs/findings.md](docs/findings.md) shows three of
  the four can't find missing topics at all).
- **rag-debugger** uses Gemini to decompose compound queries into sub-intents
  and debug them. It's for understanding individual failures; there's no
  clustering and no coverage map.
- **RAG Doctor** (an Apify actor) runs LLM-based contradiction and gap checks
  over crawled content. It's oriented toward auditing what a crawler
  collected, not probing a live RAG endpoint.
- **rag-corpus-profiler** does exploratory analysis and a security audit of a
  RAG corpus. It's useful before indexing, but it looks at the documents, not
  at which questions the system can answer.

Where Umbra is different, as far as I can tell: it probes the live system
through its public API, it scores each probe with three independent signals,
it separates probe strategies so you can see which one found what, and it
calibrates its dark/thin/adequate cutoffs against a benchmark with known gaps
instead of picking them by hand.

## How it works

1. **Generate probes.** Four strategies, each tagged so results can be sliced
   by strategy later:
   - *taxonomy*: BERTopic over the KB chunks, then Claude Haiku writes
     questions per topic
   - *adversarial*: ask for topics adjacent to the KB but not in it, write a
     hypothetical document for each (HyDE turned around), and ask questions that
     document answers
   - *counterfactual*: questions that look answerable from a real chunk but
     need one fact it doesn't contain
   - *kb_blind*: questions written from a list of topics the KB is supposed
     to cover (or a one-line domain description), without looking at the KB

   You can also pass your own probes (e.g. real user queries) with
   `--probes-file`. Everything is deduplicated at cosine 0.95.
2. **Query the RAG system** over HTTP (`POST {"query": ...}`) and take back the
   retrieved chunks and the answer.
3. **Score each probe** from 0 to 1 using three signals (below).
4. **Cluster**: UMAP to 10D, HDBSCAN, UMAP to 2D for display, Haiku names each
   cluster from its three most central questions.
5. **Classify zones**: mean score per cluster → dark (< 0.324), thin
   (0.324–0.400) or adequate (> 0.400), plus a severity score for ordering.
   The cutoffs come from a gap-injection benchmark, not the spec (below).

```
umbra audit --endpoint http://localhost:8765/query --kb-path path/to/kb \
    --strategies taxonomy,adversarial,counterfactual,kb_blind --topics-file topics.json \
    --n-probes 1000 --output report.csv
```

That writes per-probe scores to `report.csv`, the cluster table to
`report.clusters.csv`, and everything to Postgres if `POSTGRES_DSN` is set.

## Why it's built this way

**Three signals instead of one.** The score is
`0.4·RC + 0.35·(1 − SE) + 0.25·(1 − HP)`:

- RC (retrieval confidence) is the best cosine similarity between the question
  and any retrieved chunk. It's cheap and catches the obvious misses, but a
  single on-topic chunk can make it look fine.
- SE (semantic entropy) looks at how spread out the retrieved chunks are. If
  the five chunks have nothing to do with each other, the retriever is
  flailing.
- HP (hallucination probability) runs a cross-encoder
  (ms-marco-MiniLM-L-6-v2) over each question/chunk pair. It's the only signal
  that checks whether the chunk *answers* the question, as opposed to just
  being about the same thing.

Each one alone misses a failure mode the others catch. The 20 hand-labeled
cases in `tests/scoring_cases.py` are there to keep that honest.

**HP only runs on borderline probes.** The cross-encoder is the expensive
signal: five forward passes per probe, versus a dot product for RC and SE. If
RC and SE together already put a probe clearly high or clearly low (outside
0.35–0.65), HP won't change the verdict, so it's skipped. On the synthetic KB,
none of the out-of-domain probes needed HP at all. When it's skipped, the
score is RC and SE renormalized over their weights so it stays on the same
scale. The spec didn't say what to do there.

**SE isn't the formula from the spec.** The spec's version (entropy of a
histogram of pairwise distances) scores five mutually unrelated chunks as
*focused*, because their distances all land in one bin. I switched the default
to mean pairwise cosine distance after it got 20/20 hand-labeled cases right
versus 15/20. The original is still there as `--se-method spec` for
comparison. The full write-up with numbers is in
[docs/findings.md](docs/findings.md).

**Cluster in 10D, draw in 2D.** Squeezing 384-dim embeddings into 2D for a
scatter plot distorts distances. UMAP keeps the picture readable by tearing
some neighborhoods apart and gluing others together. HDBSCAN on that output
finds clusters that are partly layout artifacts. So there are two separate
projections: 10D (min_dist=0, packed tightly for density estimation) feeds
HDBSCAN, and 2D is only for the map. Both sets of coordinates are stored so the
map can be redrawn without re-projecting.

**Umbra re-embeds everything itself.** It ignores whatever similarity scores
the RAG system returns and embeds the question and the retrieved chunk text
with its own model (all-MiniLM-L6-v2). That keeps scores comparable across
systems and means the only integration is an HTTP endpoint.

**A synthetic KB with hand-written documents.** Validating a coverage tool
needs a KB where the right answer is known. `src/data/synthetic_kb_builder.py`
builds 39 gardening documents with two fully covered topics, two thin topics
(a couple of sentences buried in an unrelated document), and two topics with
nothing at all. I wrote the documents by hand instead of generating them so
nothing leaks: the builder refuses to run if a thin or absent topic shows up
anywhere it shouldn't.

**A fourth strategy that never reads the KB.** The spec's three strategies all
start from the documents, which means they can't ask about a topic that has no
documents. The benchmark confirms it: on their own they surfaced 0 of 12
injected gaps. kb_blind asks about what the KB *should* cover instead. It's
sized per topic rather than as a share of the probe budget, because a topic
needs at least 20 probes (min_cluster_size) to form a cluster of its own.

**Thresholds fit on known gaps.** The spec's 0.30/0.60 cutoffs assume scores
that can reach 1.0. With MiniLM, a question that's actually answered averages
about 0.67. The benchmark removes documents from the synthetic KB (or cuts a
topic down to a buried two-sentence mention) and runs the whole audit. It
labels the resulting clusters, and each threshold goes wherever F1 is highest
between the known-gap and known-present clusters. Three-tier accuracy goes
from 28% to 89% in-sample, 73–89% on held-out seeds.

**The noise bucket gets its own zone.** The spec says to treat HDBSCAN noise
as dark by default. On real runs the noise bucket is a mix: 140 probes with a
mean of 0.455 on the synthetic KB. So it's zoned from its own scores like any
other cluster, and it's reported separately.

## What the validation showed

On the synthetic KB (details in [docs/findings.md](docs/findings.md)):

- The two off-domain absent topics each come out as their own pure dark cluster
  and rank #1 and #2 by severity. Full-coverage topics are adequate. Of the
  two thin topics, one is thin and one falls just under the dark line.
- None of the three KB-anchored strategies ever produces a question about the
  absent topics, and on the benchmark they surface 0 of 12 injected gaps
  without kb_blind. The spec's claim that taxonomy probes find zero-coverage
  topics doesn't hold up.

## Known limitations

- **The thresholds are fit on three seeds.** The API credit ran out before
  seeds 3 and 4. Leave-one-seed-out puts dark_below anywhere from 0.322 to
  0.364. They're also tied to MiniLM, dispersion SE and this probe mix; change
  any of those and they need re-fitting.
- **Thin and absent barely separate.** In-domain absent clusters average 0.336
  and thin ones 0.348. The dark line between them is a close call, and
  hydroponics lands on the wrong side of it.
- **The adequate line hides depth gaps.** It's fit on "does this topic have
  documents", so clusters of questions the documents don't answer in depth
  (the adversarial ones, mostly) now read as adequate. Catching those needs a
  different ground truth.
- **kb_blind only covers what you list.** It finds missing topics inside the
  topic list or domain you hand it. Gaps nobody anticipated still need real
  user queries.
- **Probe generation isn't reproducible.** It goes through an LLM. Scoring,
  clustering and UMAP are deterministic given the same probes.
- **One embedding model.** RC, SE, dedup, and clustering all use MiniLM. A
  stronger embedder would likely raise the score ceiling and change where the
  thresholds should sit.
- **Semantically close topics merge.** In one run, tomato and hydroponics
  questions landed in the same cluster; they are both about growing plants.
  min_cluster_size=20 and EOM selection favor bigger clusters.
- **HP band edges are hard cutoffs.** A probe at 0.649 gets the cross-encoder
  and one at 0.651 doesn't, so scores have a small discontinuity there.

## What I'd do next / differently

- Finish the benchmark seeds, then run it on a real KB instead of the synthetic
  one.
- Calibrate the upper threshold against depth gaps (the share of a cluster's
  probes the cross-encoder says are answered), not just topic presence.
- Add the user-pattern strategy over real query logs, for the gaps nobody
  would think to list.
- Try a stronger embedding model for scoring and see how much of the ceiling
  problem goes away.
- Replace the hard HP band with something smoother, or run HP on everything
  when the probe count is small enough that cost doesn't matter.

## Running it locally

Python 3.11. Postgres with pgvector is optional (results go to CSV either way).

```
pip install -r requirements.txt && pip install -e .
cp .env.example .env              # ANTHROPIC_API_KEY, optionally POSTGRES_DSN

python -m src.data.synthetic_kb_builder
UMBRA_KB_PATH=data/synthetic_kb uvicorn src.connectors.mock_rag_server:app --port 8765

docker run -d --name umbra-pg -e POSTGRES_USER=umbra -e POSTGRES_PASSWORD=umbra \
    -e POSTGRES_DB=umbra -p 5434:5432 pgvector/pgvector:pg16
export POSTGRES_DSN=postgresql+psycopg2://umbra:umbra@localhost:5434/umbra
alembic upgrade head

umbra audit --endpoint http://localhost:8765/query --kb-path data/synthetic_kb \
    --n-probes 1000 --probes-file data/validation/validation_probes.jsonl --output out/audit.csv
python scripts/validate_phase2.py out/audit.csv

# gap-injection benchmark, threshold fit, strategy ablation
python -m src.benchmark.gap_injection --seeds 0 1 2 3 4 --out out/benchmark
python -m src.benchmark.calibrate out/benchmark
python -m src.benchmark.ablation out/benchmark
```

Tests: `pytest`. The Postgres tests run when `POSTGRES_DSN` is set and are
skipped otherwise.
