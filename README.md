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

## How it works

1. **Generate probes.** Three strategies, each tagged so results can be sliced
   by strategy later:
   - *taxonomy*: BERTopic over the KB chunks, then Claude Haiku writes
     questions per topic
   - *adversarial*: ask for topics adjacent to the KB but not in it, write a
     hypothetical document for each (HyDE turned around), and ask questions that
     document answers
   - *counterfactual*: questions that look answerable from a real chunk but
     need one fact it doesn't contain

   You can also pass your own probes (e.g. real user queries) with
   `--probes-file`. Everything is deduplicated at cosine 0.95.
2. **Query the RAG system** over HTTP (`POST {"query": ...}`) and take back the
   retrieved chunks and the answer.
3. **Score each probe** from 0 to 1 using three signals (below).
4. **Cluster**: UMAP to 10D, HDBSCAN, UMAP to 2D for display, Haiku names each
   cluster from its three most central questions.
5. **Classify zones**: mean score per cluster → dark (< 0.30), thin
   (0.30–0.60) or adequate (> 0.60), plus a severity score for ordering.

```
umbra audit --endpoint http://localhost:8765/query --kb-path path/to/kb \
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

**The noise bucket gets its own zone.** The spec says to treat HDBSCAN noise
as dark by default. On real runs the noise bucket is a mix: 140 probes with a
mean of 0.455 on the synthetic KB. So it's zoned from its own scores like any
other cluster, and it's reported separately.

## What the validation showed

On the synthetic KB (details in [docs/findings.md](docs/findings.md)):

- The two absent topics each come out as their own pure dark cluster and rank
  #1 and #2 by severity. The thin topics each get a thin cluster.
- None of the three generation strategies ever produces a question about the
  absent topics. Only the KB-blind question set does. That follows from how
  they work (they all start from the KB's documents), and it matters: the
  spec's claim that taxonomy probes find zero-coverage topics doesn't hold up.
- Full-coverage topics score around 0.58–0.68, right at the adequate line,
  because well-answered probes top out around 0.70 with MiniLM (see below).

## Known limitations

- **Zone thresholds aren't calibrated.** The 0.30/0.60 cutoffs come from the
  spec, but with MiniLM a question that's genuinely answered scores about 0.67
  on average and at most 0.75. Rankings and severity are right; absolute tiers
  run pessimistic. Thresholds should be calibrated per embedding model.
- **Out-of-domain gaps need outside questions.** See above. Without real query
  logs, Umbra finds gaps *near* the KB well and gaps *far* from it not at all.
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

- Calibrate zone thresholds with the gap-injection benchmark (remove document
  clusters from a real KB and check they turn dark) instead of hand-picked
  numbers.
- Add the user-pattern strategy over real query logs. On this evidence it's
  the only one that can reach out-of-domain gaps.
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
```

Tests: `pytest`. The Postgres tests run when `POSTGRES_DSN` is set and are
skipped otherwise.
