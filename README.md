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
5. **Classify zones**: mean score per cluster → dark (< 0.334), thin
   (0.334–0.408) or adequate (> 0.408), plus a severity score for ordering.
   The cutoffs come from a gap-injection benchmark, not the spec (below).
6. **Report**: a GapReport JSON (`docs/gap_report.schema.json`) with every
   cluster sorted by severity, sample questions per cluster, and document
   recommendations for the dark, thin, and most severe clusters.
7. **Dashboard**: the coverage map, the severity ranking, and each cluster's
   questions and recommendations, in Plotly Dash.

```
umbra audit --endpoint http://localhost:8765/query --kb-path path/to/kb \
    --strategies taxonomy,adversarial,counterfactual,kb_blind --topics-file topics.json \
    --n-probes 1000 --output out/audit.csv
umbra report --audit out/audit.csv --kb-path path/to/kb --output out/report.json
umbra dashboard --report out/report.json          # http://127.0.0.1:8050/?cluster=4
```

The audit writes per-probe scores to `audit.csv`, the cluster table to
`audit.clusters.csv`, retrieved chunk text to `audit.responses.jsonl`, and
everything to Postgres if `POSTGRES_DSN` is set. Keeping report generation
separate means an old audit can be re-reported under new thresholds, or with
web search, without re-querying the RAG system.

### Connecting to the RAG system

Three connectors, all behind the same `query(question) -> chunks + answer`
interface (`src/connectors/`):

- `--connector http` (default): POSTs `{"query": ...}` to `--endpoint` and
  reads the chunk list out of whatever keys the service uses (`chunks`,
  `contexts`, `source_documents`, ...).
- `--connector qdrant`: searches a Qdrant collection directly. The query
  vector has to come from the same model the collection was built with, which
  for `umbra index-qdrant` is Umbra's own MiniLM. Generation is optional
  (`--llm-endpoint`), since the score only looks at what was retrieved.
- `--connector langchain --chain module:attr`: any chain or retriever. It
  handles the RetrievalQA output shape (`result`/`source_documents`), the
  `create_retrieval_chain` shape (`answer`/`context`), and bare retrievers.

Queries go out 50 at a time by default (`--concurrency`). A 429 pauses the
whole pool until its Retry-After time, not just the request that got it;
otherwise the other 49 workers keep hitting the limit. 429s get their own
retry budget (20), separate from the one for 5xx and connection errors (4),
because "come back later" isn't a failure. The Qdrant connector
does the same for Qdrant Cloud's 429s. A probe that still fails is kept
in the CSV with its error, and the rest of the audit carries on.

Against a slow endpoint a big audit can spend hours in the query phase, so
every response is appended to `audit.checkpoint.jsonl` as it arrives, and the
probe set goes to `audit.resume.json` before the first query. If the run dies,
the same command with `--resume` picks the probe set back up (no regeneration,
so no API spend) and only queries what's missing. Failed queries aren't
checkpointed, which makes `--resume` after a partly failed run a retry of just
those. Both files are deleted once an audit finishes with no failures.

For nightly monitoring, generate a probe set once and re-run a fixed sample
of it. That makes no API calls, and the same seed gives the same 500 probes
each night, so the scores are comparable from run to run:

```
umbra audit --endpoint ... --kb-path ... --strategies none \
    --probes-file out/audit.csv --sample 500 --no-name-clusters --output out/nightly.csv
```

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
needs at least min_cluster_size probes (15) to form a cluster of its own.

**Thresholds fit on known gaps.** The spec's 0.30/0.60 cutoffs assume scores
that can reach 1.0. With MiniLM, a question that's actually answered averages
about 0.67. The benchmark removes documents from the synthetic KB (or cuts a
topic down to a buried two-sentence mention) and runs the whole audit. It
labels the resulting clusters, and each threshold goes wherever F1 is highest
between the known-gap and known-present clusters. Three-tier accuracy goes
from 31% to 90% in-sample, 78–96% on held-out seeds.

**min_cluster_size is 15, not the spec's 20.** At 20, a missing topic's ~30
questions often merged into a neighboring topic or split up instead of
forming a cluster of their own, and recall on injected gaps was 0.58, below
the spec's 0.75 minimum. Tested held-out (thresholds re-fit per setting,
three UMAP layouts), 15 raised recall to 0.81 *and* precision from 0.67 to
0.77. The full diagnosis is in findings section 8.

**Severity caps cluster size at 50.** The spec's `log(1 + size)` let a big,
mostly-covered cluster outrank small real depth gaps: compost pile temperature
(134 probes, 0.579) sat above fungal disease (28 probes, 61% of questions
unanswered). With the cap, size can move severity by at most 1.42× between the
smallest (15-probe) and largest cluster. The spec's own "500 at 0.1 beats 10 at 0.05"
example still holds.

**The tier and severity answer different questions.** The calibrated tier says
whether a topic exists in the KB (that's what the benchmark labels). Severity
also ranks depth gaps inside covered topics. So the report and dashboard never
hide ADEQUATE clusters; they sort everything by severity.

**Recommendations are checked, not just matched.** A KB passage gets
recommended for re-chunking only if the cross-encoder says it actually answers
one of the cluster's questions. Similarity alone kept recommending generic
sentences like "It starts with prevention: healthy soil..." to unrelated
clusters. Every candidate, from the KB or the web, is then ranked by simulated
gain: put its text into the retrieved set for each of the cluster's probes,
re-score, and measure the change.

**The map isn't red/yellow/green.** The spec asks for RdYlGn, but red and green
collapse for deuteranopic readers, and it's a rainbow ramp anyway. Zones are
ordered, so the map uses one blue ramp stepped by lightness (the most urgent
zone is the most prominent), plus a different marker shape per zone. Dark mode
has its own steps.

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
- The KB-internal recommender points the hydroponics gap at exactly the buried
  passage the builder put there. It leaves mushrooms alone, correctly, because
  that passage is already being retrieved.

## Known limitations

- **The thresholds are fit on three seeds.** The API credit ran out before
  seeds 3 and 4. Leave-one-seed-out moves them very little (dark_below
  0.332–0.335), but three seeds is still three seeds. They're also tied to
  MiniLM, dispersion SE, min_cluster_size 15 and this probe mix; change any of
  those and they need re-fitting.
- **Thin and absent barely separate.** In-domain absent clusters average 0.322
  and thin ones 0.343. Four thin clusters read as dark on the benchmark, and
  hydroponics does on the synthetic KB. A two-sentence mention behaves almost
  like no mention at all.
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
  EOM selection favors bigger clusters.
- **Recall on injected gaps clears the spec's minimum but not its target.**
  0.83 against a 0.85 target (precision 0.72, target 0.80; no false
  positives). Of the two remaining misses, one is a topic whose questions the
  KB still answers 30% of the time after its documents were removed; the
  other is flagged dark inside a mixed cluster that can't be credited to one
  topic.
- **Web search recommendations haven't run for real yet.** The code is tested
  against a fake backend, but the API credit ran out before a live run. The
  estimated improvement is simulated and optimistic: it assumes the retriever
  would return the new text.
- **HP band edges are hard cutoffs.** A probe at 0.649 gets the cross-encoder
  and one at 0.651 doesn't, so scores have a small discontinuity there.
- **The Qdrant and LangChain connectors have only seen test setups.** They're
  tested against a real Qdrant server and LangChain chains over the synthetic
  KB, but not yet against a production collection like SENTINEL-X's. For a
  collection built with another embedding model you have to pass that model's
  `embed_fn`; there's no CLI flag for it yet.

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
pip install -e ".[dev]"            # dev adds pytest and ruff; plain -e . is enough to run it
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

umbra report --audit out/audit.csv --kb-path data/synthetic_kb --output out/report.json
umbra dashboard --report out/report.json

# gap-injection benchmark (resumable), threshold fit, strategy ablation
umbra benchmark run --seeds 0,1,2,3,4 --out out/benchmark
umbra benchmark calibrate --out out/benchmark --write-thresholds out/thresholds.json
umbra benchmark ablate --out out/benchmark
umbra report ... --zone-thresholds-file out/thresholds.json   # use the fit
```

Or the whole stack in Docker: Postgres+pgvector on 5434 (same as above),
Qdrant on 6333, and the mock RAG server on 8765. Migrations run and the
synthetic KB is indexed into Qdrant on the way up:

```
docker compose up -d --wait
docker compose run --rm umbra audit --endpoint http://mock-rag:8765/query \
    --kb-path data/synthetic_kb --strategies none \
    --probes-file data/validation/validation_probes.jsonl --no-name-clusters --output out/audit.csv
docker compose run --rm umbra audit --connector qdrant --kb-path data/synthetic_kb ...
```

Output lands in `./out`. Host ports can be moved with `UMBRA_PG_PORT`,
`UMBRA_QDRANT_PORT` and `UMBRA_MOCK_PORT` if the local Postgres is already on
5434. Setting `UMBRA_MOCK_RATE_LIMIT` (requests per `UMBRA_MOCK_RATE_WINDOW`
seconds) and `UMBRA_MOCK_LATENCY` makes the mock server behave like a slow,
rate-limited production endpoint.

Tests: `pytest`. The Postgres tests run when `POSTGRES_DSN` is set, and the
Qdrant server tests when `QDRANT_URL` is set; both are skipped otherwise. The
other Qdrant tests use qdrant-client's in-process mode, so they always run.
