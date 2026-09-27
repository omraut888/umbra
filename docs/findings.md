# Findings

Things I found while building Umbra that change how the spec should be read.
Each one comes with the data that backs it up, so it can be rechecked.

## Current state

The two-minute version. The numbered sections below have the evidence.

**Where it stands.** Phases 1–3 are built: probe generation, three-signal
scoring, UMAP + HDBSCAN clustering, zone tiers and severity, recommendations,
the GapReport JSON, the Dash dashboard, and a rerunnable gap-injection
benchmark. On that benchmark (3 seeds, known gaps removed from a synthetic
KB), dark-zone detection clears every one of the spec's §10 minimums:

| metric | result | spec target | spec minimum |
|---|---|---|---|
| recall | 0.83 | 0.85 | 0.75 |
| precision | 0.72 | 0.80 | 0.70 |
| false positive rate | 0.00 | ≤ 0.15 | ≤ 0.25 |

**The decisions that differ from the spec, and why:**

- **min_cluster_size is 15, not 20** (section 8). This is the final decision.
  At 20, a removed topic's questions often merged into a neighboring topic
  instead of forming their own cluster, and recall was 0.58, below the
  spec's minimum. Tested held-out with thresholds re-fit and three UMAP
  layouts, 15 gave recall 0.81 and precision 0.77, against 0.53 and 0.67 at
  20, and the fitted thresholds stopped moving between seeds.
- **Recall was prioritized over precision, deliberately.** Both clear the
  minimums, but precision (0.72) sits nearer its floor than recall (0.83).
  The cost is four *thin* topics that now read DARK. Each has a
  two-sentence mention left, and the KB answers 0–10% of their questions.
  Missing a real gap costs more than flagging a nearly-empty topic a tier
  too dark, and no *covered* topic is ever flagged (FPR 0).
- **Zone thresholds are 0.334 / 0.408, fit on known gaps** (sections 2, 4,
  8), not the spec's 0.30 / 0.60. With MiniLM embeddings a fully answered
  question averages about 0.67, so the spec's 0.60 line called most covered
  topics thin. The tier means "does this topic exist in the KB", because
  that's what the benchmark labels. Depth gaps inside covered topics are
  ranked by severity instead.
- **Semantic entropy uses mean pairwise distance, not the spec's histogram
  entropy** (section 1). The spec formula scores mutually unrelated chunks as
  *focused*, because their distances all land in one histogram bin. It got
  15 of 20 hand-labeled cases right, against 20 of 20 for dispersion. The
  original is kept as `--se-method spec`.
- **A fourth probe strategy, kb_blind** (sections 3 and 5). Taxonomy,
  adversarial, and counterfactual generation all start from the KB's own
  content, so they can't ask about a topic that has no documents. On the
  benchmark they surfaced 0 of 12 removed topics on their own. kb_blind
  writes questions from a list of what the KB *should* cover, without
  reading it, and is what makes absolute-gap detection possible at all.
- **Severity caps cluster size at 50** (section 6). The spec's
  `log(1 + size)` let a large, mostly covered cluster (compost pile
  temperature, 134 probes) outrank small real depth gaps (pest control,
  fungal disease). With the cap, both depth gaps rank above every compost
  and tomato cluster.
- **Purity doesn't help, so it isn't used** (sections 4 and 8). Using
  cluster purity to split DARK from THIN did worse than the score alone at
  both cluster sizes (held-out dark F1 0.60 vs 0.78 at 15). Absent and thin
  clusters have the same purity. It's kept as an option and documented as a
  negative result.

**Open, and blocked on API credits.** Everything below needs new LLM calls,
and the key currently returns "credit balance too low":

- **Seeds 3 and 4 of the 5-seed calibration.** The thresholds and the recall
  above are from seeds 0–2. A background job will run them automatically at
  min_cluster_size 15.
- **More than 30 kb_blind probes per topic.** Recall rose steadily from 15
  to 30 probes per topic (section 8). Whether it keeps rising past 30 is
  untested, because only fewer-than-generated can be tested by subsampling.
- **Real web search.** External recommendations only run against a fake
  backend in tests. Until the real path is validated, treat any
  `estimated_improvement` in a report as coming from KB-internal
  recommendations only, and as optimistic.
- **Re-clustering the Phase 2 dashboard run.** `out/phase2.csv` was clustered
  at min_cluster_size 20. Its report already uses the new thresholds, but
  re-clustering at 15 needs new cluster names from Haiku.

**Known limits that aren't bugs.** kb_blind only finds gaps inside the topic
list it's given; gaps nobody anticipated still need real user queries.
Removing a topic's documents doesn't always remove the topic: one missed gap
still had 30% of its questions answered by another document. And every
threshold here is tied to MiniLM, dispersion SE, and min_cluster_size 15;
re-fit them (`umbra benchmark calibrate`) if any of those changes.

## 1. The spec's semantic entropy formula reads irrelevant retrievals as focused

### What the spec says

Section 4 of the spec defines SE as the Shannon entropy of the distribution of
retrieved-chunk embeddings, and says what it should mean:

> Low entropy = all retrieved chunks are semantically similar to each other
> (focused, relevant retrieval). High entropy = retrieved chunks are
> semantically diverse (the system is grasping at unrelated content).

The formula it gives:

```python
distances = pdist(chunk_embs, metric="cosine")
hist, _ = np.histogram(distances, bins=10, density=True)
hist = hist + 1e-10
return float(entropy(hist / hist.sum()))
```

SE goes into the composite score as `β·(1 − SE)`, so low SE is rewarded.

### Why it doesn't measure that

The histogram entropy measures how *varied* the pairwise distances are, not how
*large* they are. Take the case SE is supposed to catch: the system retrieves 5
chunks that have nothing to do with each other. Every pair is roughly equally
unrelated, so all 10 pairwise distances sit around 0.9–1.0. They all fall in the
same histogram bin, entropy is ~0, and the retrieval gets the maximum
`β·(1 − SE)` bonus, exactly like 5 near-duplicate chunks would.

Entropy is highest when the distances are spread across many bins, i.e. a mix
of close and far pairs. That shape shows up in *good* retrievals (a few very
similar chunks plus a couple of looser ones) at least as often as in bad ones.

There's a second, smaller problem. As written, `np.histogram` takes its range
from the data's own min and max, so the result doesn't depend on scale at all:
ten distances between 0.30 and 0.35 bin the same way as ten between 0.1 and
0.9. On the test cases below, that version of SE lands between 0.54 and 0.78
for every case, good or bad. It carries essentially no signal. Umbra's `spec`
method pins the histogram range to [0, 1] and divides by ln(10) so SE fits in
[0, 1]. That's the minimum needed for the formula to mean anything, and it
still fails for the reason above.

### The fix

Use the mean pairwise cosine distance between retrieved chunks ("dispersion"):
0 for identical chunks, near 1 for unrelated ones. It measures what the spec
describes. It's the default now, and the original formula is still available
as `--se-method spec` so the two can be compared on any KB.

Separately: the spec defines SE = 0 when fewer than 2 chunks are retrieved,
which gave an empty retrieval a free `β` and a coverage score of 0.35. Empty
retrievals now score 0 under both methods.

### Validation: 20 hand-labeled cases

`tests/scoring_cases.py` has 20 (query, retrieved chunks) cases covering several
unrelated domains. I labeled each one *before* running the scorer, using the
spec's own zone thresholds: good means CS > 0.60, bad means CS < 0.30,
borderline means 0.30–0.60. Borderline covers "on-topic chunks that don't
contain the answer" and "one relevant chunk among unrelated ones". Weights,
HP band and everything else are identical between the two columns; only SE
differs. ✗ marks a miss.

| case | expected | SE spec | CS spec | SE dispersion | CS dispersion |
|---|---|---|---|---|---|
| good_python_list_sort | > 0.60 | 0.439 | 0.740 | 0.308 | 0.801 |
| good_photosynthesis | > 0.60 | 0.577 | 0.695 | 0.419 | 0.667 |
| good_http_404 | > 0.60 | 0.377 | 0.674 | 0.510 | 0.708 |
| good_passport_renewal | > 0.60 | 0.377 | 0.735 | 0.460 | 0.696 |
| good_sourdough_starter | > 0.60 | 0.577 | 0.729 | 0.365 | 0.738 |
| good_git_undo_commit | > 0.60 | 0.439 | 0.699 | 0.296 | 0.766 |
| good_vitamin_d | > 0.60 | 0.577 | 0.736 | 0.424 | 0.720 |
| bad_quantum_vs_baking | < 0.30 | 0.377 | **0.344** ✗ | 0.529 | 0.273 |
| bad_tax_vs_football | < 0.30 | 0.377 | **0.340** ✗ | 0.747 | 0.168 |
| bad_kubernetes_vs_poetry | < 0.30 | 0.276 | **0.305** ✗ | 0.791 | 0.167 |
| bad_heart_attack_vs_cars | < 0.30 | 0.540 | 0.270 | 0.854 | 0.123 |
| bad_roman_empire_vs_skincare | < 0.30 | 0.577 | 0.228 | 0.623 | 0.207 |
| bad_mortgage_vs_birds | < 0.30 | 0.577 | 0.233 | 0.789 | 0.134 |
| bad_empty_retrieval | < 0.30 | 0.000 | 0.000 | 0.000 | 0.000 |
| borderline_related_no_answer_compost_temp | 0.30–0.60 | 0.477 | 0.396 | 0.453 | 0.405 |
| borderline_related_no_answer_python_gil | 0.30–0.60 | 0.377 | 0.401 | 0.491 | 0.361 |
| borderline_one_relevant_among_noise_insulin | 0.30–0.60 | 0.196 | **0.794** ✗ | 0.957 | 0.579 |
| borderline_one_relevant_among_noise_wifi | 0.30–0.60 | 0.000 | **0.885** ✗ | 0.975 | 0.573 |
| borderline_adjacent_topic_electric_cars | 0.30–0.60 | 0.377 | 0.347 | 0.640 | 0.340 |
| borderline_adjacent_topic_marathon | 0.30–0.60 | 0.439 | 0.456 | 0.520 | 0.428 |

**Spec formula: 15/20 in the expected zone. Dispersion: 20/20.**

The two "one relevant chunk among noise" rows show the failure most clearly.
The retrieved set is one good chunk plus four random facts, which is
maximally diverse, and the spec formula gives it the *lowest* SE of all 20
cases (0.000 and 0.196). The off-topic misses happen the same way:
mutually unrelated chunks earn most of the 0.35 `β` term, which lifts a
clearly failed retrieval above the dark-zone line.

With the empty-retrieval fix left out, the spec formula scores 14/20: the
empty case comes out at 0.350.

### Validation: synthetic KB

The synthetic KB (`src/data/synthetic_kb_builder.py`) is 39 home-gardening
documents with six topics at known coverage tiers: composting and tomato
growing are **full** (9 documents each), hydroponics and mushroom growing are
**thin** (a 2–3 sentence passage buried in a document about something else),
crypto taxes and orbital mechanics are **absent**. The RAG system under test is
the mock server (MiniLM cosine retrieval, top 5).

**KB-blind probes.** Claude Haiku wrote 40 questions per topic from a one-line
topic description, without seeing the KB (237 after dedup, saved in
`data/validation/validation_probes.jsonl`). Both methods scored the same probes
against the same retrievals:

| topic | true tier | n | mean CS, spec | mean CS, dispersion | probes in dark zone, spec | probes in dark zone, dispersion |
|---|---|---|---|---|---|---|
| composting | full | 40 | 0.567 | 0.571 | 0 | 0 |
| tomato_growing | full | 40 | 0.557 | 0.565 | 0 | 0 |
| hydroponics | thin | 40 | 0.327 | 0.315 | 14 | 11 |
| mushroom_cultivation | thin | 38 | 0.338 | 0.313 | 6 | 17 |
| cryptocurrency_taxation | absent | 40 | 0.265 | **0.231** | 31 (78%) | **37 (93%)** |
| orbital_mechanics | absent | 39 | 0.293 | **0.251** | 22 (56%) | **32 (82%)** |

Both methods order the tiers correctly (full > thin > absent). The difference
is in the absent topics, where dispersion makes the call more decisively.
Under the spec formula, orbital mechanics averages 0.293, 0.007 above the
dark-zone line, and only 56% of its probes land dark. With dispersion, 82–93%
of absent-topic probes are dark, and none of them need the cross-encoder:
their preliminary RC + SE score is already under the HP band. That's the
spec's own intent, "use the cheaper RC and SE signals as a fast filter".
Under the spec formula, 8% and 28% of absent probes still needed HP, because
unrelated chunks kept SE low and propped up the preliminary score.

Full topics are essentially unchanged (±0.01), which is what you'd expect.
When retrieval is on-topic, both formulas see a moderately spread set of
related chunks.

**Taxonomy probes.** 1000 probes generated from BERTopic topics of the KB
itself (100 per topic), each BERTopic topic mapped to the ground-truth label
held by most of its chunks:

| mapped topic | true tier | n | mean CS, spec | mean CS, dispersion |
|---|---|---|---|---|
| composting | full | 200 | 0.608 | 0.620 |
| tomato_growing | full | 300 | 0.619 | 0.633 |

Both full topics land in the adequate zone under either method. By construction,
the taxonomy strategy can't produce probes for topics with no documents, so it
says nothing about the absent tier. More on that in the probe-strategy notes.

## 2. The adequate threshold sits close to the score ceiling

With the KB-blind probes, the two full-coverage topics average 0.57, just under
the spec's 0.60 adequate line, so they classify as thin under both SE methods.
I looked into why before accepting it. There are two causes, and neither is a
scoring error.

**The probes ask beyond the KB.** The lowest-scoring composting and tomato
questions are about bokashi, worm-bin leachate, composting in very wet
climates, telling fungal from bacterial disease, and recovering a snapped main
stem. None of that is in the 18 documents, and the cross-encoder agrees
(HP ≈ 0.95–0.999 on all of them). Across the 80 KB-blind questions for the two full
topics, the cross-encoder finds the answer in the retrieved text for 51, and
those average 0.64. The other 29 average 0.43. "Full coverage" in the ground truth means "several documents
dedicated to the topic", not "every question about the topic is answerable".
The per-probe scores are picking up real depth gaps.

**Well-covered probes top out near 0.70.** Across all 1237 probes, the highest
CS is 0.752 and the 95th percentile is 0.719. Probes where the cross-encoder is
confident the answer is in the retrieved text (HP < 0.05, n=434) average only
0.673. The ceiling comes from the signals, not the KB. MiniLM cosine similarity
between a question and the passage that answers it rarely goes above 0.8 (RC
p95 = 0.764), and five on-topic chunks from a real KB have a mean pairwise
distance around 0.45–0.55. So a perfect probe scores roughly
0.4·0.7 + 0.35·0.5 + 0.25·1.0 ≈ 0.70, and the 0.60 threshold leaves only about
0.1 of headroom. A topic mixing answered and unanswered probes drops below it
quickly.

The ranking is right. The absolute thresholds were set without a particular
embedding model in mind. They've since been re-fit on the gap-injection
benchmark; see section 4.

## 3. None of the KB-derived probe strategies can find a topic the KB never mentions

The Phase 2 audit ran all three generation strategies (1000 probes: taxonomy,
HyDE adversarial, counterfactual) together with the 237 KB-blind probes, then
clustered everything (1218 probes after dedup, 26 clusters plus noise).

The clustering does what it should with the absent topics. Crypto taxes and
orbital mechanics each form their own cluster, 100% pure (40/40 and 39/39),
both DARK (mean CS 0.231 and 0.251), and they rank first and second by
severity. The thin topics also get their own clusters: hydroponics (28 of 40
probes, THIN, 0.311) and mushroom growing (31 of 38, THIN, 0.335). Those two
are the next most severe clusters after the noise bucket.

Here's where each strategy's probes landed:

| strategy | probes | in DARK clusters | in THIN | in ADEQUATE | mean CS |
|---|---|---|---|---|---|
| taxonomy | 445 | 0% | 89% | 11% | 0.611 |
| counterfactual | 207 | 0% | 93% | 7% | 0.525 |
| adversarial (HyDE) | 331 | 0% | 99% | 1% | 0.439 |
| KB-blind (stand-in for real user queries) | 235 | 34% | 64% | 3% | 0.373 |

Every probe in the two dark clusters came from the KB-blind set. That isn't a
tuning problem; it follows from how the strategies work. Taxonomy probes are
generated from the KB's own topics. Counterfactual probes are near-misses
around real KB chunks. Adversarial probes ask for topics *adjacent* to the KB.
Asked for boundary topics of a gardening KB, Haiku came back with 108 of them:
micronutrient deficiencies, rainwater harvesting, herbicide persistence,
damping-off. Useful gaps, all of them gardening. None of the strategies will
ever ask a gardening KB about crypto taxes.

The thin topics are nearly as invisible. Of the 67 probes in the hydroponics
and mushroom clusters, only 8 came from generated strategies (3 adversarial,
5 counterfactual), all in the mushroom cluster. BERTopic folds the two buried
passages into their host topics (seed starting, mulch), so taxonomy never
targets them.

What this means for the spec's strategy table: it credits taxonomy-guided
generation with finding "absolute gaps: topics with zero coverage", and on
this data it can't. In practice, absolute gaps in the out-of-domain sense only
show up from outside the KB: real query logs (the spec's user-pattern
strategy, which needs production traffic) or some other source of questions
that doesn't start from the documents. The generated strategies are good at
something else. Adversarial probes average 0.439, clearly below taxonomy's
0.611, and they surface boundary gaps inside the domain. That's worth keeping
in mind for the per-strategy ablation.

Two smaller things from the same run:

- Clusters from the full-coverage topics sit at 0.58–0.68, mostly just under
  the spec's 0.60 adequate line. That's the ceiling effect from finding 2,
  and it's what section 4 fixes. By
  severity the tiers still come out in the right order: absent (2.74, 2.62),
  then thin (2.28, 2.25), then everything built on the full topics (1.10–1.80).
- The adversarial topic parser accepted markdown headings ("# Boundary Topics")
  as topic names, which accounted for 9 of the 1218 probes. Fixed after this
  run.

## 4. Zone thresholds, re-fit on the gap-injection benchmark

> **Update:** the thresholds in this section were fit with HDBSCAN at
> min_cluster_size 20. Section 8 lowers that to 15 and re-fits them to
> **0.334 / 0.408**. The method and the conclusions here still hold; the
> numbers in section 8 supersede the ones below.

The spec's 0.30 / 0.60 cutoffs don't fit this scorer (section 2), so I fit new
ones on known gaps instead of picking new round numbers.

### The benchmark

`src/benchmark/gap_injection.py` follows the spec's section 10 method with one
addition. Each seed splits the 14 in-domain topics of the synthetic KB
(composting, tomatoes, and the 12 background topics) into:

- **absent** (4 topics): every document removed
- **thin** (3 topics): every document removed except a two-sentence passage,
  buried in a document about another topic. This is the same construction the
  builder uses for hydroponics and mushrooms. The spec's benchmark has no thin
  tier, but without one there's nothing to calibrate the upper threshold on.
- **present** (7 topics): untouched

The full audit then runs against a mock RAG server built on the reduced KB,
using all four probe strategies (600 KB-anchored probes plus 30 kb_blind
probes per topic, ~1000 after dedup). The steps are scoring, then UMAP +
HDBSCAN. A cluster gets a ground-truth tier when at least 5 of its probes are
kb_blind probes and at least 60% of those are about one topic. kb_blind probes
are the only ones that carry a topic label.

I planned 5 seeds. The API credit ran out partway through seed 3, so the
numbers below are from seeds 0–2: 63 clusters, 46 of them labeled (9 absent,
9 thin, 28 present). Seeds 3 and 4 should be rerun before these thresholds are
treated as final.

### What the scores look like

| true tier | clusters | mean | p10 | p90 | range |
|---|---|---|---|---|---|
| absent | 9 | 0.336 | 0.297 | 0.358 | 0.297–0.482 |
| thin | 9 | 0.348 | 0.304 | 0.382 | 0.304–0.393 |
| present | 28 | 0.516 | 0.420 | 0.598 | 0.375–0.649 |

The same picture holds without clustering. Using the mean score of each
topic's kb_blind probes, absent topics average 0.322, thin 0.341, and present
0.492.

Present separates cleanly from the other two. Absent and thin mostly don't.
Once a topic's documents are gone, the questions about it retrieve whatever
gardening text is left, and a two-sentence mention barely moves that. Part of
the reason is that "absent" in this benchmark means *the topic's own documents
are removed*, not that nothing related remains. When seed 0 removed
seed_starting, the tomato seed-starting document was still there, and that
cluster scored 0.482. The off-domain topics from section 1 (crypto taxes at
0.231, orbital mechanics at 0.251) sit well below every in-domain absent
cluster.

### Fitting

Each threshold is placed where it best separates two groups by F1. When
several cut points tie, the middle one is used, so the line sits in the gap
rather than against one class.

- `dark_below`: absent vs thin + present → **0.324**
- `adequate_above`: present vs absent + thin → **0.400**

| thresholds | 3-tier accuracy | dark P / R / F1 | adequate P / R / F1 | spec §10 precision / recall / FPR (mean over seeds) |
|---|---|---|---|---|
| spec 0.30 / 0.60 | 28% | 1.00 / 0.11 / 0.20 | 1.00 / 0.11 / 0.19 | 0.33 / 0.08 / 0.00 |
| calibrated 0.324 / 0.400 | 89% | 0.78 / 0.78 / 0.78 | 0.96 / 0.96 / 0.96 | 0.78 / 0.58 / 0.00 |

Under the spec thresholds, 25 of 28 present clusters and 8 of 9 absent ones
both come out THIN. The spec cutoffs mostly just call everything thin.

89% is in-sample. Leave-one-seed-out (fit on two seeds, score the third) gives
**73%, 89%, and 85%** on the held-out seeds, against 27%, 33%, and 23% for the
spec thresholds. The fitted values move from seed to seed: dark_below is
0.322–0.364, adequate_above 0.372–0.400. With three seeds that spread is the
honest error bar.

The spec §10 recall of 0.58 has two sources. Three of the 12 absent topics
never formed a labeled cluster (their probes scattered into mixed clusters or
noise), and two absent clusters scored above the dark line (the seed_starting
case above, and one at 0.358).

### Before vs after on the full synthetic KB

The Phase 2 run from section 3, re-zoned (cluster means don't depend on the
thresholds):

| ground-truth topic | true tier | cluster mean | spec zone | calibrated zone |
|---|---|---|---|---|
| composting (19 probes in cluster 0) | full | 0.579 | THIN | ADEQUATE |
| tomato_growing (spread; largest group in noise) | full | 0.432–0.678 | THIN | ADEQUATE (all 40 probes) |
| mushroom_cultivation | thin | 0.335 | THIN | THIN |
| hydroponics | thin | 0.311 | THIN | **DARK** |
| cryptocurrency_taxation | absent | 0.231 | DARK | DARK |
| orbital_mechanics | absent | 0.251 | DARK | DARK |

The full topics are now right and the absent ones stay right. Hydroponics
drops to DARK, which is the thin/absent overlap above showing up on real
data. At 0.311 it's closer to the in-domain absent clusters than to anything
present.

### What the tier means, and what it doesn't

**The calibrated tier answers "does this topic exist in the KB", not "is
there enough depth".** That's what the gap-injection ground truth labels: a
topic is present if its documents are there, absent if they were removed, thin
if one buried passage is left. A present topic's cluster still contains plenty
of questions its documents don't answer (section 2), and the thresholds were
fit to call those clusters ADEQUATE anyway, because by the benchmark's
definition they are.

So on the full KB, "Vegetable Garden Pest Control" (0.444) and "Fungal Disease
Prevention" (0.432) reading ADEQUATE is consistent with the ground truth, not a
bug. Pest management and tomato disease both have documents. The questions in
those clusters (mostly adversarial probes) go past what those documents say.
That's a depth gap, and the tier isn't built to show it. The calibrated THIN
band is narrow (0.324–0.400, vs the spec's 0.30–0.60), and 23 of the 26
clusters on the full KB read ADEQUATE.

**Severity is the signal for depth, regardless of tier.** It's computed the
same way for every cluster, `(1 − mean_cs) · log(1 + size) · (1 − std_cs)`, so
it keeps ranking low-scoring, consistent clusters even after they're called
adequate. Among the ADEQUATE clusters on the full KB, Pest Control (1.87) and
Fungal Disease (1.67) rank above well-covered ones like the C:N-ratio cluster
(1.14) and tomato ripening (1.10). Severity originally grew with cluster size without limit, so a large,
reasonably covered cluster could outrank a small depth gap: compost pile
temperature (134 probes, 0.579) scored 1.80. The size term is now capped;
section 6 has the before and after.

**For Phase 3:** the report must keep severity-sorted output visible for
ADEQUATE clusters too, not just list DARK and THIN zones. A report filtered to
non-adequate tiers would hide exactly the depth gaps that adversarial probing
is good at finding.

### Negative result: cluster purity doesn't break the DARK/THIN tie

Since score alone barely separates absent from thin, I tried cluster purity as
a second signal. Purity is the share of a cluster's probes traceable to its
most common true topic. In the tested rule, the score still decides ADEQUATE
vs not, and inside not-adequate, purity alone decides DARK vs THIN. The purity
cutoff and which side of it counts as DARK were both fit on the benchmark, the
same way as the score thresholds. Two versions:

- **ground-truth purity**: kb_blind probes carry their topic, and
  counterfactual probes trace to one through their source chunk's document.
  This can't be computed in a real audit (there's no ground truth), so it's
  an upper bound for what purity could do.
- **kb_blind purity**: only the kb_blind topic labels, which a real audit has
  whenever kb_blind runs.

On seeds 0–2 (9 absent, 9 thin, 28 present labeled clusters):

| DARK vs THIN decided by | rule | dark P / R / F1 | thin P / R / F1 | 3-tier acc | held-out dark F1 | held-out acc |
|---|---|---|---|---|---|---|
| score (dark_below 0.324) | — | 0.78 / 0.78 / 0.78 | 0.78 / 0.78 / 0.78 | 89% | **0.71** | **82%** |
| ground-truth purity | dark if ≥ 0.627 | 0.75 / 0.67 / 0.71 | 0.70 / 0.78 / 0.74 | 87% | 0.61 | 80% |
| kb_blind purity | dark if ≥ 0.627 | 0.75 / 0.67 / 0.71 | 0.70 / 0.78 / 0.74 | 87% | 0.61 | 80% |

Score alone beats purity, in-sample and held-out. The reason is visible in the
raw values: absent clusters average 0.60 purity and thin clusters 0.62, with
almost the same spread (p10–p90 of 0.30–0.71 vs 0.45–0.78). A missing topic's
questions cluster together just as tightly as a barely-covered topic's
questions do. The two purity versions come out identical because the few
counterfactual probes in these clusters trace to *other* topics, so they never
change the most common label.

The purity rule stays available (`PurityRule` in `src/clustering/zones.py`,
and every cluster now reports its kb_blind purity), but it's not the default.
If seeds 3–4 change this materially, it'll be noted here.

Current defaults: `ZoneThresholds(0.324, 0.400)` in `src/clustering/zones.py`,
overridable with `umbra audit --zone-thresholds`. `SPEC_THRESHOLDS` stays
available for comparison. The fitted values depend on the embedding model
(MiniLM), the SE method (dispersion), and the probe mix, so all three should
be re-fit if any of them changes.

## 5. KB-anchored strategies structurally cannot detect absolute gaps

Taxonomy, adversarial, and counterfactual generation all derive probes from
KB content: its topics, its neighborhood, its chunks. A topic with no
documents leaves nothing for them to start from. Only generation that never
looks at the KB, and works from a description of what the KB *should* cover,
can reliably ask about a topic that's missing. The spec's claim that
taxonomy-guided generation finds "absolute gaps: topics with zero coverage"
is wrong.

That's now a real strategy, `kb_blind` (`src/probe_generation/kb_blind.py`),
selectable in the audit:

```
umbra audit ... --strategies taxonomy,adversarial,counterfactual,kb_blind --topics-file topics.json
umbra audit ... --strategies ...,kb_blind --domain "home food gardening: growing vegetables, fruit and herbs"
```

With `--topics-file`, it asks about each listed topic. With `--domain`, Haiku
first expands the description into topics, still without seeing the KB. It's
sized per topic (default 30), not as a share of `--n-probes`. A topic needs at
least min_cluster_size (20) probes to come out as its own cluster.

### Confirmed on the benchmark

`src/benchmark/ablation.py` re-clusters each benchmark seed's already-scored
probes using only some strategies. Each probe is attributed to a topic through
the full-run cluster it sat in. For the 12 injected absent topics across seeds
0–2:

| strategies in the probe set | absent topics surfaced as their own cluster | ...and DARK |
|---|---|---|
| all four | 9 / 12 | 7 / 12 |
| taxonomy + adversarial + counterfactual | **0 / 12** | **0 / 12** |
| adversarial only | 0 / 12 | 0 / 12 |
| kb_blind only | 2 / 12 | 2 / 12 |

Without kb_blind, not one injected gap surfaced, even though these are
in-domain gaps adjacent to what's left in the KB, the case where the
KB-anchored strategies have the best chance. On the full synthetic KB
(section 3), the off-domain gaps got zero probes from those strategies at all.

Two details from the same data:

- **The KB-anchored probes still matter.** Of the 386 probes in absent-labeled
  clusters, 241 are kb_blind, 107 adversarial, 26 counterfactual, and 12
  taxonomy. Adversarial probes do wander into the gap's neighborhood, just
  never enough on their own to form a cluster. kb_blind alone surfaced only
  2 of 12: 420 probes on their own cluster coarsely (8 clusters for 14 topics
  in seed 0), and missing topics get merged into mixed clusters. The
  combination is what works.
- **The scope is whatever you give it.** kb_blind finds gaps inside the topic
  list or domain description it's handed. A gardening description won't
  produce a question about crypto taxes. Section 3 only caught those because
  the topic list included them. Gaps nobody anticipated still need real user
  queries (the spec's user-pattern strategy).

## 6. Capping the size term in severity

The spec's severity is `(1 − mean_cs) · log(1 + size) · (1 − std_cs)`. The
log term was meant to rank a big consistent failure above a tiny one, and it
does. But it also let a large, mostly-covered cluster outrank small, real
depth gaps. On the full synthetic KB, "Compost Pile Temperature Management"
(134 probes, mean 0.579) scored 1.80, above "Fungal Disease Prevention" (28
probes, mean 0.432, 1.67), and was within 0.07 of "Vegetable Garden Pest
Control" (47 probes, 1.87). A top-N list built on that would put a
well-covered topic next to, or ahead of, the gaps it's supposed to surface.

Pest Control and Fungal Disease have ground truth as depth gaps: the pest and
disease documents exist, and the questions in these clusters go beyond them.
The data backs this up. The cross-encoder finds no answer for 62% and 61% of
their probes, against 41% for compost pile temperature and 6–38% for the
other compost and tomato clusters. (A probe counts as unanswered when HP ≥
0.5, or when HP wasn't computed and the preliminary score was under the band.)

The size term is now `log(1 + min(size, 50))`. 50 was 2.5× min_cluster_size
at the time, and capping there bounds the size effect: the largest possible
cluster gets at most log(51)/log(21) ≈ 1.29× the weight of the smallest one.
(With min_cluster_size lowered to 15 in section 8, the smallest cluster is now
15 probes, and the bound is log(51)/log(16) ≈ 1.42×.) To outrank a
smaller cluster on size alone, a cluster has to be within 29% of it on
`(1 − mean) · (1 − std)`. The spec's own example still holds (500 probes at
0.1 score 3.54, 10 probes at 0.05 score 2.28).

I picked the cap on that bound rather than tuning it to these two clusters.
Here are the alternatives I checked. "Depth gaps first" means both Pest
Control and Fungal Disease rank above every compost and tomato cluster.
"Rank correlation" is between severity and the unanswered share, over the 26
non-noise clusters. The benchmark AUC is the probability that a random absent
cluster outranks a random present one, over seeds 0–2.

| size term | depth gaps first | rank correlation with unanswered share | benchmark AUC, absent over present |
|---|---|---|---|
| log(1 + n), spec | no | 0.863 | 0.988 |
| log(1 + min(n, 100)) | no | 0.863 | 0.988 |
| **log(1 + min(n, 50))** | **yes** | **0.902** | **0.968** |
| log(1 + min(n, 40)) | yes | 0.921 | 0.964 |
| sqrt(log(1 + n)) | yes | 0.920 | 0.980 |
| no size term | yes | 0.907 | 0.952 |

A cap at 100 changes nothing on this KB, because only two clusters are
bigger than that. Any cap at 50 or below, or the square-root compression,
fixes the ordering and tracks the unanswered share better than the spec
formula. The cost is a slightly lower benchmark AUC (0.988 → 0.968): some
large absent clusters lose a little of their lead. The square root keeps more
of that AUC, but its size effect is unbounded. With a cap, a 5,000-probe
cluster can't dominate the list.

Before and after, on the full synthetic KB (zones use the calibrated
thresholds):

| cluster | size | mean | zone | unanswered | severity before (rank) | severity after (rank) |
|---|---|---|---|---|---|---|
| Cryptocurrency Tax Reporting Requirements | 40 | 0.231 | DARK | 100% | 2.74 (1) | 2.74 (1) |
| Orbital Mechanics and Parameters | 39 | 0.251 | DARK | 100% | 2.62 (2) | 2.62 (2) |
| Mushroom Cultivation Methods and Environments | 39 | 0.335 | THIN | 95% | 2.28 (4) | 2.28 (3) |
| Water Quality and Nutrient Management | 28 | 0.311 | DARK | 100% | 2.25 (5) | 2.25 (4) |
| Pruning Perennial Herbs Safely | 66 | 0.452 | ADEQUATE | 53% | 2.01 (6) | 1.88 (5) |
| **Vegetable Garden Pest Control** | 47 | 0.444 | ADEQUATE | 62% | 1.87 (7) | **1.87 (6)** |
| noise (unclustered) | 140 | 0.455 | ADEQUATE | 62% | 2.29 (3) | 1.82 (7) |
| **Fungal Disease Prevention in Crops** | 28 | 0.432 | ADEQUATE | 61% | 1.67 (10) | **1.67 (8)** |
| Vegetable Garden Watering Requirements | 94 | 0.523 | ADEQUATE | 43% | 1.83 (8) | 1.58 (9) |
| Compost Pile Temperature Management | 134 | 0.579 | ADEQUATE | 41% | 1.80 (9) | 1.44 (13) |
| Indeterminate Tomato Support Systems | 61 | 0.588 | ADEQUATE | 28% | 1.50 (13) | 1.43 (15) |
| Composting Materials Guidelines | 39 | 0.579 | ADEQUATE | 36% | 1.39 (16) | 1.39 (16) |
| Late Blight Identification and Symptoms | 28 | 0.572 | ADEQUATE | 21% | 1.27 (22) | 1.27 (22) |
| Tomato Planting Soil Temperature | 29 | 0.557 | ADEQUATE | 38% | 1.27 (23) | 1.27 (23) |
| Carbon-to-Nitrogen Ratio in Composting | 34 | 0.647 | ADEQUATE | 6% | 1.14 (26) | 1.14 (26) |
| Tomato Ripening Temperature Control | 41 | 0.678 | ADEQUATE | 7% | 1.10 (27) | 1.10 (27) |

(Clusters ranked 10–12, 14 and 17–21 and 24–25 are unchanged and omitted.)

Both depth gaps now rank above every compost and tomato cluster. The highest
of those, compost pile temperature, dropped from 9th to 13th. The noise
bucket also dropped from 3rd to 7th, which is right: it's 140 unrelated
leftovers, not one gap. Clusters of 50 probes or fewer keep exactly the same
severity. Only the large ones changed.

## 7. Recommendations: what the KB-internal check finds, and what it took

The report recommends documents for every DARK and THIN cluster plus the ten
most severe clusters of any tier (`src/reports/`). Candidates come from two
places. **KB-internal** candidates are passages already in the KB that
retrieval isn't reaching (spec §7 Strategy A). **External** candidates are web
search results for queries Haiku writes from the cluster's questions (Strategy
B). All candidates are ranked by simulated gain: the candidate's text replaces
the lowest-ranked retrieved chunk for every probe in the cluster, and the
probes are re-scored with the same three signals.

**Validated against the thin topics' ground truth.** The builder buried two
passages on purpose, and the KB-internal check handles both correctly:

- *Hydroponics*: it recommends exactly the buried passage,
  `seeds-01-indoor#1` ("...the Kratky method, is just a jar of nutrient
  solution and a net cup..."). The passage matches the cluster at 0.517, while
  the seed-starting chunk it sits in scores only 0.371 and is retrieved for 14%
  of the cluster's probes. The cross-encoder says it answers 3 of the 28
  questions.
- *Mushrooms*: it does **not** recommend re-chunking the mulch passage, which
  is also right. That chunk is already retrieved for 87% of the cluster's
  probes. The content is found; there just isn't enough of it. That's a case
  for new documents, not re-chunking.
- *Crypto taxes and orbital mechanics*: nothing is recommended, because the
  best matching KB passage is at 0.13.

**Similarity alone wasn't enough.** The first version flagged any passage
that matched the cluster (≥ 0.45), clearly beat its own chunk (by ≥ 0.10), and
was retrieved for under half the probes. That found the hydroponics passage,
but it also recommended generic sentences for unrelated clusters: "It starts
with prevention: healthy soil, resistant varieties..." for watering,
crop-rotation and square-foot-gardening clusters, often with a *negative*
simulated gain. Adding one requirement fixed it: the cross-encoder has to say
the passage answers at least one of the cluster's questions. Every generic
passage answered 0. The hydroponics and mushroom-substrate passages answered 3
and 2. Candidates whose simulated gain isn't positive are also dropped.

**Not yet exercised end to end: web search.** External recommendations use
Claude's web search tool by default (or Brave with `BRAVE_API_KEY`), and the
parsing and ranking are covered by tests with a fake backend. The API credit
ran out before they could run on the synthetic KB, so the current report has
only the two KB-internal recommendations. Its estimated improvement is
correspondingly small: 0.504 → 0.505 overall after applying them. The
depth-gap clusters (Pest Control, Fungal Disease) get no recommendations yet,
because nothing in the KB answers their questions. That's exactly what the
web search leg is for.

**Spec §10 detection targets.** `umbra benchmark calibrate` now scores the
calibrated thresholds against the spec's own targets. (Superseded by section
8, which gets recall above the minimum.) On seeds 0–2 at min_cluster_size 20,
averaged per seed:

| metric | value | target | acceptable minimum | verdict |
|---|---|---|---|---|
| precision | 0.78 | ≥ 0.80 | ≥ 0.70 | acceptable |
| recall | 0.58 | ≥ 0.85 | ≥ 0.75 | **below minimum** |
| false positive rate | 0.00 | ≤ 0.15 | ≤ 0.25 | target |

Recall is the weak spot, for the reasons in section 4. Three of the 12
injected absent topics never formed a labeled cluster (their probes scattered
into mixed clusters or noise). Two more scored above the dark line, because
related content was still in the KB. Nothing is falsely flagged, but a
quarter of injected gaps go unseen at the cluster level. Their probes are
still scored and are visible on the map and in the per-probe CSV.

## 8. Recall below the spec's minimum: diagnosis and fix

At min_cluster_size 20, the calibrated thresholds detected 7 of the 12
injected absent topics across seeds 0–2. That's a spec §10 recall of 0.58,
against a target of 0.85 and a minimum of 0.75. Recall on known gaps is the
number this project is supposed to be good at, so every miss was traced.

### Where each of the 5 misses went

A topic counts as detected when a cluster labeled with it (at least 5 kb_blind
probes, at least 60% of them about the topic) is DARK. Each missed topic falls
into one of two buckets:

| seed | topic | kb_blind probes | bucket | what happened |
|---|---|---|---|---|
| 1 | tools | 30 | (a) no cluster of its own | 29 probes merged with 29 fruit_trees probes into one 168-probe cluster ("pruners" and "pruning"), mean 0.481 |
| 1 | tomato_growing | 30 | (a) no cluster of its own | split: 9 with pest_management, 7 in noise, 5 with tools/fruit trees |
| 2 | season_extension | 29 | (a) no cluster of its own | 18 merged with 19 raised_beds probes (a *thin* topic that seed). The merged cluster **is DARK (0.298)**, but at 18/38 its purity is below the 0.6 labeling rule, so it isn't credited to either topic |
| 0 | seed_starting | 30 | (b) own cluster, above the dark line | cluster mean 0.482 (ADEQUATE) |
| 0 | irrigation | 30 | (b) own cluster, above the dark line | cluster mean 0.358 (THIN) |

So 3 of the misses are bucket (a) and 2 are bucket (b).

### Bucket (a): not a raw probe shortage, but a density problem

None of the three was short of probes in raw terms: each had 29–30 kb_blind
probes, above min_cluster_size 20. The problem is what those probes land next
to. Tools and season extension sit semantically close to a neighboring topic,
and HDBSCAN's excess-of-mass selection prefers one bigger cluster over two
smaller ones. Tomato questions are broad (disease, pests, pruning, harvest),
so 30 of them spread thin across several neighborhoods.

Two levers address that: more probes per topic, so each topic's region is
denser, or a smaller min_cluster_size, so smaller regions count as clusters.
Both were tested by re-clustering the already-scored probes (no new LLM
calls). Probe counts can only be tested *below* the 30 generated per topic,
by subsampling. Thresholds were held at the old 0.324 / 0.400 for this sweep:

| min_cluster_size | kb_blind probes per topic | absent topics with own cluster | recall | precision | FPR |
|---|---|---|---|---|---|
| 20 | 15 | 0.50 | 0.25 | 0.50 | 0.00 |
| 20 | 20 | 0.50 | 0.25 | 0.33 | 0.00 |
| 20 | 25 | 0.75 | 0.58 | 1.00 | 0.00 |
| 20 | 30 (as generated) | 0.75 | 0.58 | 0.78 | 0.00 |
| 15 | 15 | 0.67 | 0.33 | 0.61 | 0.00 |
| 15 | 20 | 0.67 | 0.42 | 0.83 | 0.00 |
| 15 | 25 | 0.83 | 0.67 | 0.81 | 0.00 |
| **15** | **30** | **0.92** | **0.83** | **0.72** | **0.00** |
| 10 | 30 | 1.00 | 0.92 | 0.66 | 0.00 |

The min_cluster_size 20, 30-probe row reproduces the calibration numbers
exactly, which checks the re-clustering. Two things stand out:

- **Recall rises with probes per topic.** At min_cluster_size 15, recall goes
  0.33 → 0.42 → 0.67 → 0.83 as probes go from 15 to 30, so a missing topic
  does need enough questions to form a dense region. The trend suggests more
  than 30 per topic would help further, but that needs new generation and is
  untested (the API credit ran out).
- **Smaller clusters help, down to a point.** 10 recovers every topic's
  cluster but costs precision (0.66, below the 0.70 minimum): more small
  clusters means more thin topics come out DARK.

One sweep at fixed thresholds isn't enough to pick a setting. The thresholds
were fit at 20, one topic moves recall by 0.083, and the UMAP layout is a
single random draw. So I compared 15 and 20 properly: leave-one-seed-out,
with thresholds re-fit on the other seeds for each setting, repeated over
three UMAP random states (42, 7, 123), 9 held-out evaluations each:

| min_cluster_size | held-out recall | held-out precision | FPR | recall by UMAP layout | precision by UMAP layout |
|---|---|---|---|---|---|
| 20 | 0.53 | 0.67 | 0.00 | 0.58 / 0.58 / 0.42 | 0.64 / 0.89 / 0.47 |
| **15** | **0.81** | **0.77** | **0.00** | **0.83 / 0.83 / 0.75** | **0.72 / 0.74 / 0.85** |

At 15, recall is above the 0.75 minimum in every layout, and precision goes
*up*, not down. At 20, precision swings from 0.47 to 0.89 depending on the
layout; at 15 it stays between 0.72 and 0.85. The smaller setting isn't just
better on average, it's more stable.

### Bucket (b): the remaining KB still answers some of the questions

For both bucket-(b) topics, the likely cause is the same one behind
hydroponics and mushrooms: related content still in the KB pulls the score
up. I checked this with two measures per absent topic, both computed without
reference to the score:

- **residual overlap**: for each chunk of the removed documents, the best
  cosine similarity to any chunk left in the reduced KB, averaged
- **still answered**: the share of the topic's kb_blind probes the
  cross-encoder says are answered by what the RAG system retrieved from the
  reduced KB (HP < 0.5, or a preliminary score above the HP band)

| seed | absent topic | residual overlap | still answered | probe mean CS |
|---|---|---|---|---|
| 0 | **seed_starting** | **0.540** | **30%** | **0.416** |
| 0 | cover_crops | 0.486 | 3% | 0.317 |
| 0 | season_extension | 0.484 | 7% | 0.327 |
| 1 | tools | 0.475 | 0% | 0.291 |
| 2 | garden_planning | 0.466 | 0% | 0.309 |
| 2 | season_extension | 0.463 | 3% | 0.313 |
| 2 | mulch | 0.459 | 0% | 0.324 |
| 0 | **irrigation** | 0.444 | **10%** | **0.340** |
| 2 | tomato_growing | 0.434 | 3% | 0.310 |
| 1 | seed_starting | 0.427 | 0% | 0.307 |
| 1 | tomato_growing | 0.398 | 3% | 0.311 |
| 1 | composting | 0.392 | 0% | 0.299 |

Across the 12 absent topics, the "still answered" share predicts the score
strongly (Spearman 0.81). Raw embedding overlap predicts it only moderately
(0.47). What drags a removed topic's score up isn't text that *looks* similar;
it's text that still *answers* some of its questions. The two bucket-(b)
misses have the two highest answered shares.

There's a clean natural experiment in the data. seed_starting was removed in
both seed 0 and seed 1. In seed 0 the tomato seed-starting document
(`tomato-02-starting-seeds`: sowing depth, heat mats, grow lights) stayed in
the KB. 30% of the questions were still answered, and the cluster scored
0.482. In seed 1, tomatoes were removed too, taking that document with them.
0% were answered, and the cluster scored 0.312, well inside DARK.

For reference, across all 42 topic instances: absent topics average 5% still
answered, thin 9%, present 50%.

### What I changed, and what I didn't

**Changed: min_cluster_size is now 15** (the spec says 20, and so did the
Phase 2 instructions). `DEFAULT_MIN_CLUSTER_SIZE` in
`src/clustering/hdbscan_clusterer.py` is used by `umbra audit`, the
benchmark, and every tool that clusters. The finished seeds were re-clustered
from their scored probes (`umbra benchmark recluster`), and the thresholds
were re-fit on the result:

| | min_cluster_size 20 | min_cluster_size 15 |
|---|---|---|
| labeled clusters (absent / thin / present) | 46 (9 / 9 / 28) | 59 (12 / 11 / 36) |
| absent topics with a labeled cluster | 9 / 12 | 12 / 12 |
| fitted thresholds (dark_below / adequate_above) | 0.324 / 0.400 | **0.334 / 0.408** |
| threshold spread under leave-one-seed-out | 0.322–0.364 / 0.372–0.400 | 0.332–0.335 / 0.407–0.408 |
| 3-tier accuracy, in-sample | 89% | 90% |
| 3-tier accuracy, held-out seeds | 73% / 89% / 85% | 78% / 96% / 93% |
| spec §10 recall | 0.58 (below minimum) | **0.83 (acceptable; target 0.85)** |
| spec §10 precision | 0.78 (acceptable) | 0.72 (acceptable) |
| spec §10 FPR | 0.00 | 0.00 |

The spec thresholds at min_cluster_size 15, for comparison: recall 0.25,
precision 0.67, 31% three-tier accuracy.

Recall now clears the spec's minimum. It's 0.02 short of the target, and
precision is closer to its floor than it was. The precision cost is four thin
clusters that now read DARK: fruit_trees (0.310), soil (0.318), irrigation
(0.304), and pest_management (0.289). The remaining KB answers 0–10% of their
questions. They are "thin" in the ground truth, but a two-sentence mention
behaves almost exactly like no mention at all (section 4's absent/thin
overlap). The false-positive rate on *present* topics is still zero: all 36
present clusters are ADEQUATE.

The other results were re-checked at the new setting and still hold:

- **KB-anchored strategies still surface 0 of 12 injected gaps without
  kb_blind** (section 5). With all four strategies, 11 of 12 surface and 10
  are DARK. kb_blind alone now gets 8 surfaced and 6 DARK, up from 2 and 2,
  because smaller clusters let its probes split by topic. The combination is
  still what works.
- **Purity still doesn't break the DARK/THIN tie** (section 4). Score only:
  held-out dark F1 0.78, accuracy 89%. Purity rule: held-out dark F1 0.60,
  accuracy 78%. Absent and thin clusters have the same purity (0.65 vs 0.66).

**Not changed: the two remaining misses.**

- *Seed 0 seed_starting* (bucket b, now 0.392, THIN): 30% of its questions
  are still answered from the tomato seed-starting document. Calling this
  topic thin rather than dark is arguably the correct reading of that KB.
  "Fixing" it would mean making the scorer ignore content that does answer
  the question. I'd rather report it as a limit of the benchmark's
  ground truth: removing a topic's documents doesn't remove the topic.
- *Seed 2 season_extension* (bucket a): still merged with the thin
  raised_beds topic in a DARK cluster (0.298) at 0.50 purity. A user of the
  report would see these questions in a dark zone. The benchmark just can't
  credit the detection to one topic. The labeling rule is part of the
  evaluation, not the product, so it stays at 60% purity rather than being
  loosened to make the number better.

A probe-level view makes the season_extension point concrete. Count a topic
as detected when most of its kb_blind probes sit in DARK clusters, however
those clusters are labeled. On that view, recall at min_cluster_size 20 was
0.67 instead of 0.58. The headline number stays the cluster-level one,
because that's the spec's definition.

**Still to do when the API credit is back.** Generate more than 30 kb_blind
probes per topic (`umbra benchmark run --per-topic 45` on fresh seeds) and
check whether the upward trend continues. Also re-run seeds 3–4, which will
cluster at 15 automatically. The Phase 2 full-KB audit in `out/phase2.csv`
was clustered at 20; re-clustering it at 15 needs new cluster names from
Haiku, so the report and dashboard for it stay on the old clustering (with
the new thresholds) until then.

The numbers before this change are reproducible from `out/benchmark_mcs20/`.
The sweep and the held-out comparison are
`umbra benchmark recall --out out/benchmark` and `... --refit`.

### Reproducing

```
python -m src.data.synthetic_kb_builder
UMBRA_KB_PATH=data/synthetic_kb uvicorn src.connectors.mock_rag_server:app --port 8765

umbra audit --endpoint http://localhost:8765/query --kb-path data/synthetic_kb \
    --strategies taxonomy --n-probes 1000 --no-cluster --output out/audit_dispersion.csv
for m in dispersion spec; do
  umbra audit --endpoint http://localhost:8765/query --kb-path data/synthetic_kb \
      --strategies none --probes-file data/validation/validation_probes.jsonl \
      --no-cluster --se-method $m --output out/labeled_$m.csv
done
python scripts/validate_phase1.py report \
    --labeled dispersion=out/labeled_dispersion.csv --labeled spec=out/labeled_spec.csv \
    --taxonomy out/audit_dispersion.csv

# finding 3
umbra audit --endpoint http://localhost:8765/query --kb-path data/synthetic_kb --n-probes 1000 \
    --probes-file data/validation/validation_probes.jsonl --output out/phase2.csv
python scripts/validate_phase2.py out/phase2.csv

# findings 4 to 7
umbra benchmark run --seeds 0,1,2,3,4 --out out/benchmark      # resumable
umbra benchmark calibrate --out out/benchmark --write-thresholds out/thresholds.json
umbra benchmark ablate --out out/benchmark
umbra report --audit out/phase2.csv --kb-path data/synthetic_kb --output out/report.json --web-search none
umbra dashboard --report out/report.json
```

Probe generation goes through an LLM, so reruns won't reproduce these numbers
exactly. The KB-blind probe set is checked in so that part stays fixed.

The 20-case table comes straight from `pytest tests/test_scoring.py`. The spec
formula's five misses are pinned as strict xfails, so the comparison can't
quietly drift.
