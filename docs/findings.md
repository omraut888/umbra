# Findings

Things I found while building Umbra that change how the spec should be read.
Each one comes with the data that backs it up, so it can be rechecked.

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
embedding model in mind. I've left the spec's 0.30 / 0.60 in place for now.
The better fix is probably to calibrate the thresholds per embedding model
(e.g. from the RC distribution on known-good pairs) rather than hard-code them.
I'd do that against the gap-injection benchmark in the spec's section 10 rather
than tuning them on this one KB.

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
  the 0.60 adequate line. That's the ceiling effect from finding 2. By
  severity the tiers still come out in the right order: absent (2.74, 2.62),
  then thin (2.28, 2.25), then everything built on the full topics (1.10–1.80).
- The adversarial topic parser accepted markdown headings ("# Boundary Topics")
  as topic names, which accounted for 9 of the 1218 probes. Fixed after this
  run.

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
```

Probe generation goes through an LLM, so reruns won't reproduce these numbers
exactly. The KB-blind probe set is checked in so that part stays fixed.

The 20-case table comes straight from `pytest tests/test_scoring.py`. The spec
formula's five misses are pinned as strict xfails, so the comparison can't
quietly drift.
