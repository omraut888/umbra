"""Fit zone thresholds on gap-injection results and compare them with the spec's.

    python -m src.benchmark.calibrate out/benchmark

Each threshold is picked where it best separates two groups of labeled
clusters (by F1), not at a round number:
  dark_below      absent clusters vs thin + present
  adequate_above  present clusters vs absent + thin
Stability is checked with leave-one-seed-out: fit on four seeds, score the fifth.
"""

from __future__ import annotations

import json
import statistics as st
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from src.clustering.zones import SPEC_THRESHOLDS, ZoneThresholds, classify_zone

EXPECTED = {"absent": "DARK", "thin": "THIN", "present": "ADEQUATE"}


def load(out_dir: Path) -> List[dict]:
    return [json.loads(p.read_text()) for p in sorted(out_dir.glob("seed_*/result.json"))]


def prf(scores: Sequence[float], positive: Sequence[bool], t: float, below: bool) -> Tuple[float, float, float]:
    pred = [(s < t) if below else (s > t) for s in scores]
    tp = sum(p and y for p, y in zip(pred, positive))
    fp = sum(p and not y for p, y in zip(pred, positive))
    fn = sum(y and not p for p, y in zip(pred, positive))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def fit_threshold(scores: Sequence[float], positive: Sequence[bool], below: bool) -> float:
    xs = sorted(set(scores))
    candidates = [(a + b) / 2 for a, b in zip(xs, xs[1:])] or xs
    f1s = [prf(scores, positive, t, below)[2] for t in candidates]
    best = max(f1s)
    # several cut points can tie on the same F1; take the middle of that run
    # so the threshold sits in the gap between the classes, not against one edge
    tied = [t for t, f in zip(candidates, f1s) if f == best]
    return tied[len(tied) // 2]


def fit(clusters: Sequence[dict]) -> ZoneThresholds:
    scores = [c["mean_cs"] for c in clusters]
    dark = fit_threshold(scores, [c["tier"] == "absent" for c in clusters], below=True)
    adequate = fit_threshold(scores, [c["tier"] == "present" for c in clusters], below=False)
    return ZoneThresholds(dark, max(dark, adequate))


def accuracy(clusters: Sequence[dict], th: ZoneThresholds) -> float:
    return sum(classify_zone(c["mean_cs"], th) == EXPECTED[c["tier"]] for c in clusters) / len(clusters)


def spec10_metrics(result: dict, th: ZoneThresholds) -> Dict[str, float]:
    """Spec §10 detection metrics for one seed: a topic counts as detected when
    at least one cluster labeled with it is DARK."""
    labeled = [c for c in result["clusters"] if c["tier"]]
    dark = [c for c in labeled if classify_zone(c["mean_cs"], th) == "DARK"]
    absent = [u for u, t in result["tiers"].items() if t == "absent"]
    present = [u for u, t in result["tiers"].items() if t == "present"]
    dark_topics = {c["topic"] for c in dark}
    return {
        "precision": sum(c["tier"] == "absent" for c in dark) / len(dark) if dark else 0.0,
        "recall": len(dark_topics & set(absent)) / len(absent),
        "fpr": len(dark_topics & set(present)) / len(present),
    }


def summarize(results: List[dict]) -> dict:
    clusters = [dict(c, seed=r["seed"]) for r in results for c in r["clusters"] if c["tier"]]
    calibrated = fit(clusters)

    loso = []
    for r in results:
        train = [c for c in clusters if c["seed"] != r["seed"]]
        test = [c for c in clusters if c["seed"] == r["seed"]]
        if not test or not train:
            continue
        th = fit(train)
        loso.append({"seed": r["seed"], "thresholds": [th.dark_below, th.adequate_above],
                     "accuracy": accuracy(test, th), "spec_accuracy": accuracy(test, SPEC_THRESHOLDS)})

    def by_tier(values):
        return {t: {"n": len(v), "mean": st.fmean(v), "min": min(v), "max": max(v),
                    "p10": sorted(v)[int(0.1 * (len(v) - 1))], "p90": sorted(v)[int(0.9 * (len(v) - 1))]}
                for t, v in values.items() if v}

    cluster_scores = {t: [c["mean_cs"] for c in clusters if c["tier"] == t] for t in EXPECTED}
    topic_scores = {t: [r["topic_means"][u] for r in results for u, tier in r["tiers"].items() if tier == t]
                    for t in EXPECTED}

    scores = [c["mean_cs"] for c in clusters]
    comparison = {}
    for name, th in (("spec", SPEC_THRESHOLDS), ("calibrated", calibrated)):
        comparison[name] = {
            "thresholds": [th.dark_below, th.adequate_above],
            "accuracy": accuracy(clusters, th),
            "dark": dict(zip(("precision", "recall", "f1"),
                             prf(scores, [c["tier"] == "absent" for c in clusters], th.dark_below, True))),
            "adequate": dict(zip(("precision", "recall", "f1"),
                                 prf(scores, [c["tier"] == "present" for c in clusters], th.adequate_above, False))),
            "confusion": {t: dict(Counter(classify_zone(c["mean_cs"], th) for c in clusters if c["tier"] == t))
                          for t in EXPECTED},
            "spec10_per_seed": {r["seed"]: spec10_metrics(r, th) for r in results},
        }

    absent_clusters = [c for c in clusters if c["tier"] == "absent"]
    mix = Counter()
    for c in absent_clusters:
        mix.update(c["strategy_mix"])
    strategy_totals = Counter()
    for r in results:
        strategy_totals.update(r["strategy_counts"])

    return {
        "n_seeds": len(results),
        "n_clusters": sum(len(r["clusters"]) for r in results),
        "n_labeled": len(clusters),
        "labeled_by_tier": dict(Counter(c["tier"] for c in clusters)),
        "cluster_scores": by_tier(cluster_scores),
        "topic_scores": by_tier(topic_scores),
        "comparison": comparison,
        "loso": loso,
        "absent_cluster_strategy_mix": dict(mix),
        "strategy_totals": dict(strategy_totals),
        "calibrated": [calibrated.dark_below, calibrated.adequate_above],
    }


def print_summary(s: dict) -> None:
    print(f"{s['n_seeds']} seeds, {s['n_clusters']} clusters, {s['n_labeled']} labeled {s['labeled_by_tier']}\n")
    for label, key in (("Cluster mean CS by true tier", "cluster_scores"), ("kb_blind topic mean CS by true tier", "topic_scores")):
        print(label)
        for t, d in s[key].items():
            print(f"  {t:<8} n={d['n']:<4} mean {d['mean']:.3f}  p10 {d['p10']:.3f}  p90 {d['p90']:.3f}  "
                  f"range {d['min']:.3f}-{d['max']:.3f}")
        print()
    print(f"{'':<11}{'thresholds':<14}{'3-tier acc':>11}{'dark P/R/F1':>20}{'adequate P/R/F1':>22}")
    for name, c in s["comparison"].items():
        d, a = c["dark"], c["adequate"]
        print(f"{name:<11}{c['thresholds'][0]:.3f}/{c['thresholds'][1]:.3f}  {c['accuracy']:>10.0%}"
              f"   {d['precision']:.2f}/{d['recall']:.2f}/{d['f1']:.2f}      {a['precision']:.2f}/{a['recall']:.2f}/{a['f1']:.2f}")
    for name, c in s["comparison"].items():
        print(f"\n{name} confusion (true tier -> predicted zone): {c['confusion']}")
        m = c["spec10_per_seed"].values()
        print(f"{name} spec §10 metrics, mean over seeds: precision {st.fmean(x['precision'] for x in m):.2f}, "
              f"recall {st.fmean(x['recall'] for x in m):.2f}, FPR {st.fmean(x['fpr'] for x in m):.2f}")
    print("\nLeave-one-seed-out")
    for row in s["loso"]:
        print(f"  seed {row['seed']}: fit {row['thresholds'][0]:.3f}/{row['thresholds'][1]:.3f} on the others -> "
              f"held-out accuracy {row['accuracy']:.0%} (spec {row['spec_accuracy']:.0%})")
    print(f"\nStrategy mix of probes in absent-labeled clusters: {s['absent_cluster_strategy_mix']}")
    print(f"All probes by strategy: {s['strategy_totals']}")


def main() -> None:
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "out/benchmark")
    s = summarize(load(out_dir))
    (out_dir / "calibration.json").write_text(json.dumps(s, indent=2) + "\n")
    print_summary(s)


if __name__ == "__main__":
    main()
