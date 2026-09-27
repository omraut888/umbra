"""Fit zone thresholds on gap-injection results and compare them with the spec's.

    python -m src.benchmark.calibrate out/benchmark

Each threshold is picked where it best separates two groups of labeled
clusters (by F1), not at a round number:
  dark_below      absent clusters vs thin + present
  adequate_above  present clusters vs absent + thin
Stability is checked with leave-one-seed-out: fit on four seeds, score the fifth.
"""

from __future__ import annotations

import csv
import json
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from src.clustering.zones import SPEC_THRESHOLDS, PurityRule, ZoneThresholds, classify_zone, label_purity
from src.data.synthetic_kb_builder import ALL_DOCS

DOC_TOPIC = {d.doc_id: d.topic for d in ALL_DOCS}

EXPECTED = {"absent": "DARK", "thin": "THIN", "present": "ADEQUATE"}
# spec §10 targets for dark-zone detection: (target, acceptable minimum)
SPEC10_TARGETS = {"precision": (0.80, 0.70), "recall": (0.85, 0.75), "fpr": (0.15, 0.25)}


def load(out_dir: Path) -> List[dict]:
    results = []
    for path in sorted(out_dir.glob("seed_*/result.json")):
        r = json.loads(path.read_text())
        add_purities(r, list(csv.DictReader(open(path.parent / "probes.csv"))))
        results.append(r)
    return results


def add_purities(result: dict, rows: List[dict]) -> None:
    """Two purity values per cluster.

    gt_purity uses the synthetic KB's ground truth: kb_blind probes carry their
    topic, and counterfactual probes trace to one through their source chunk's
    document. It can't be computed in a real audit, so it's an upper bound.
    blind_purity uses only kb_blind topic labels, which a real audit has
    whenever kb_blind runs.
    """
    members = defaultdict(list)
    for r in rows:
        if r["cluster_id"] != "":
            members[r["cluster_id"]].append(r)
    for c in result["clusters"]:
        ms = members[str(c["cluster_id"])]
        blind = [m["probe_topic"] if m["generation_strategy"] == "kb_blind" else None for m in ms]
        traced = [DOC_TOPIC.get(m["probe_topic"].split("#")[0]) if m["generation_strategy"] == "counterfactual" else b
                  for m, b in zip(ms, blind)]
        c["blind_purity"] = label_purity(blind)
        c["gt_purity"] = label_purity(traced)


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


def zone_of(c: dict, th: ZoneThresholds, key: Optional[str] = None, rule: Optional[PurityRule] = None) -> str:
    return classify_zone(c["mean_cs"], th, c.get(key) if key else None, rule)


def fit_purity_rule(clusters: Sequence[dict], th: ZoneThresholds, key: str) -> PurityRule:
    """Fit on the clusters the score already calls not-adequate: absent -> DARK."""
    pool = [c for c in clusters if c["mean_cs"] <= th.adequate_above and c[key] is not None]
    xs = [c[key] for c in pool]
    absent = [c["tier"] == "absent" for c in pool]
    best = None
    for dark_if_above in (True, False):
        # prf's "below" means positive when x < t, i.e. dark_if_above=False
        t = fit_threshold(xs, absent, below=not dark_if_above)
        f1 = prf(xs, absent, t, below=not dark_if_above)[2]
        if best is None or f1 > best[0]:
            best = (f1, PurityRule(t, dark_if_above))
    return best[1]


def tier_prf(clusters: Sequence[dict], th: ZoneThresholds, key=None, rule=None) -> Dict[str, Tuple[float, float, float]]:
    pred = [zone_of(c, th, key, rule) for c in clusters]
    out = {}
    for tier, zone in (("absent", "DARK"), ("thin", "THIN")):
        tp = sum(p == zone and c["tier"] == tier for p, c in zip(pred, clusters))
        n_pred = sum(p == zone for p in pred)
        n_true = sum(c["tier"] == tier for c in clusters)
        precision = tp / n_pred if n_pred else 0.0
        recall = tp / n_true if n_true else 0.0
        out[tier] = (precision, recall, 2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    out["accuracy"] = sum(p == EXPECTED[c["tier"]] for p, c in zip(pred, clusters)) / len(clusters)
    return out


def purity_comparison(clusters: List[dict], seeds: Sequence[int]) -> dict:
    variants = {"score only": None, "gt purity (upper bound)": "gt_purity", "kb_blind purity": "blind_purity"}
    out = {}
    th = fit(clusters)
    for name, key in variants.items():
        rule = fit_purity_rule(clusters, th, key) if key else None
        in_sample = tier_prf(clusters, th, key, rule)
        loso = []
        for seed in seeds:
            train = [c for c in clusters if c["seed"] != seed]
            test = [c for c in clusters if c["seed"] == seed]
            if not test:
                continue
            th_s = fit(train)
            rule_s = fit_purity_rule(train, th_s, key) if key else None
            loso.append(tier_prf(test, th_s, key, rule_s))
        out[name] = {
            "rule": None if rule is None else {"threshold": rule.threshold, "dark_if_above": rule.dark_if_above},
            "in_sample": in_sample,
            "loso_mean": {
                "absent_f1": st.fmean(x["absent"][2] for x in loso),
                "thin_f1": st.fmean(x["thin"][2] for x in loso),
                "accuracy": st.fmean(x["accuracy"] for x in loso),
            },
        }
    return out


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

    purity_by_tier = {
        key: {t: [c[key] for c in clusters if c["tier"] == t and c[key] is not None] for t in EXPECTED}
        for key in ("gt_purity", "blind_purity")
    }

    return {
        "purity": purity_comparison(clusters, [r["seed"] for r in results]),
        "purity_by_tier": {k: by_tier(v) for k, v in purity_by_tier.items()},
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
    print("\nPurity by true tier (labeled clusters)")
    for key, tiers in s["purity_by_tier"].items():
        for t, d in tiers.items():
            print(f"  {key:<13} {t:<8} n={d['n']:<3} mean {d['mean']:.2f}  p10 {d['p10']:.2f}  p90 {d['p90']:.2f}")
    print("\nDARK vs THIN inside not-adequate: score line vs purity rule")
    print(f"  {'':<25}{'rule':<22}{'dark P/R/F1':<18}{'thin P/R/F1':<18}{'3-tier acc':<11}{'LOSO dark F1':<14}{'LOSO thin F1':<14}LOSO acc")
    for name, v in s["purity"].items():
        ins, lo = v["in_sample"], v["loso_mean"]
        rule = "-" if v["rule"] is None else f"dark if {'>=' if v['rule']['dark_if_above'] else '<'} {v['rule']['threshold']:.3f}"
        fmt = lambda x: f"{x[0]:.2f}/{x[1]:.2f}/{x[2]:.2f}"
        print(f"  {name:<25}{rule:<22}{fmt(ins['absent']):<18}{fmt(ins['thin']):<18}{ins['accuracy']:<11.0%}"
              f"{lo['absent_f1']:<14.2f}{lo['thin_f1']:<14.2f}{lo['accuracy']:.0%}")
    if "spec10_check" in s:
        print("\nSpec §10 dark-zone detection targets (calibrated thresholds, mean over seeds)")
        for metric, v in s["spec10_check"].items():
            op = "<=" if metric == "fpr" else ">="
            print(f"  {metric:<10} {v['value']:.2f}   target {op} {v['target']:.2f}, minimum {op} {v['minimum']:.2f}"
                  f"  -> {v['verdict']}")
    print(f"\nStrategy mix of probes in absent-labeled clusters: {s['absent_cluster_strategy_mix']}")
    print(f"All probes by strategy: {s['strategy_totals']}")


def spec10_check(s: dict, which: str = "calibrated") -> Dict[str, dict]:
    per_seed = s["comparison"][which]["spec10_per_seed"].values()
    out = {}
    for metric, (target, minimum) in SPEC10_TARGETS.items():
        value = st.fmean(x[metric] for x in per_seed)
        if metric == "fpr":
            verdict = "target" if value <= target else "acceptable" if value <= minimum else "below minimum"
        else:
            verdict = "target" if value >= target else "acceptable" if value >= minimum else "below minimum"
        out[metric] = {"value": value, "target": target, "minimum": minimum, "verdict": verdict}
    return out


def write_thresholds(s: dict, path: Path) -> None:
    dark, adequate = s["calibrated"]
    path.write_text(json.dumps({
        "dark_below": round(dark, 4), "adequate_above": round(adequate, 4),
        "fit_on": f"{s['n_seeds']} gap-injection seeds, {s['n_labeled']} labeled clusters",
    }, indent=2) + "\n")


def calibrate(out_dir: Path, thresholds_out: Optional[Path] = None) -> dict:
    s = summarize(load(out_dir))
    s["spec10_check"] = spec10_check(s)
    (out_dir / "calibration.json").write_text(json.dumps(s, indent=2) + "\n")
    if thresholds_out:
        write_thresholds(s, thresholds_out)
    return s


def main() -> None:
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "out/benchmark")
    print_summary(calibrate(out_dir))


if __name__ == "__main__":
    main()
