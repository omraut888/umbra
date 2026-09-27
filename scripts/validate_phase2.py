"""Check a clustered audit against the synthetic KB's ground truth.

    python scripts/validate_phase2.py out/phase2.csv

Only the KB-blind probes (strategy "kb_blind") carry a ground-truth topic,
so they're the ones used to check where each topic ended up. Generated probes
show up in the strategy mix.
"""

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.clustering.zones import DEFAULT_THRESHOLDS, SPEC_THRESHOLDS, classify_zone  # noqa: E402

KB = Path("data/synthetic_kb")
EXPECTED_ZONE = {"full": "ADEQUATE", "thin": "THIN", "absent": "DARK"}


def main(path: str) -> None:
    probes = list(csv.DictReader(open(path)))
    clusters = list(csv.DictReader(open(Path(path).with_suffix(".clusters.csv"))))
    # zones are recomputed from each cluster's mean, so old runs can be read under new thresholds
    for c in clusters:
        c["spec_zone"] = classify_zone(float(c["mean_cs"]), SPEC_THRESHOLDS)
        c["zone"] = classify_zone(float(c["mean_cs"]), DEFAULT_THRESHOLDS)
    tiers = {k: v["tier"] for k, v in json.loads((KB / "ground_truth.json").read_text())["topics"].items()}

    gt_by_cluster = defaultdict(Counter)
    for p in probes:
        if p["generation_strategy"] in ("kb_blind", "ground_truth") and p["cluster_id"] != "":
            gt_by_cluster[p["cluster_id"]][p["probe_topic"]] += 1

    print(f"thresholds: spec {SPEC_THRESHOLDS.dark_below}/{SPEC_THRESHOLDS.adequate_above}, "
          f"calibrated {DEFAULT_THRESHOLDS.dark_below}/{DEFAULT_THRESHOLDS.adequate_above}\n")
    print(f"{'id':>4}  {'name':<38} {'size':>5} {'mean_cs':>8} {'spec tier':<10}{'calibrated':<11} {'severity':>8}  "
          f"{'strategies (tax/adv/cf/blind)':<27} ground-truth probes in cluster")
    for c in sorted(clusters, key=lambda c: -float(c["severity"])):
        mix = json.loads(c["strategy_mix"])
        mix["kb_blind"] = mix.get("kb_blind", 0) + mix.pop("ground_truth", 0)  # older runs used "ground_truth"
        mix_s = "/".join(str(mix.get(s, 0)) for s in ("taxonomy", "adversarial", "counterfactual", "kb_blind"))
        gt = ", ".join(f"{t}:{n}" for t, n in gt_by_cluster[c["cluster_id"]].most_common())
        print(f"{c['cluster_id']:>4}  {c['name'][:38]:<38} {c['query_count']:>5} {float(c['mean_cs']):>8.3f} "
              f"{c['spec_zone']:<10}{c['zone']:<11} {float(c['severity']):>8.2f}  {mix_s:<27} {gt}")

    zone_of = {c["cluster_id"]: c["zone"] for c in clusters}
    name_of = {c["cluster_id"]: c["name"] for c in clusters}
    print("\nWhere each ground-truth topic landed")
    for topic, tier in tiers.items():
        where = Counter(p["cluster_id"] for p in probes
                        if p["generation_strategy"] in ("kb_blind", "ground_truth") and p["probe_topic"] == topic)
        total = sum(where.values())
        top, n = where.most_common(1)[0]
        zones = Counter()
        for cid, k in where.items():
            zones[zone_of[cid]] += k
        ok = zones[EXPECTED_ZONE[tier]] / total
        print(f"  {topic:<25} {tier:<7} {n}/{total} in cluster {top} ({name_of[top]}, {zone_of[top]}); "
              f"{ok:.0%} of its probes in {EXPECTED_ZONE[tier]} clusters; zones {dict(zones)}")

    print("\nCluster zone of each strategy's probes")
    by_strategy = defaultdict(Counter)
    for p in probes:
        if p["cluster_id"] != "":
            strategy = "kb_blind" if p["generation_strategy"] == "ground_truth" else p["generation_strategy"]
            by_strategy[strategy][zone_of[p["cluster_id"]]] += 1
    for s, zc in sorted(by_strategy.items()):
        total = sum(zc.values())
        print(f"  {s:<15} n={total:<5} " + "  ".join(f"{z} {zc[z] / total:.0%}" for z in ("DARK", "THIN", "ADEQUATE")))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "out/phase2.csv")
