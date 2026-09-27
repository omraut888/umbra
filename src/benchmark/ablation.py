"""Which strategies are needed to surface an injected gap?

    python -m src.benchmark.ablation out/benchmark

Re-clusters each seed's already-scored probes using only some strategies. No
new LLM calls: a probe's coverage score doesn't depend on the other probes,
only the clustering does. Each probe carries a topic label taken from the
full-run cluster it sat in (kb_blind probes use their own topic). An ablated
cluster is attributed to the majority label of its members.

A topic counts as surfaced when some cluster is attributed to it at >= 60%
purity, and as detected when that cluster is also DARK under the given
thresholds.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from src.clustering.hdbscan_clusterer import NOISE, identify_clusters
from src.clustering.umap_projector import project_for_clustering
from src.clustering.zones import ZoneThresholds, classify_zone
from src.embeddings import embed

SUBSETS = {
    "all four": {"taxonomy", "adversarial", "counterfactual", "kb_blind"},
    "KB-anchored only": {"taxonomy", "adversarial", "counterfactual"},
    "adversarial only": {"adversarial"},
    "kb_blind only": {"kb_blind"},
}


def probe_labels(rows: List[dict], clusters: List[dict]) -> List[str | None]:
    topic_of_cluster = {str(c["cluster_id"]): c["topic"] for c in clusters}
    return [r["probe_topic"] if r["generation_strategy"] == "kb_blind" else topic_of_cluster.get(r["cluster_id"])
            for r in rows]


def surfaced_topics(labels: np.ndarray, members: Sequence[str | None], scores: np.ndarray,
                    thresholds: ZoneThresholds) -> Dict[str, str]:
    """topic -> best zone among clusters attributed to it."""
    order = {"DARK": 0, "THIN": 1, "ADEQUATE": 2}
    out: Dict[str, str] = {}
    for cid in set(labels.tolist()) - {NOISE}:
        idx = np.flatnonzero(labels == cid)
        named = Counter(members[i] for i in idx if members[i])
        if not named:
            continue
        topic, n = named.most_common(1)[0]
        if n / len(idx) < 0.6:
            continue
        zone = classify_zone(float(scores[idx].mean()), thresholds)
        if topic not in out or order[zone] < order[out[topic]]:
            out[topic] = zone
    return out


def run(out_dir: Path, thresholds: ZoneThresholds) -> dict:
    table = defaultdict(lambda: {"absent": 0, "surfaced": 0, "detected": 0})
    per_seed = {}
    for result_path in sorted(out_dir.glob("seed_*/result.json")):
        result = json.loads(result_path.read_text())
        rows = [r for r in csv.DictReader(open(result_path.parent / "probes.csv")) if r["coverage_score"]]
        labels = probe_labels(rows, result["clusters"])
        absent = [u for u, t in result["tiers"].items() if t == "absent"]
        embs_all = embed([r["query"] for r in rows])
        per_seed[result["seed"]] = {}
        for name, strategies in SUBSETS.items():
            keep = [i for i, r in enumerate(rows) if r["generation_strategy"] in strategies]
            clustering = identify_clusters(project_for_clustering(embs_all[keep]))
            scores = np.array([float(rows[i]["coverage_score"]) for i in keep])
            found = surfaced_topics(clustering.labels, [labels[i] for i in keep], scores, thresholds)
            t = table[name]
            t["absent"] += len(absent)
            t["surfaced"] += sum(u in found for u in absent)
            t["detected"] += sum(found.get(u) == "DARK" for u in absent)
            per_seed[result["seed"]][name] = {u: found.get(u) for u in absent}
    return {"thresholds": [thresholds.dark_below, thresholds.adequate_above], "totals": dict(table), "per_seed": per_seed}


def main() -> None:
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "out/benchmark")
    calibrated = json.loads((out_dir / "calibration.json").read_text())["calibrated"]
    summary = run(out_dir, ZoneThresholds(*calibrated))
    (out_dir / "ablation.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Injected absent topics surfaced as their own cluster / detected as DARK "
          f"(thresholds {calibrated[0]:.3f}/{calibrated[1]:.3f})")
    for name, t in summary["totals"].items():
        print(f"  {name:<18} surfaced {t['surfaced']:>2}/{t['absent']}   dark {t['detected']:>2}/{t['absent']}")
    for seed, subsets in summary["per_seed"].items():
        print(f"  seed {seed}: " + "; ".join(f"{n}: {s}" for n, s in subsets.items()))


if __name__ == "__main__":
    main()
