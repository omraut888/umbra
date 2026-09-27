"""Why do injected absent topics get missed? Sensitivity of recall to clustering choices.

    umbra benchmark recall --out out/benchmark

Re-clusters each finished seed's already-scored probes (no LLM calls) under
different min_cluster_size values and kb_blind probe counts (subsampled from
the 30 generated per topic, so only fewer-than-generated can be tested), and
reports spec §10 recall / precision / FPR at fixed zone thresholds.

Two recall numbers are reported:
  cluster recall  the spec §10 definition used everywhere else: some cluster
                  labeled with the topic (>= 5 kb_blind probes, >= 60% of
                  them about it) is DARK
  probe recall    the majority of the topic's kb_blind probes sit in DARK
                  clusters, whatever those clusters are labeled. This counts
                  a gap flagged inside a mixed cluster, which cluster recall
                  doesn't.
"""

from __future__ import annotations

import csv
import json
import random
import statistics as st
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from src.benchmark.gap_injection import MIN_LABELED, MIN_PURITY
from src.clustering.hdbscan_clusterer import NOISE, identify_clusters
from src.clustering.umap_projector import project_for_clustering
from src.clustering.zones import DEFAULT_THRESHOLDS, ZoneThresholds, classify_zone
from src.embeddings import embed


def labeled_clusters(rows: List[dict], embs: np.ndarray, tiers: Dict[str, str], min_cluster_size: int,
                     umap_state: int) -> List[dict]:
    """Cluster, then label each cluster the way the benchmark does (by its kb_blind probes)."""
    labels = identify_clusters(project_for_clustering(embs, umap_state), min_cluster_size=min_cluster_size).labels
    scores = np.array([float(r["coverage_score"]) for r in rows])
    out = []
    for cid in set(labels.tolist()) - {NOISE}:
        idx = np.flatnonzero(labels == cid)
        blind = Counter(rows[i]["probe_topic"] for i in idx if rows[i]["generation_strategy"] == "kb_blind")
        topic = None
        if blind:
            t, n = blind.most_common(1)[0]
            if sum(blind.values()) >= MIN_LABELED and n / sum(blind.values()) >= MIN_PURITY:
                topic = t
        out.append({"cluster_id": cid, "mean_cs": float(scores[idx].mean()), "topic": topic,
                    "tier": tiers[topic] if topic else None})
    return out


def refit_comparison(out_dir: Path, sizes=(15, 20), umap_states=(42, 7, 123)) -> dict:
    """Leave-one-seed-out with thresholds re-fit per setting, over several UMAP layouts."""
    from src.benchmark.calibrate import fit

    seeds = []
    for path in sorted(out_dir.glob("seed_*/result.json")):
        result = json.loads(path.read_text())
        rows = [r for r in csv.DictReader(open(path.parent / "probes.csv")) if r["coverage_score"]]
        seeds.append((result, rows, embed([r["query"] for r in rows])))
    out = {}
    for mcs in sizes:
        runs = []
        for state in umap_states:
            clusters = {r["seed"]: labeled_clusters(rows, embs, r["tiers"], mcs, state) for r, rows, embs in seeds}
            for r, _, _ in seeds:
                train = [c for s, cs in clusters.items() if s != r["seed"] for c in cs if c["tier"]]
                th = fit(train)
                test = clusters[r["seed"]]
                absent = [u for u, t in r["tiers"].items() if t == "absent"]
                present = [u for u, t in r["tiers"].items() if t == "present"]
                dark = [c for c in test if c["tier"] and classify_zone(c["mean_cs"], th) == "DARK"]
                found = {c["topic"] for c in dark}
                runs.append({
                    "umap_state": state, "seed": r["seed"], "thresholds": [th.dark_below, th.adequate_above],
                    "recall": len(found & set(absent)) / len(absent),
                    "precision": sum(c["tier"] == "absent" for c in dark) / len(dark) if dark else 0.0,
                    "fpr": len(found & set(present)) / len(present),
                })
        out[mcs] = {
            "recall": st.fmean(x["recall"] for x in runs), "precision": st.fmean(x["precision"] for x in runs),
            "fpr": st.fmean(x["fpr"] for x in runs),
            "recall_by_layout": {s: st.fmean(x["recall"] for x in runs if x["umap_state"] == s) for s in umap_states},
            "precision_by_layout": {s: st.fmean(x["precision"] for x in runs if x["umap_state"] == s) for s in umap_states},
            "dark_below_range": [min(x["thresholds"][0] for x in runs), max(x["thresholds"][0] for x in runs)],
            "adequate_above_range": [min(x["thresholds"][1] for x in runs), max(x["thresholds"][1] for x in runs)],
            "runs": runs,
        }
    return out


def evaluate_seed(rows: List[dict], embs: np.ndarray, tiers: Dict[str, str], min_cluster_size: int,
                  thresholds: ZoneThresholds, per_topic: Optional[int] = None, seed: int = 0) -> dict:
    keep = list(range(len(rows)))
    if per_topic is not None:
        rng = random.Random(seed)
        by_topic: Dict[str, List[int]] = {}
        for i, r in enumerate(rows):
            if r["generation_strategy"] == "kb_blind":
                by_topic.setdefault(r["probe_topic"], []).append(i)
        dropped = {i for idx in by_topic.values() if len(idx) > per_topic for i in rng.sample(idx, len(idx) - per_topic)}
        keep = [i for i in keep if i not in dropped]
    labels = identify_clusters(project_for_clustering(embs[keep]), min_cluster_size=min_cluster_size).labels
    scores = np.array([float(rows[i]["coverage_score"]) for i in keep])
    zone_of, topic_of = {}, {}
    for cid in set(labels.tolist()) - {NOISE}:
        idx = np.flatnonzero(labels == cid)
        zone_of[cid] = classify_zone(float(scores[idx].mean()), thresholds)
        blind = Counter(rows[keep[i]]["probe_topic"] for i in idx if rows[keep[i]]["generation_strategy"] == "kb_blind")
        if blind:
            topic, n = blind.most_common(1)[0]
            if sum(blind.values()) >= MIN_LABELED and n / sum(blind.values()) >= MIN_PURITY:
                topic_of[cid] = topic

    absent = [u for u, t in tiers.items() if t == "absent"]
    present = [u for u, t in tiers.items() if t == "present"]
    dark_labeled = [c for c in topic_of if zone_of[c] == "DARK"]
    detected = {topic_of[c] for c in dark_labeled}

    def probe_dark_share(u):
        mine = [labels[j] for j, i in enumerate(keep)
                if rows[i]["generation_strategy"] == "kb_blind" and rows[i]["probe_topic"] == u]
        return sum(zone_of.get(c) == "DARK" for c in mine) / max(1, len(mine))

    return {
        "clusters": len(zone_of),
        "cluster_recall": len(detected & set(absent)) / len(absent),
        "probe_recall": sum(probe_dark_share(u) > 0.5 for u in absent) / len(absent),
        "precision": sum(tiers[topic_of[c]] == "absent" for c in dark_labeled) / len(dark_labeled) if dark_labeled else 0.0,
        "fpr": len(detected & set(present)) / len(present),
        "probe_fpr": sum(probe_dark_share(u) > 0.5 for u in present) / len(present),
        "own_cluster": sum(u in topic_of.values() for u in absent) / len(absent),
        "unlabeled_dark": sum(1 for c, z in zone_of.items() if z == "DARK" and c not in topic_of),
    }


def run(out_dir: Path, thresholds: ZoneThresholds = DEFAULT_THRESHOLDS,
        sizes=(10, 15, 20), per_topics=(15, 20, 25, None)) -> dict:
    seeds = []
    for path in sorted(out_dir.glob("seed_*/result.json")):
        result = json.loads(path.read_text())
        rows = [r for r in csv.DictReader(open(path.parent / "probes.csv")) if r["coverage_score"]]
        seeds.append((result, rows, embed([r["query"] for r in rows])))

    table = {}
    for mcs in sizes:
        for pt in per_topics:
            per_seed = [evaluate_seed(rows, embs, r["tiers"], mcs, thresholds, pt, seed=r["seed"]) for r, rows, embs in seeds]
            table[f"min_cluster_size={mcs}, kb_blind/topic={pt or 30}"] = {
                k: st.fmean(x[k] for x in per_seed) for k in per_seed[0]
            }
    return {"thresholds": [thresholds.dark_below, thresholds.adequate_above], "n_seeds": len(seeds), "table": table}


def main() -> None:
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "out/benchmark")
    if "--refit" in sys.argv:
        r = refit_comparison(out_dir)
        (out_dir / "recall_refit.json").write_text(json.dumps(r, indent=2, default=str) + "\n")
        print("Held-out (leave-one-seed-out, thresholds re-fit per setting), mean over seeds x UMAP layouts")
        for mcs, v in r.items():
            print(f"  min_cluster_size={mcs}: recall {v['recall']:.2f}  precision {v['precision']:.2f}  FPR {v['fpr']:.2f}"
                  f"   recall by layout {({k: round(x, 2) for k, x in v['recall_by_layout'].items()})}"
                  f"   precision by layout {({k: round(x, 2) for k, x in v['precision_by_layout'].items()})}"
                  f"   dark_below {v['dark_below_range'][0]:.3f}-{v['dark_below_range'][1]:.3f}")
        return
    s = run(out_dir)
    (out_dir / "recall_sensitivity.json").write_text(json.dumps(s, indent=2) + "\n")
    print(f"{s['n_seeds']} seeds, thresholds {s['thresholds'][0]:.3f}/{s['thresholds'][1]:.3f}, means over seeds\n")
    print(f"{'setting':<42}{'clusters':>9}{'own cl.':>8}{'cl. recall':>11}{'probe recall':>13}{'precision':>10}"
          f"{'FPR':>6}{'probe FPR':>10}{'unlab. dark':>12}")
    for name, v in s["table"].items():
        print(f"{name:<42}{v['clusters']:>9.1f}{v['own_cluster']:>8.2f}{v['cluster_recall']:>11.2f}{v['probe_recall']:>13.2f}"
              f"{v['precision']:>10.2f}{v['fpr']:>6.2f}{v['probe_fpr']:>10.2f}{v['unlabeled_dark']:>12.1f}")


if __name__ == "__main__":
    main()
