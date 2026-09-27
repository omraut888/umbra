"""Gap-injection benchmark (spec §10), extended with a thin tier.

For each seed, the 14 in-domain topics of the synthetic KB are split into:
  absent   every document of the topic is removed
  thin     all its documents are removed except a two-sentence passage, which
           gets buried in a document about some other topic (the same way the
           builder makes hydroponics and mushrooms thin)
  present  left alone

Then the full audit runs against a mock RAG server built on the reduced KB:
all four probe strategies, scoring, UMAP + HDBSCAN. Clusters are labeled by the
kb_blind probes inside them (the only probes that carry a topic label), which
gives a set of clusters with known tiers to calibrate the zone thresholds on.

    umbra benchmark run --seeds 0 1 2 3 4 --out out/benchmark
    umbra benchmark calibrate --out out/benchmark --write-thresholds thresholds.json
    umbra benchmark ablate --out out/benchmark

Seeds that already have a result.json are skipped, so an interrupted run can
be resumed with the same command.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import re
import shutil
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import httpx
from dotenv import load_dotenv

from src.audit import cluster_outcomes, run_probes, write_cluster_csv, write_csv
from src.clustering.hdbscan_clusterer import DEFAULT_MIN_CLUSTER_SIZE
from src.connectors.http import HTTPRAGConnector
from src.connectors.mock_rag_server import create_app
from src.data.kb_loader import load_chunks
from src.data.synthetic_kb_builder import ALL_DOCS, BACKGROUND_DESCRIPTIONS, EVALUATED_TOPICS, Document
from src.probe_generation.kb_blind import TopicSpec
from src.probe_generation.strategies import generate_probe_set
from src.probe_generation.taxonomy import anthropic_completer
from src.scoring.scorer import CoverageScorer

log = logging.getLogger(__name__)

UNITS: Dict[str, str] = {
    "composting": EVALUATED_TOPICS["composting"]["description"],
    "tomato_growing": EVALUATED_TOPICS["tomato_growing"]["description"],
    **BACKGROUND_DESCRIPTIONS,
}
N_ABSENT, N_THIN = 4, 3
MIN_LABELED = 5  # kb_blind probes a cluster needs before its label counts
MIN_PURITY = 0.6


@dataclass
class Injection:
    seed: int
    tiers: Dict[str, str]
    thin_passages: Dict[str, dict]
    docs: List[Document]


def _two_sentences(doc: Document) -> str:
    first_para = doc.body.strip().split("\n\n")[0]
    return " ".join(re.split(r"(?<=[.!?])\s+", first_para)[:2])


def inject(seed: int, docs: List[Document] = ALL_DOCS, n_absent: int = N_ABSENT, n_thin: int = N_THIN) -> Injection:
    if n_absent + n_thin >= len(UNITS):
        raise ValueError(f"need at least one present topic: {n_absent} absent + {n_thin} thin of {len(UNITS)}")
    rng = random.Random(seed)
    units = sorted(UNITS)
    rng.shuffle(units)
    tiers = {u: "absent" for u in units[:n_absent]}
    tiers.update({u: "thin" for u in units[n_absent:n_absent + n_thin]})
    tiers.update({u: "present" for u in units[n_absent + n_thin:]})

    kept = [d for d in docs if tiers.get(d.topic, "present") == "present"]
    hosts = rng.sample([d for d in kept if d.topic in tiers], n_thin)
    passages = {}
    for unit, host in zip([u for u, t in tiers.items() if t == "thin"], hosts):
        source = next(d for d in docs if d.topic == unit)
        passage = _two_sentences(source)
        passages[unit] = {"host": host.doc_id, "source": source.doc_id, "passage": passage}
        # bury it at the end of the host's second paragraph, not as its own chunk
        paras = host.body.strip().split("\n\n")
        paras[min(1, len(paras) - 1)] += " " + passage
        kept[kept.index(host)] = Document(host.doc_id, host.title, host.topic, "\n\n".join(paras))
    return Injection(seed, tiers, passages, kept)


def write_kb(injection: Injection, kb_dir: Path) -> None:
    if kb_dir.exists():
        shutil.rmtree(kb_dir)
    (kb_dir / "docs").mkdir(parents=True)
    for d in injection.docs:
        (kb_dir / "docs" / f"{d.doc_id}.md").write_text(d.to_markdown())


def label_clusters(outcomes, clusters, tiers: Dict[str, str]) -> List[dict]:
    rows = []
    for c in clusters:
        blind = Counter(o.probe.topic for o in outcomes
                        if o.cluster_id == c.cluster_id and o.probe.strategy == "kb_blind")
        label, n = blind.most_common(1)[0] if blind else (None, 0)
        total = sum(blind.values())
        labeled = (not c.is_noise) and total >= MIN_LABELED and n / total >= MIN_PURITY
        rows.append({
            "cluster_id": c.cluster_id, "size": c.query_count, "mean_cs": c.mean_cs, "std_cs": c.std_cs,
            "topic": label if labeled else None, "tier": tiers[label] if labeled else None,
            "purity": n / total if total else 0.0, "kb_blind_probes": total, "strategy_mix": c.strategy_mix,
        })
    return rows


@dataclass
class BenchmarkConfig:
    n_probes: int = 600  # for the three KB-anchored strategies
    per_topic: int = 30  # kb_blind probes per topic
    n_absent: int = N_ABSENT
    n_thin: int = N_THIN
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE


async def run_seed(seed: int, out_dir: Path, config: BenchmarkConfig, complete) -> dict:
    seed_dir = out_dir / f"seed_{seed}"
    injection = inject(seed, n_absent=config.n_absent, n_thin=config.n_thin)
    n_probes, per_topic = config.n_probes, config.per_topic
    write_kb(injection, seed_dir / "kb")
    log.info("seed %d: %s", seed, {t: [u for u, x in injection.tiers.items() if x == t] for t in ("absent", "thin")})

    chunks = load_chunks(seed_dir / "kb")
    topics = [TopicSpec(u, d) for u, d in UNITS.items()]
    probes, taxonomy = await generate_probe_set(
        chunks, n_probes, ["taxonomy", "adversarial", "counterfactual", "kb_blind"], complete,
        kb_blind_topics=topics, kb_blind_per_topic=per_topic)

    transport = httpx.ASGITransport(app=create_app(seed_dir / "kb"))
    async with HTTPRAGConnector("http://mock/query", transport=transport) as conn:
        outcomes = await run_probes(probes, conn, CoverageScorer())
    clusters = await asyncio.to_thread(cluster_outcomes, outcomes, config.min_cluster_size)

    write_csv(outcomes, seed_dir / "probes.csv")
    (seed_dir / "probes.taxonomy.json").write_text(json.dumps([t.as_dict() for t in taxonomy or []], indent=2) + "\n")
    write_cluster_csv(clusters, seed_dir / "probes.clusters.csv")
    result = {
        "seed": seed,
        "config": asdict(config),
        "tiers": injection.tiers,
        "thin_passages": injection.thin_passages,
        "n_docs": len(injection.docs),
        "n_probes": len(outcomes),
        "strategy_counts": dict(Counter(p.strategy for p in probes)),
        "clusters": label_clusters(outcomes, clusters, injection.tiers),
        "topic_means": {
            u: sum(o.score.score for o in outcomes if o.probe.topic == u and o.score)
            / max(1, sum(1 for o in outcomes if o.probe.topic == u and o.score))
            for u in UNITS
        },
    }
    (seed_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def recluster_seed(seed_dir: Path, min_cluster_size: int) -> dict:
    """Redo clustering and labels for a finished seed from its scored probes."""
    from src.reports.load import load_probes

    result = json.loads((seed_dir / "result.json").read_text())
    outcomes = load_probes(seed_dir / "probes.csv", load_chunks(seed_dir / "kb"))
    clusters = cluster_outcomes(outcomes, min_cluster_size)
    write_csv(outcomes, seed_dir / "probes.csv")
    write_cluster_csv(clusters, seed_dir / "probes.clusters.csv")
    result["clusters"] = label_clusters(outcomes, clusters, result["tiers"])
    config = result.get("config") or asdict(BenchmarkConfig(min_cluster_size=20))  # early seeds didn't record one
    config["min_cluster_size"] = min_cluster_size
    result["config"] = config
    (seed_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


async def run_benchmark(seeds: List[int], out_dir: Path, config: BenchmarkConfig, force: bool = False,
                        complete=None, echo=print) -> List[dict]:
    complete = complete or anthropic_completer()
    results = []
    for seed in seeds:
        done = out_dir / f"seed_{seed}" / "result.json"
        if done.exists() and not force:
            r = json.loads(done.read_text())
            if r.get("config") != asdict(config):
                echo(f"seed {seed}: already run with a different config {r.get('config')}; use --force to redo it")
            else:
                echo(f"seed {seed}: already done, skipping")
            results.append(r)
            continue
        r = await run_seed(seed, out_dir, config, complete)
        labeled = [c for c in r["clusters"] if c["tier"]]
        echo(f"seed {seed}: {r['n_probes']} probes, {len(r['clusters'])} clusters, {len(labeled)} labeled "
             f"({dict(Counter(c['tier'] for c in labeled))})")
        results.append(r)
    return results


def main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpx2", "sentence_transformers", "BERTopic", "numba", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--out", type=Path, default=Path("out/benchmark"))
    parser.add_argument("--n-probes", type=int, default=600, help="for the three KB-anchored strategies")
    parser.add_argument("--per-topic", type=int, default=30, help="kb_blind probes per topic")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    asyncio.run(run_benchmark(args.seeds, args.out, BenchmarkConfig(args.n_probes, args.per_topic), args.force,
                              echo=lambda m: print(m, flush=True)))


if __name__ == "__main__":
    main()
