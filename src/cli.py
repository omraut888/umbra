"""Umbra command-line interface.

    umbra audit --endpoint URL --kb-path PATH --n-probes 1000 --output report.csv
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Optional, Tuple

import click
from dotenv import load_dotenv

from src.clustering.hdbscan_clusterer import DEFAULT_MIN_CLUSTER_SIZE
from src.clustering.zones import DEFAULT_THRESHOLDS, ZoneThresholds
from src.probe_generation.strategies import ALL_STRATEGIES
from src.probe_generation.taxonomy import CLAUDE_MODEL, Probe
from src.scoring.composite import DEFAULT_HP_BAND, ScoringWeights
from src.scoring.signals import DEFAULT_SE_METHOD, SE_METHODS


def _parse_band(value: str) -> Tuple[float, float]:
    lo, hi = (float(x) for x in value.split(","))
    if not 0.0 <= lo <= hi <= 1.0:
        raise click.BadParameter(f"expected 0 <= lo <= hi <= 1, got {value!r}")
    return lo, hi


def load_probes_file(path: Path) -> List[Probe]:
    """.jsonl: {"query", "topic"?, "strategy"?} per line. Anything else: one query per line."""
    probes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        if path.suffix == ".jsonl":
            obj = json.loads(line)
            probes.append(Probe(query=obj["query"], topic=obj.get("topic"), strategy=obj.get("strategy", "provided")))
        else:
            probes.append(Probe(query=line.strip(), strategy="provided"))
    return probes


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Debug logging.")
def cli(verbose: bool) -> None:
    """Umbra: map the blind spots of a RAG system."""
    load_dotenv()
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    for noisy in ("httpx", "httpx2", "sentence_transformers", "BERTopic", "numba", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


@cli.command()
@click.option("--endpoint", required=True, help="RAG query endpoint URL (POST {\"query\": ...}).")
@click.option("--kb-path", required=True, type=click.Path(exists=True, path_type=Path),
              help="Knowledge base directory (.md/.txt files), used for probe generation.")
@click.option("--n-probes", default=1000, show_default=True, type=click.IntRange(min=0),
              help="Probes to generate across the chosen strategies (before dedup).")
@click.option("--output", required=True, type=click.Path(dir_okay=False, path_type=Path), help="Per-probe CSV report.")
@click.option("--strategies", default="taxonomy,adversarial,counterfactual", show_default=True,
              help=f"Comma-separated, from: {', '.join(ALL_STRATEGIES)}; or 'none' to only use --probes-file. "
                   "kb_blind also needs --topics-file or --domain.")
@click.option("--topics-file", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Topics the KB should cover, for kb_blind (JSON: [{name, description}] or {name: description}).")
@click.option("--domain", help="One-line description of what the KB should cover; kb_blind expands it into topics.")
@click.option("--kb-blind-per-topic", default=30, show_default=True, help="kb_blind probes per topic.")
@click.option("--probes-file", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Extra probes added to the generated ones (.jsonl or one query per line).")
@click.option("--weights", default="0.4,0.35,0.25", show_default=True, help="alpha,beta,gamma for RC, 1-SE, 1-HP.")
@click.option("--hp-band", default=",".join(map(str, DEFAULT_HP_BAND)), show_default=True,
              help="Compute HP only when the preliminary score is in this range.")
@click.option("--se-method", type=click.Choice(SE_METHODS), default=DEFAULT_SE_METHOD, show_default=True,
              help="dispersion = mean pairwise cosine distance; spec = the original spec §4 histogram entropy.")
@click.option("--cluster/--no-cluster", default=True, show_default=True, help="UMAP + HDBSCAN + zone summary.")
@click.option("--min-cluster-size", default=DEFAULT_MIN_CLUSTER_SIZE, show_default=True)
@click.option("--zone-thresholds", default=f"{DEFAULT_THRESHOLDS.dark_below},{DEFAULT_THRESHOLDS.adequate_above}",
              show_default=True, help="dark_below,adequate_above for cluster zones.")
@click.option("--zone-thresholds-file", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Thresholds written by `umbra benchmark calibrate --write-thresholds`.")
@click.option("--payload", default="{}", help="Extra JSON merged into each request body, e.g. '{\"top_k\": 5}'.")
@click.option("--auth-header", envvar="RAG_AUTH_HEADER", help="Authorization header value for the endpoint.")
@click.option("--concurrency", default=50, show_default=True, help="Concurrent RAG queries.")
@click.option("--model", default=CLAUDE_MODEL, show_default=True, help="Claude model for generation and naming.")
@click.option("--db-dsn", envvar="POSTGRES_DSN", help="Postgres DSN; results are stored when set. [env: POSTGRES_DSN]")
@click.option("--no-db", is_flag=True, help="Skip Postgres even if POSTGRES_DSN is set.")
def audit(endpoint, kb_path, n_probes, output, strategies, topics_file, domain, kb_blind_per_topic, probes_file,
          weights, hp_band, se_method, cluster, min_cluster_size, zone_thresholds, zone_thresholds_file, payload,
          auth_header, concurrency, model, db_dsn, no_db) -> None:
    """Probe a RAG system, score coverage per probe, and cluster the results into zones."""
    from src.audit import (
        cluster_outcomes, overall_score, persist, run_probes, write_cluster_csv, write_csv, write_responses,
    )
    from src.clustering.naming import name_clusters
    from src.connectors.http import HTTPRAGConnector
    from src.data.kb_loader import kb_fingerprint, load_chunks, load_documents
    from src.probe_generation.kb_blind import enumerate_domain_topics, load_topics_file
    from src.probe_generation.strategies import generate_probe_set
    from src.probe_generation.taxonomy import anthropic_completer
    from src.scoring.scorer import CoverageScorer, ScorerConfig

    try:
        config = ScorerConfig(weights=ScoringWeights.parse(weights), hp_band=_parse_band(hp_band), se_method=se_method)
        extra_payload = json.loads(payload)
        thresholds, _ = _thresholds(zone_thresholds, zone_thresholds_file)
    except (ValueError, json.JSONDecodeError, KeyError) as exc:
        raise click.BadParameter(str(exc)) from exc
    chosen = [] if strategies.strip() == "none" else [s.strip() for s in strategies.split(",") if s.strip()]
    if not chosen and not probes_file:
        raise click.UsageError("nothing to run: pick --strategies or pass --probes-file")
    if set(chosen) - {"kb_blind"} and n_probes == 0:
        raise click.UsageError("--n-probes must be > 0 when generating probes")
    if "kb_blind" in chosen and not (topics_file or domain):
        raise click.UsageError("kb_blind needs --topics-file or --domain")

    docs = load_documents(kb_path)
    fingerprint = kb_fingerprint(docs)
    click.echo(f"KB: {len(docs)} documents, fingerprint {fingerprint[:12]}")

    extra_config = {"strategies": chosen, "n_probes_requested": n_probes if chosen else 0,
                    "zone_thresholds": [thresholds.dark_below, thresholds.adequate_above]}
    extra_probes = []
    if probes_file:
        extra_probes = load_probes_file(probes_file)
        extra_config["probes_file"] = str(probes_file)
        click.echo(f"Loaded {len(extra_probes)} probes from {probes_file}")

    complete = anthropic_completer(model=model) if (chosen or cluster) else None
    blind_topics = []
    if "kb_blind" in chosen:
        blind_topics = load_topics_file(topics_file) if topics_file else asyncio.run(enumerate_domain_topics(domain, complete))
        click.echo(f"kb_blind: {len(blind_topics)} topics x {kb_blind_per_topic} probes"
                   + ("" if topics_file else f" (expanded from --domain: {', '.join(t.name for t in blind_topics)})"))
        extra_config.update(kb_blind_topics=[t.name for t in blind_topics], kb_blind_per_topic=kb_blind_per_topic)
    if chosen:
        chunks = load_chunks(kb_path)
        click.echo(f"Generating probes with {model}: {', '.join(chosen)}")
        probes, topics = asyncio.run(generate_probe_set(chunks, n_probes, chosen, complete, extra_probes,
                                                        blind_topics, kb_blind_per_topic))
        if topics:
            for t in topics:
                click.echo(f"  taxonomy topic {t.topic_id:>2} ({t.size:>3} chunks): {t.label}")
            taxonomy_path = output.with_suffix(".taxonomy.json")
            taxonomy_path.parent.mkdir(parents=True, exist_ok=True)
            taxonomy_path.write_text(json.dumps([t.as_dict() for t in topics], indent=2) + "\n")
            extra_config["n_topics"] = len(topics)
        extra_config["model"] = model
    else:
        probes = extra_probes
    click.echo(f"{len(probes)} probes after dedup: {dict(Counter(p.strategy for p in probes))}")

    async def _run():
        async with HTTPRAGConnector(endpoint, auth_header=auth_header, extra_payload=extra_payload) as conn:
            return await run_probes(probes, conn, CoverageScorer(config), concurrency=concurrency)

    click.echo(f"Querying {endpoint} and scoring...")
    outcomes = asyncio.run(_run())
    scored = [o for o in outcomes if o.score]
    n_hp = sum(o.score.hp_computed for o in scored)
    click.echo(f"Scored {len(scored)}/{len(outcomes)} probes; HP computed for {n_hp} (preliminary in {config.hp_band})")

    clusters = []
    if cluster and len(scored) < 2 * min_cluster_size:
        click.echo(f"Skipping clustering: {len(scored)} scored probes is too few for min_cluster_size={min_cluster_size}")
    elif cluster:
        click.echo("Clustering (UMAP 10D -> HDBSCAN, UMAP 2D for display)...")
        clusters = cluster_outcomes(outcomes, min_cluster_size=min_cluster_size, thresholds=thresholds)
        asyncio.run(name_clusters(clusters, complete))
        cluster_path = output.with_suffix(".clusters.csv")
        write_cluster_csv(clusters, cluster_path)
        _echo_cluster_table(clusters)
        click.echo(f"Wrote cluster summary to {cluster_path}")

    write_csv(outcomes, output)
    write_responses(outcomes, output.with_suffix(".responses.jsonl"))
    click.echo(f"Wrote {len(outcomes)} rows to {output}")
    overall = overall_score(outcomes)
    if overall is not None:
        click.echo(f"Overall coverage score: {overall:.3f}")
        _echo_group_means("strategy", [(o.probe.strategy, o.score.score) for o in scored])

    if db_dsn and not no_db:
        report_id = persist(db_dsn, outcomes, kb_fingerprint=fingerprint, endpoint_url=endpoint,
                            kb_path=str(kb_path), config=config, extra_config=extra_config, clusters=clusters)
        click.echo(f"Stored audit run {report_id} in Postgres")
    else:
        click.echo("Postgres storage skipped (set POSTGRES_DSN or --db-dsn to enable)")


@cli.command()
@click.option("--audit", "audit_csv", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Probe CSV written by `umbra audit` (its .clusters.csv must sit next to it).")
@click.option("--kb-path", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--output", required=True, type=click.Path(dir_okay=False, path_type=Path), help="GapReport JSON.")
@click.option("--web-search", type=click.Choice(["none", "anthropic", "brave"]), default="anthropic", show_default=True,
              help="Backend for external recommendations; 'none' gives KB-internal ones only.")
@click.option("--top-k", default=10, show_default=True,
              help="Besides every DARK and THIN cluster, recommend for the k most severe clusters of any tier.")
@click.option("--zone-thresholds", default=f"{DEFAULT_THRESHOLDS.dark_below},{DEFAULT_THRESHOLDS.adequate_above}",
              show_default=True)
@click.option("--zone-thresholds-file", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Thresholds written by `umbra benchmark calibrate --write-thresholds` (overrides --zone-thresholds).")
@click.option("--dashboard-uri", help="Where the coverage map will be served, stored as coverage_map_uri.")
@click.option("--model", default=CLAUDE_MODEL, show_default=True)
def report(audit_csv, kb_path, output, web_search, top_k, zone_thresholds, zone_thresholds_file, dashboard_uri,
           model) -> None:
    """Build a GapReport (zones, severity ranking, recommendations) from an audit."""
    from src.data.kb_loader import kb_fingerprint, load_chunks, load_documents
    from src.probe_generation.taxonomy import anthropic_completer
    from src.reports.builder import build_report
    from src.reports.load import load_audit
    from src.reports.search import make_provider

    thresholds, source = _thresholds(zone_thresholds, zone_thresholds_file)
    chunks = load_chunks(kb_path)
    loaded = load_audit(audit_csv, chunks)
    complete = search = None
    if web_search != "none":
        complete, search = anthropic_completer(model=model), make_provider(web_search)
    gap_report = asyncio.run(build_report(
        loaded.outcomes, loaded.clusters, chunks, kb_fingerprint=kb_fingerprint(load_documents(kb_path)),
        thresholds=thresholds, thresholds_source=source, complete=complete, search=search, top_k=top_k,
        coverage_map_uri=dashboard_uri, config={"audit": str(audit_csv), "web_search": web_search, "top_k": top_k},
    ))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(gap_report.model_dump_json(indent=2) + "\n")

    click.echo(f"{gap_report.cluster_count} clusters: {len(gap_report.dark_zones)} dark, {len(gap_report.thin_zones)} thin")
    click.echo(f"{'rank':>4}  {'name':<40} {'zone':<9} {'severity':>8} {'unanswered':>10}  recs")
    for c in gap_report.clusters[:15]:
        click.echo(f"{c.severity_rank:>4}  {c.name[:40]:<40} {c.zone:<9} {c.severity:>8.2f} {c.unanswered_share:>10.0%}  "
                   f"{len(c.recommendations)}")
    d = gap_report.estimated_improvement_detail
    click.echo(f"Overall {d.overall_before:.3f} -> {d.overall_after:.3f} after top {d.recommendations_applied} "
               f"recommendations (simulated, optimistic)")
    click.echo(f"Wrote {output}")


@cli.command()
@click.option("--report", "report_path", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--audit", "audit_csv", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Probe CSV for the map points (default: the audit recorded in the report).")
@click.option("--port", default=8050, show_default=True)
def dashboard(report_path, audit_csv, port) -> None:
    """Serve the coverage map and cluster ranking for a GapReport."""
    from src.dashboard.app import main

    main(str(report_path), str(audit_csv) if audit_csv else None, port=port)


@cli.group()
def benchmark() -> None:
    """Gap-injection benchmark: known gaps in, calibrated thresholds out."""


@benchmark.command("run")
@click.option("--seeds", default="0,1,2,3,4", show_default=True, help="Comma-separated seeds.")
@click.option("--out", "out_dir", default="out/benchmark", show_default=True, type=click.Path(path_type=Path))
@click.option("--n-probes", default=600, show_default=True, help="Probes for the three KB-anchored strategies.")
@click.option("--per-topic", default=30, show_default=True, help="kb_blind probes per topic.")
@click.option("--n-absent", default=4, show_default=True, help="Topics removed entirely per seed.")
@click.option("--n-thin", default=3, show_default=True, help="Topics cut down to a buried passage per seed.")
@click.option("--min-cluster-size", default=DEFAULT_MIN_CLUSTER_SIZE, show_default=True)
@click.option("--force", is_flag=True, help="Redo seeds that already have results.")
def benchmark_run(seeds, out_dir, n_probes, per_topic, n_absent, n_thin, min_cluster_size, force) -> None:
    """Run seeds (skipping finished ones unless --force)."""
    from src.benchmark.gap_injection import BenchmarkConfig, run_benchmark

    config = BenchmarkConfig(n_probes=n_probes, per_topic=per_topic, n_absent=n_absent, n_thin=n_thin,
                             min_cluster_size=min_cluster_size)
    asyncio.run(run_benchmark([int(x) for x in seeds.split(",")], out_dir, config, force, echo=click.echo))


@benchmark.command("recluster")
@click.option("--out", "out_dir", default="out/benchmark", show_default=True,
              type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--min-cluster-size", default=DEFAULT_MIN_CLUSTER_SIZE, show_default=True)
def benchmark_recluster(out_dir, min_cluster_size) -> None:
    """Re-cluster finished seeds from their scored probes (no LLM calls), e.g. after changing min_cluster_size."""
    from src.benchmark.gap_injection import recluster_seed

    for result_path in sorted(out_dir.glob("seed_*/result.json")):
        r = recluster_seed(result_path.parent, min_cluster_size)
        labeled = [c for c in r["clusters"] if c["tier"]]
        click.echo(f"seed {r['seed']}: {len(r['clusters'])} clusters, {len(labeled)} labeled at "
                   f"min_cluster_size={min_cluster_size}")


@benchmark.command("recall")
@click.option("--out", "out_dir", default="out/benchmark", show_default=True,
              type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--refit", is_flag=True, help="Held-out comparison with thresholds re-fit per min_cluster_size.")
def benchmark_recall(out_dir, refit) -> None:
    """Recall sensitivity to min_cluster_size and kb_blind probe count (no LLM calls)."""
    import sys

    from src.benchmark import recall

    sys.argv = ["recall", str(out_dir)] + (["--refit"] if refit else [])
    recall.main()


@benchmark.command("calibrate")
@click.option("--out", "out_dir", default="out/benchmark", show_default=True,
              type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--write-thresholds", type=click.Path(dir_okay=False, path_type=Path),
              help="Write the fitted thresholds as JSON, for --zone-thresholds-file.")
def benchmark_calibrate(out_dir, write_thresholds) -> None:
    """Fit zone thresholds on every finished seed and compare with the spec's."""
    from src.benchmark.calibrate import calibrate, print_summary

    print_summary(calibrate(out_dir, write_thresholds))
    if write_thresholds:
        click.echo(f"\nWrote {write_thresholds}")


@benchmark.command("ablate")
@click.option("--out", "out_dir", default="out/benchmark", show_default=True,
              type=click.Path(exists=True, file_okay=False, path_type=Path))
def benchmark_ablate(out_dir) -> None:
    """Re-cluster finished seeds with strategy subsets (no new LLM calls)."""
    import sys

    from src.benchmark import ablation

    sys.argv = ["ablation", str(out_dir)]
    ablation.main()


def _thresholds(spec: str, path: Optional[Path]) -> Tuple[ZoneThresholds, str]:
    if path:
        data = json.loads(path.read_text())
        return ZoneThresholds(data["dark_below"], data["adequate_above"]), f"{path} ({data.get('fit_on', 'file')})"
    th = ZoneThresholds.parse(spec)
    return th, "calibrated default" if th == DEFAULT_THRESHOLDS else "--zone-thresholds"


def _echo_group_means(label: str, pairs) -> None:
    groups = defaultdict(list)
    for key, score in pairs:
        groups[key].append(score)
    click.echo(f"Mean coverage by {label}:")
    for key, vals in sorted(groups.items(), key=lambda kv: sum(kv[1]) / len(kv[1])):
        click.echo(f"  {sum(vals) / len(vals):.3f}  n={len(vals):<5} {key}")


def _echo_cluster_table(clusters) -> None:
    click.echo(f"  {'id':>4}  {'name':<40} {'size':>5} {'mean_cs':>8} {'zone':<9} {'severity':>8}")
    for c in clusters:
        click.echo(f"  {c.cluster_id:>4}  {(c.name or '')[:40]:<40} {c.query_count:>5} {c.mean_cs:>8.3f} "
                   f"{c.zone:<9} {c.severity:>8.3f}")


if __name__ == "__main__":
    cli()
