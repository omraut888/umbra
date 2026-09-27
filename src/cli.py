"""Umbra command-line interface.

    umbra audit --endpoint URL --kb-path PATH --n-probes 1000 --output report.csv
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import click
from dotenv import load_dotenv

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
    for noisy in ("httpx", "sentence_transformers", "BERTopic", "numba", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


@cli.command()
@click.option("--endpoint", required=True, help="RAG query endpoint URL (POST {\"query\": ...}).")
@click.option("--kb-path", required=True, type=click.Path(exists=True, path_type=Path),
              help="Knowledge base directory (.md/.txt files), used for probe generation.")
@click.option("--n-probes", default=1000, show_default=True, type=click.IntRange(min=1))
@click.option("--output", required=True, type=click.Path(dir_okay=False, path_type=Path), help="Per-probe CSV report.")
@click.option("--probes-file", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Use these probes instead of generating them (.jsonl or one query per line).")
@click.option("--weights", default="0.4,0.35,0.25", show_default=True, help="alpha,beta,gamma for RC, 1-SE, 1-HP.")
@click.option("--hp-band", default=",".join(map(str, DEFAULT_HP_BAND)), show_default=True,
              help="Compute HP only when the preliminary score is in this range.")
@click.option("--se-method", type=click.Choice(SE_METHODS), default=DEFAULT_SE_METHOD, show_default=True,
              help="dispersion = mean pairwise cosine distance; spec = the original spec §4 histogram entropy.")
@click.option("--payload", default="{}", help="Extra JSON merged into each request body, e.g. '{\"top_k\": 5}'.")
@click.option("--auth-header", envvar="RAG_AUTH_HEADER", help="Authorization header value for the endpoint.")
@click.option("--concurrency", default=50, show_default=True, help="Concurrent RAG queries.")
@click.option("--model", default=CLAUDE_MODEL, show_default=True, help="Claude model for probe generation.")
@click.option("--db-dsn", envvar="POSTGRES_DSN", help="Postgres DSN; results are stored when set. [env: POSTGRES_DSN]")
@click.option("--no-db", is_flag=True, help="Skip Postgres even if POSTGRES_DSN is set.")
def audit(endpoint, kb_path, n_probes, output, probes_file, weights, hp_band, se_method, payload,
          auth_header, concurrency, model, db_dsn, no_db) -> None:
    """Run probes against a RAG system and write per-probe coverage scores."""
    from src.audit import overall_score, persist, run_probes, write_csv
    from src.connectors.http import HTTPRAGConnector
    from src.data.kb_loader import kb_fingerprint, load_chunks, load_documents
    from src.probe_generation.taxonomy import anthropic_completer, extract_taxonomy, taxonomy_guided_generation
    from src.scoring.scorer import CoverageScorer, ScorerConfig

    try:
        config = ScorerConfig(weights=ScoringWeights.parse(weights), hp_band=_parse_band(hp_band), se_method=se_method)
        extra_payload = json.loads(payload)
    except (ValueError, json.JSONDecodeError) as exc:
        raise click.BadParameter(str(exc)) from exc

    docs = load_documents(kb_path)
    fingerprint = kb_fingerprint(docs)
    click.echo(f"KB: {len(docs)} documents, fingerprint {fingerprint[:12]}")

    extra_config = {"n_probes_requested": n_probes}
    if probes_file:
        probes = load_probes_file(probes_file)
        extra_config["probes_file"] = str(probes_file)
        click.echo(f"Loaded {len(probes)} probes from {probes_file}")
    else:
        chunks = load_chunks(kb_path)
        click.echo(f"Extracting topic taxonomy from {len(chunks)} chunks (BERTopic)...")
        topics = extract_taxonomy(chunks)
        for t in topics:
            click.echo(f"  topic {t.topic_id:>2} ({t.size:>3} chunks): {t.label}")
        taxonomy_path = output.with_suffix(".taxonomy.json")
        taxonomy_path.parent.mkdir(parents=True, exist_ok=True)
        taxonomy_path.write_text(json.dumps([t.as_dict() for t in topics], indent=2) + "\n")
        click.echo(f"Generating {n_probes} probes with {model} (taxonomy -> {taxonomy_path})...")
        try:
            complete = anthropic_completer(model=model)
            probes = asyncio.run(taxonomy_guided_generation(topics, n_probes, complete))
        except Exception as exc:
            if type(exc).__name__ in ("AuthenticationError", "PermissionDeniedError") or "api_key" in str(exc).lower():
                raise click.ClickException(
                    f"Claude probe generation failed: {exc}\n"
                    "Set ANTHROPIC_API_KEY in .env, or pass --probes-file to skip generation."
                ) from exc
            raise
        extra_config.update(model=model, n_topics=len(topics))
        click.echo(f"Generated {len(probes)} probes after deduplication")

    async def _run():
        async with HTTPRAGConnector(endpoint, auth_header=auth_header, extra_payload=extra_payload) as conn:
            return await run_probes(probes, conn, CoverageScorer(config), concurrency=concurrency)

    click.echo(f"Querying {endpoint} and scoring {len(probes)} probes...")
    outcomes = asyncio.run(_run())
    write_csv(outcomes, output)

    scored = [o for o in outcomes if o.score]
    n_hp = sum(o.score.hp_computed for o in scored)
    overall = overall_score(outcomes)
    click.echo(f"Wrote {len(outcomes)} rows to {output}")
    click.echo(f"Scored {len(scored)}/{len(outcomes)} probes; HP computed for {n_hp} (preliminary in {config.hp_band})")
    if overall is not None:
        click.echo(f"Overall coverage score: {overall:.3f}")
        _echo_topic_summary(scored)

    if db_dsn and not no_db:
        report_id = persist(db_dsn, outcomes, kb_fingerprint=fingerprint, endpoint_url=endpoint,
                            kb_path=str(kb_path), config=config, extra_config=extra_config)
        click.echo(f"Stored audit run {report_id} in Postgres")
    else:
        click.echo("Postgres storage skipped (set POSTGRES_DSN or --db-dsn to enable)")


def _echo_topic_summary(scored) -> None:
    groups = defaultdict(list)
    for o in scored:
        groups[o.probe.topic or "(none)"].append(o.score.score)
    if len(groups) <= 1:
        return
    click.echo("Mean coverage by probe topic:")
    for topic, vals in sorted(groups.items(), key=lambda kv: sum(kv[1]) / len(kv[1])):
        click.echo(f"  {sum(vals) / len(vals):.3f}  n={len(vals):<4} {topic}")


if __name__ == "__main__":
    cli()
