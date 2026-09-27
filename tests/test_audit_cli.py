"""End-to-end: CLI -> HTTPRAGConnector -> mock RAG server -> scorer -> CSV (+ Postgres)."""

import csv
import json
import os
import socket
import threading
import time

import pytest
import uvicorn
from click.testing import CliRunner

from src.cli import cli
from src.connectors.mock_rag_server import create_app
from src.data import synthetic_kb_builder


@pytest.fixture(scope="module")
def kb_path(tmp_path_factory):
    out = tmp_path_factory.mktemp("kb")
    synthetic_kb_builder.build(out)
    return out


@pytest.fixture(scope="module")
def server_url(kb_path):
    """Run the mock RAG server on a real port so the CLI talks HTTP to it."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(create_app(kb_path), host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 60
    while not server.started:
        if time.time() > deadline:
            raise RuntimeError("mock server did not start")
        time.sleep(0.1)
    yield f"http://127.0.0.1:{port}/query"
    server.should_exit = True
    thread.join(timeout=10)


@pytest.fixture
def probes_file(tmp_path):
    path = tmp_path / "probes.jsonl"
    rows = [
        {"query": "How often should I turn a compost pile?", "topic": "composting"},
        {"query": "When should I remove suckers from indeterminate tomatoes?", "topic": "tomato_growing"},
        {"query": "How is staking income from Ethereum taxed?", "topic": "cryptocurrency_taxation"},
        {"query": "What is a Hohmann transfer orbit?", "topic": "orbital_mechanics"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


def run_audit(server_url, kb_path, probes_file, output, *extra):
    result = CliRunner().invoke(cli, [
        "audit", "--endpoint", server_url, "--kb-path", str(kb_path), "--probes-file", str(probes_file),
        "--output", str(output), *extra,
    ], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    return result


def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_audit_writes_scored_csv(server_url, kb_path, probes_file, tmp_path):
    out = tmp_path / "report.csv"
    result = run_audit(server_url, kb_path, probes_file, out, "--no-db")
    assert "Overall coverage score" in result.output

    rows = read_rows(out)
    assert len(rows) == 4
    by_topic = {r["probe_topic"]: r for r in rows}
    for r in rows:
        assert r["error"] == "" and int(r["n_chunks"]) == 5
        assert 0.0 <= float(r["coverage_score"]) <= 1.0
        assert len(json.loads(r["retrieved_chunk_ids"])) == 5
        assert r["hp_computed"] in ("True", "False")
    for covered in ("composting", "tomato_growing"):
        for absent in ("cryptocurrency_taxation", "orbital_mechanics"):
            assert float(by_topic[covered]["coverage_score"]) > float(by_topic[absent]["coverage_score"])


def test_audit_reports_failed_queries_without_aborting(kb_path, probes_file, tmp_path):
    out = tmp_path / "report.csv"
    run_audit("http://127.0.0.1:9/query", kb_path, probes_file, out, "--no-db")
    rows = read_rows(out)
    assert len(rows) == 4 and all(r["error"] and r["coverage_score"] == "" for r in rows)


def test_invalid_weights_rejected(server_url, kb_path, probes_file, tmp_path):
    result = CliRunner().invoke(cli, [
        "audit", "--endpoint", server_url, "--kb-path", str(kb_path), "--probes-file", str(probes_file),
        "--output", str(tmp_path / "r.csv"), "--weights", "0.5,0.5,0.5", "--no-db",
    ])
    assert result.exit_code != 0 and "sum to 1.0" in result.output


@pytest.mark.postgres
@pytest.mark.skipif(not os.environ.get("POSTGRES_DSN"), reason="POSTGRES_DSN not set")
def test_audit_persists_to_postgres(server_url, kb_path, probes_file, tmp_path):
    from sqlalchemy import create_engine, text

    out = tmp_path / "report.csv"
    result = run_audit(server_url, kb_path, probes_file, out, "--se-method", "dispersion")
    report_id = result.output.split("Stored audit run ")[1].split()[0]

    engine = create_engine(os.environ["POSTGRES_DSN"])
    try:
        with engine.connect() as conn:
            run = conn.execute(text("SELECT probe_count, config, overall_score FROM audit_runs WHERE report_id = :r"),
                               {"r": report_id}).one()
            probes = conn.execute(text(
                "SELECT query_text, coverage_score, vector_dims(query_embedding), retrieved_chunk_ids "
                "FROM probe_results WHERE report_id = :r"), {"r": report_id}).all()
            conn.execute(text("DELETE FROM audit_runs WHERE report_id = :r"), {"r": report_id})
            conn.commit()
    finally:
        engine.dispose()

    assert run.probe_count == 4 and run.config["se_method"] == "dispersion" and run.config["alpha"] == 0.4
    csv_scores = {r["query"]: float(r["coverage_score"]) for r in read_rows(out)}
    assert len(probes) == 4
    for p in probes:
        assert p[2] == 384 and len(p[3]) == 5
        assert p[1] == pytest.approx(csv_scores[p[0]], abs=1e-4)
