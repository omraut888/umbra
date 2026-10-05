"""Kill/resume stress test for `umbra audit` against a faulty endpoint.

    python scripts/stress/run.py [--kills 0.4,0.75] [--concurrency 50]

Generates a few thousand template probes (no LLM), starts the mock RAG server
behind chaos_server.py (latency, 429 bursts) and drop_proxy.py (connection
resets), SIGKILLs the audit once the checkpoint reaches each --kills fraction,
resumes with --resume, and lets the last run finish. Then it checks the final
CSV against the probe set and the server's per-query log: nothing lost,
nothing duplicated, and how much work the retries and crashes cost.
Everything goes to out/stress/.
"""

import argparse
import csv
import itertools
import json
import os
import random
import shutil
import signal
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.probe_generation.taxonomy import Probe, dedup_probes  # noqa: E402

SUBJECTS = [
    "tomatoes", "peppers", "carrots", "garlic", "onions", "potatoes", "lettuce", "spinach", "kale", "beans", "peas",
    "squash", "cucumbers", "zucchini", "pumpkins", "strawberries", "blueberries", "raspberries", "apple trees",
    "pear trees", "roses", "lavender", "basil", "mint", "rosemary", "thyme", "sunflowers", "tulips", "hostas", "ferns",
    "lawn grass", "clover", "raised beds", "compost piles", "worm bins", "mulch", "drip irrigation", "rain barrels",
    "greenhouses", "cold frames", "seedlings", "houseplants", "succulents", "orchids", "fig trees", "grapevines",
    "asparagus", "rhubarb", "beets", "radishes", "corn", "melons", "cabbage", "broccoli", "cauliflower", "leeks",
    "hops", "bamboo", "hedges",
]
ASPECTS = [
    "watering in a heatwave", "aphid infestations", "frost protection", "soil pH", "nitrogen deficiency",
    "spacing and layout", "pruning timing", "overwintering", "powdery mildew", "companion planting",
    "harvest timing", "growing in containers", "clay soil", "seed saving", "deer damage", "shade tolerance",
]
STEMS = [
    "What should I know about {a} for {s}?", "How do I deal with {a} when growing {s}?",
    "Is {a} a big problem for {s}?", "Best practices for {s}: {a}?",
]


def write_probes(path: Path) -> list:
    qs = [st.format(s=s, a=a) for s, a, st in itertools.product(SUBJECTS, ASPECTS, STEMS)]
    random.Random(0).shuffle(qs)
    probes = [p.query for p in dedup_probes([Probe(query=q, strategy="provided") for q in qs])]
    path.write_text("".join(json.dumps({"query": q}) + "\n" for q in probes))
    return probes


def lines(path: Path) -> int:
    return sum(1 for _ in path.open()) if path.exists() else 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kills", default="0.4,0.75", help="checkpoint fractions to SIGKILL at; '' for none")
    ap.add_argument("--concurrency", type=int, default=50)
    ap.add_argument("--port", type=int, default=8790, help="proxy port; the server gets port+1")
    ap.add_argument("--out", type=Path, default=REPO / "out/stress")
    args = ap.parse_args()
    kills = [float(k) for k in args.kills.split(",") if k]

    out = args.out.resolve()
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    probes = write_probes(out / "probes.jsonl")
    print(f"{len(probes)} probes")
    csv_path, ckpt = out / "audit.csv", out / "audit.checkpoint.jsonl"
    bin_ = Path(sys.executable).parent

    env = {**os.environ, "STRESS_LOG": str(out / "server.log"), "HF_HUB_OFFLINE": "1"}
    server = subprocess.Popen([bin_ / "uvicorn", "scripts.stress.chaos_server:app", "--port", str(args.port + 1),
                               "--log-level", "warning", "--backlog", "2048"], cwd=REPO, env=env)
    proxy = subprocess.Popen([sys.executable, Path(__file__).with_name("drop_proxy.py"), str(args.port),
                              str(args.port + 1), out / "proxy.log"])
    for _ in range(120):
        try:
            httpx.get(f"http://127.0.0.1:{args.port + 1}/health", timeout=1)
            break
        except httpx.HTTPError:
            time.sleep(1)

    base = [str(bin_ / "umbra"), "audit", "--endpoint", f"http://127.0.0.1:{args.port}/query",
            "--kb-path", str(REPO / "data/synthetic_kb"), "--strategies", "none",
            "--probes-file", str(out / "probes.jsonl"), "--no-db", "--no-name-clusters",
            "--output", str(csv_path), "--concurrency", str(args.concurrency)]
    runs = []

    def run(label, extra, kill_at=None):
        t0 = time.time()
        with (out / f"{label}.log").open("w") as log:
            p = subprocess.Popen(base + extra, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
            while p.poll() is None:
                if kill_at and lines(ckpt) >= kill_at:
                    p.send_signal(signal.SIGKILL)
                    p.wait()
                    shutil.copy(ckpt, out / f"{label}.checkpoint_at_kill.jsonl")
                    break
                time.sleep(0.05)
        runs.append({"run": label, "killed": p.returncode == -signal.SIGKILL, "rc": p.returncode,
                     "start": t0, "end": time.time()})
        print(f"{label}: {'killed' if runs[-1]['killed'] else f'exit {p.returncode}'} after "
              f"{time.time() - t0:.1f}s, checkpoint {lines(ckpt)} lines")

    try:
        for i, k in enumerate(kills):
            run(f"run{i + 1}", ["--resume"] if i else [], kill_at=int(len(probes) * k))
        run(f"run{len(kills) + 1}", ["--resume"] if kills else [])
    finally:
        server.terminate()
        proxy.terminate()
    report(out, probes, runs)


def report(out: Path, probes: list, runs: list) -> None:
    rows = list(csv.DictReader((out / "audit.csv").open()))
    qs = [r["query"] for r in rows]
    print(f"\ntotal {runs[-1]['end'] - runs[0]['start']:.0f}s over {len(runs)} runs")
    print(f"final csv: {len(rows)} rows, {len(set(qs))} unique, same probes {set(qs) == set(probes)}, "
          f"same order {qs == probes}, errors {sum(bool(r.get('error')) for r in rows)}, "
          f"unscored {sum(not r['coverage_score'] for r in rows)}")
    left = [p.name for p in (out / "audit.checkpoint.jsonl", out / "audit.resume.json") if p.exists()]
    print(f"resume state left behind: {left or 'none'}")
    for f in sorted(out.glob("*.checkpoint_at_kill.jsonl")):
        bad, seen = 0, []
        for line in f.read_text().splitlines():
            try:
                seen.append(json.loads(line)["question"])
            except ValueError:
                bad += 1
        print(f"{f.name}: {len(seen)} lines, {bad} torn, {len(seen) - len(set(seen))} duplicate")

    def run_of(t):
        return next((r["run"] for r in runs if r["start"] <= t <= r["end"] + 0.5), "?")

    attempts = [json.loads(line) for f in ("server.log", "proxy.log") for line in (out / f).open()]
    by_run = defaultdict(Counter)
    for a in attempts:
        by_run[run_of(a["t"])][a["outcome"]] += 1
    for r in sorted(by_run):
        print(f"{r} requests: {dict(by_run[r])}")
    failed = [a for a in attempts if a["outcome"] != "served"]
    print(f"requests: {len(attempts)}, failed and retried: {len(failed)}")

    probe_set = set(probes)
    served = defaultdict(list)
    for a in attempts:
        if a["outcome"] == "served" and a["q"] in probe_set:
            served[a["q"]].append(run_of(a["t"]))
    twice = Counter(tuple(r) for r in served.values() if len(r) > 1)
    print(f"probes never served: {len(probe_set - set(served))}; served more than once: {sum(twice.values())} "
          f"(by runs: {dict(twice)}; in flight at a kill, redone on resume)")
    per_probe = Counter(a["q"] for a in failed if a["q"])
    unknown = sum(a["q"] is None for a in failed)
    print(f"probes that needed a retry: {len(per_probe)} known, up to {len(per_probe) + unknown} counting "
          f"{unknown} drops not tied to a query; most failed attempts on one probe: {max(per_probe.values(), default=0)}")


if __name__ == "__main__":
    main()
