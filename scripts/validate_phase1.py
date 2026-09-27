"""Phase 1 validation against the synthetic KB's ground truth.

    # 1. Labeled probes: N questions per ground-truth topic, written by Claude
    #    from the topic *description* only (it never sees the KB).
    python scripts/validate_phase1.py probes --n-per-topic 40

    # 2. Run audits with the CLI (see README), then:
    python scripts/validate_phase1.py report \
        --labeled dispersion=out/labeled_dispersion.csv --labeled spec=out/labeled_spec.csv \
        --taxonomy out/audit_dispersion.csv

The taxonomy audit's probes are generated from BERTopic topics, not from
ground-truth topics, so each BERTopic topic is mapped to the ground-truth
label held by the majority of its chunks' documents (via <csv>.taxonomy.json).
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv  # noqa: E402

KB = Path("data/synthetic_kb")
EXPECTED_ZONE = {"full": "ADEQUATE", "thin": "THIN", "absent": "DARK"}
TIER_ORDER = {"full": 0, "thin": 1, "absent": 2}

PROBE_PROMPT = """Write {k} different questions that a user might ask a knowledge base about this topic:

{description}

Mix simple questions with detailed, specific ones, and vary the type (factual lookup, comparison, causal, procedural, definitional).
{avoid}
Return only the questions, one per line. No numbering, no explanation."""


def zone(score: float) -> str:
    """Spec §6 thresholds."""
    return "DARK" if score < 0.30 else "THIN" if score <= 0.60 else "ADEQUATE"


async def _generate(topic: str, description: str, n: int, complete) -> List[str]:
    from src.probe_generation.taxonomy import parse_questions

    questions: Dict[str, None] = {}
    for _ in range(n):  # hard cap on calls
        if len(questions) >= n:
            break
        avoid = ""
        if questions:
            avoid = "\nDo not repeat any of these:\n" + "\n".join(f"- {q}" for q in list(questions)[-30:]) + "\n"
        text = await complete(PROBE_PROMPT.format(k=min(10, n - len(questions)), description=description, avoid=avoid))
        for q in parse_questions(text):
            questions.setdefault(q, None)
    return list(questions)[:n]


def cmd_probes(args) -> None:
    from src.probe_generation.taxonomy import Probe, anthropic_completer, dedup_probes

    gt = json.loads((args.kb / "ground_truth.json").read_text())
    complete = anthropic_completer()

    async def run():
        return await asyncio.gather(*(
            _generate(name, t["description"], args.n_per_topic, complete) for name, t in gt["topics"].items()
        ))

    results = asyncio.run(run())
    probes = [Probe(query=q, topic=name, strategy="ground_truth")
              for name, qs in zip(gt["topics"], results) for q in qs]
    probes = dedup_probes(probes)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for p in probes:
            f.write(json.dumps({"query": p.query, "topic": p.topic, "strategy": p.strategy}) + "\n")
    counts = Counter(p.topic for p in probes)
    print(f"Wrote {len(probes)} labeled probes to {args.out}: {dict(counts)}")


def _rows(path: Path) -> List[dict]:
    with path.open(newline="") as f:
        return [r for r in csv.DictReader(f) if r["coverage_score"]]


def _stats(rows: List[dict]) -> dict:
    cs = [float(r["coverage_score"]) for r in rows]
    hp_rows = [r for r in rows if r["hp_computed"] == "True"]
    return {
        "n": len(rows),
        "mean": statistics.fmean(cs),
        "median": statistics.median(cs),
        "rc": statistics.fmean(float(r["rc_score"]) for r in rows),
        "se": statistics.fmean(float(r["se_score"]) for r in rows),
        "hp_pct": 100 * len(hp_rows) / len(rows),
        "zones": Counter(zone(x) for x in cs),
    }


def _print_table(title: str, groups: Dict[str, List[dict]], tiers: Dict[str, str]) -> int:
    print(f"\n{title}")
    header = (f"{'topic':<26}{'true tier':<10}{'n':>5}{'mean CS':>9}{'median':>8}{'RC':>7}{'SE':>7}"
              f"{'HP%':>6}  {'zone(mean)':<11}{'expected':<10}{'match':<6}  per-probe zones D/T/A")
    print(header)
    print("-" * len(header))
    mismatches = 0
    for topic in sorted(groups, key=lambda t: (TIER_ORDER.get(tiers.get(t, ""), 3), t)):
        s = _stats(groups[topic])
        tier = tiers.get(topic, "background")
        expected = EXPECTED_ZONE.get(tier, "-")
        z = zone(s["mean"])
        match = "-" if expected == "-" else ("yes" if z == expected else "NO")
        mismatches += match == "NO"
        zc = s["zones"]
        print(f"{topic:<26}{tier:<10}{s['n']:>5}{s['mean']:>9.3f}{s['median']:>8.3f}{s['rc']:>7.3f}{s['se']:>7.3f}"
              f"{s['hp_pct']:>5.0f}%  {z:<11}{expected:<10}{match:<6}  {zc['DARK']}/{zc['THIN']}/{zc['ADEQUATE']}")
    return mismatches


def cmd_report(args) -> None:
    gt = json.loads((args.kb / "ground_truth.json").read_text())
    tiers = {name: t["tier"] for name, t in gt["topics"].items()}
    doc_topic = gt["document_topics"]
    total_mismatches = 0

    for spec in args.labeled or []:
        method, path = spec.split("=", 1)
        groups = defaultdict(list)
        for r in _rows(Path(path)):
            groups[r["probe_topic"]].append(r)
        total_mismatches += _print_table(
            f"Labeled probes (per ground-truth topic), SE method = {method}   [{path}]", groups, tiers)

    for path in args.taxonomy or []:
        path = Path(path)
        taxonomy = json.loads(path.with_suffix(".taxonomy.json").read_text())
        mapping = {}
        print(f"\nTaxonomy audit: BERTopic topic -> majority ground-truth label   [{path}]")
        for t in taxonomy:
            labels = Counter(doc_topic[c.split("#")[0]] for c in t["chunk_ids"])
            label, count = labels.most_common(1)[0]
            mapping[t["name"]] = label
            print(f"  {t['name'][:50]:<52} -> {label:<18} ({count}/{t['size']} chunks)  {dict(labels)}")
        groups = defaultdict(list)
        for r in _rows(path):
            groups[mapping.get(r["probe_topic"], "?")].append(r)
        _print_table(f"Taxonomy audit, grouped by mapped ground-truth label   [{path}]", groups, tiers)

    print(f"\nTier mismatches across labeled tables: {total_mismatches}")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kb", type=Path, default=KB)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probes", help="generate labeled probes per ground-truth topic")
    p.add_argument("--n-per-topic", type=int, default=40)
    p.add_argument("--out", type=Path, default=KB / "validation_probes.jsonl")
    p.set_defaults(func=cmd_probes)
    r = sub.add_parser("report", help="per-topic tables vs ground truth")
    r.add_argument("--labeled", action="append", metavar="METHOD=CSV")
    r.add_argument("--taxonomy", action="append", metavar="CSV")
    r.set_defaults(func=cmd_report)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
