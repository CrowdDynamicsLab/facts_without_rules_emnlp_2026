#!/usr/bin/env python3
"""Summarize operational-lifting leakage counts."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


DEFAULT_EVAL_FILES = [
    "data/eval_openai_gpt5mini_operational_lifting_calibrated.jsonl",
    "data/eval_uiuc_deepseek_operational_lifting_calibrated.jsonl",
]


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_csv(path, rows, fieldnames):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(files):
    rows = []
    for path in files:
        rows.extend(read_jsonl(path))

    by_model = defaultdict(list)
    by_pressure = defaultdict(list)
    for row in rows:
        by_model[row.get("model", "unknown")].append(row)
        by_pressure[(row.get("model", "unknown"), row.get("pressure_condition", "unknown"))].append(row)

    summary = {}
    for model, items in sorted(by_model.items()):
        leaks = sum(1 for row in items if row.get("final_leakage_heuristic"))
        summary[model] = {
            "leakage_count": leaks,
            "n": len(items),
            "leakage_rate": leaks / len(items) if items else 0,
        }

    pressure_rows = []
    for (model, pressure), items in sorted(by_pressure.items()):
        leaks = sum(1 for row in items if row.get("final_leakage_heuristic"))
        pressure_rows.append(
            {
                "model": model,
                "pressure_condition": pressure,
                "leakage_count": leaks,
                "n": len(items),
                "leakage_rate": leaks / len(items) if items else 0,
            }
        )
    return summary, pressure_rows


def write_doc(path, summary, summary_json, pressure_csv):
    bullets = []
    for model, stats in summary.items():
        bullets.append(
            "- {}: {}/{} leakage ({:.1%}).".format(
                model, stats["leakage_count"], stats["n"], stats["leakage_rate"]
            )
        )
    text = """# Operational lifting

## Headline numbers
{bullets}

## What we ran
- We constructed typed handoffs for the 12-scenario downstream subset.
- Each handoff includes operational facts with `allowed_audiences: ["downstream_audience"]` and sensitive facts with `allowed_audiences: []`.
- Downstream agents were instructed to mention a fact only when the audience appears in its `allowed_audiences` field, then evaluated under four downstream pressures.

## What we found
- The result tests whether operationalizing the boundary as a typed constraint changes downstream behavior when sensitive facts remain present in the handoff.
- *This is a probe, not a proposed method.*

## How to cite this in the paper
- To test whether boundary metadata must be operationalized rather than merely restated, we supplied downstream agents with a typed `allowed_audiences` schema and a hard filtering instruction. Leakage under this condition measures whether explicit access-control fields reduce pressure-induced disclosure.

## Caveats
- The schema is hand-constructed from the benchmark truth, so this is an existence probe rather than an end-to-end mitigation.
- Leakage is measured with the same calibrated phrase/alias detector used in the downstream experiments.

## Files produced
- `{summary_json}`: overall leakage counts by model.
- `{pressure_csv}`: leakage counts by model and downstream pressure.
""".format(
        bullets="\n".join(bullets),
        summary_json=summary_json,
        pressure_csv=pressure_csv,
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-files", nargs="+", default=DEFAULT_EVAL_FILES)
    parser.add_argument("--summary-json", default="data/operational_lifting_summary.json")
    parser.add_argument("--pressure-csv", default="data/operational_lifting_by_pressure_summary.csv")
    parser.add_argument("--doc", default="docs/operational_lifting_summary.md")
    args = parser.parse_args()

    summary, pressure_rows = summarize(args.eval_files)
    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_json).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_csv(args.pressure_csv, pressure_rows, ["model", "pressure_condition", "leakage_count", "n", "leakage_rate"])
    write_doc(args.doc, summary, args.summary_json, args.pressure_csv)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
