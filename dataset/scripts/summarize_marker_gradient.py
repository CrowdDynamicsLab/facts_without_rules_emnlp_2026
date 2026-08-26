#!/usr/bin/env python3
"""Summarize marker-gradient leakage rates by level and model."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


LEVEL_MAP = {
    "dump_no_marker": ("L0_no_marker", 0),
    "dump_weak_marker": ("L1_vague_hedge", 1),
    "L2_soft_suggestion": ("L2_soft_suggestion", 2),
    "L3_constraint": ("L3_constraint", 3),
    "dump_with_marker": ("L4_imperative", 4),
}

EVAL_FILES = [
    "data/eval_openai_gpt5mini_phase2b_downstream_calibrated.jsonl",
    "data/eval_openai_gpt5mini_phase2c_downstream_calibrated.jsonl",
    "data/eval_openai_gpt5mini_marker_gradient_calibrated.jsonl",
    "data/eval_uiuc_deepseek_phase2b_downstream_calibrated.jsonl",
    "data/eval_uiuc_deepseek_phase2c_downstream_calibrated.jsonl",
    "data/eval_uiuc_deepseek_marker_gradient_calibrated.jsonl",
]


def read_jsonl(path):
    rows = []
    path = Path(path)
    if not path.exists():
        return rows
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
        for row in rows:
            writer.writerow(row)


def collect(files):
    by_key = {}
    for path in files:
        for row in read_jsonl(path):
            variant = row.get("handoff_variant")
            if variant not in LEVEL_MAP:
                continue
            key = (row.get("model"), row.get("prompt_id"))
            by_key[key] = row
    return list(by_key.values())


def summarize(rows):
    buckets = defaultdict(list)
    pressure_buckets = defaultdict(list)
    for row in rows:
        level, order = LEVEL_MAP[row["handoff_variant"]]
        key = (row["model"], level, order)
        buckets[key].append(row)
        pressure_buckets[(row["model"], level, order, row.get("pressure_condition"))].append(row)

    summary_rows = []
    for (model, level, order), items in sorted(buckets.items(), key=lambda item: (item[0][0], item[0][2])):
        leaks = sum(1 for row in items if row.get("final_leakage_heuristic"))
        summary_rows.append(
            {
                "model": model,
                "marker_level": level,
                "level_order": order,
                "leakage_count": leaks,
                "n": len(items),
                "leakage_rate": leaks / len(items) if items else 0,
            }
        )

    pressure_rows = []
    for (model, level, order, pressure), items in sorted(
        pressure_buckets.items(), key=lambda item: (item[0][0], item[0][2], item[0][3] or "")
    ):
        leaks = sum(1 for row in items if row.get("final_leakage_heuristic"))
        pressure_rows.append(
            {
                "model": model,
                "marker_level": level,
                "level_order": order,
                "pressure_condition": pressure,
                "leakage_count": leaks,
                "n": len(items),
                "leakage_rate": leaks / len(items) if items else 0,
            }
        )
    return summary_rows, pressure_rows


def plot(rows, path):
    import matplotlib.pyplot as plt

    by_model = defaultdict(list)
    for row in rows:
        by_model[row["model"]].append(row)
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    colors = {"gpt-5-mini": "#4C78A8", "deepseek-r1:32b": "#E45756"}
    for model, items in sorted(by_model.items()):
        items = sorted(items, key=lambda row: int(row["level_order"]))
        ax.plot(
            [row["level_order"] for row in items],
            [row["leakage_rate"] for row in items],
            marker="o",
            linewidth=2,
            label=model,
            color=colors.get(model),
        )
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_xticklabels(["L0\nnone", "L1\nvague", "L2\nsoft", "L3\nconstraint", "L4\nimperative"])
    ax.set_ylim(-0.03, 1.03)
    ax.set_ylabel("Leakage rate")
    ax.set_xlabel("Marker strength")
    ax.grid(True, axis="y", color="#E6E6E6", linewidth=0.8)
    ax.legend(frameon=False)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")


def write_doc(path, summary_rows, csv_path, pressure_csv_path, fig_path):
    lines = []
    by_model = defaultdict(list)
    for row in summary_rows:
        by_model[row["model"]].append(row)
    for model, rows in sorted(by_model.items()):
        parts = []
        for row in sorted(rows, key=lambda item: int(item["level_order"])):
            parts.append("{} {}/{}".format(row["marker_level"], row["leakage_count"], row["n"]))
        lines.append("- {}: {}.".format(model, "; ".join(parts)))
    text = """# Marker-strength gradient

## Headline numbers
{headline}

## What we ran
- We added two marker-strength levels to the existing 12-scenario downstream pressure setup: L3 explicit constraint and L2 soft suggestion.
- We reuse existing L0 no-marker, L1 vague-hedge, and L4 imperative results, yielding a five-level leakage curve over four downstream pressures per scenario and model.

## What we found
- The curve estimates how leakage changes as the boundary marker becomes more explicit.
- *The key comparison is whether L2 remains close to L0/L1 while L3 moves toward L4.*

## How to cite this in the paper
- We convert the strong/weak/no-marker endpoint comparison into a dose-response probe by holding the sensitive fact constant and varying only marker explicitness. The resulting leakage curve tests whether downstream behavior changes gradually or has a sharp transition when the boundary becomes an explicit constraint.

## Caveats
- L0, L1, and L4 are reused from existing Phase 2B/2C runs; L2 and L3 are newly generated.
- Leakage is measured with the calibrated phrase/alias detector used for downstream experiments.

## Files produced
- `{csv_path}`: leakage rate by marker level and model.
- `{pressure_csv_path}`: leakage rate by marker level, model, and downstream pressure.
- `{fig_path}`: marker-gradient curve.
""".format(
        headline="\n".join(lines),
        csv_path=csv_path,
        pressure_csv_path=pressure_csv_path,
        fig_path=fig_path,
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-files", nargs="+", default=EVAL_FILES)
    parser.add_argument("--summary-csv", default="data/marker_gradient_summary.csv")
    parser.add_argument("--pressure-csv", default="data/marker_gradient_by_pressure_summary.csv")
    parser.add_argument("--figure", default="figures/marker_gradient_curve.pdf")
    parser.add_argument("--doc", default="docs/marker_gradient_summary.md")
    args = parser.parse_args()

    rows = collect(args.eval_files)
    summary_rows, pressure_rows = summarize(rows)
    write_csv(args.summary_csv, summary_rows, ["model", "marker_level", "level_order", "leakage_count", "n", "leakage_rate"])
    write_csv(
        args.pressure_csv,
        pressure_rows,
        ["model", "marker_level", "level_order", "pressure_condition", "leakage_count", "n", "leakage_rate"],
    )
    plot(summary_rows, args.figure)
    write_doc(args.doc, summary_rows, args.summary_csv, args.pressure_csv, args.figure)
    print(json.dumps(summary_rows, indent=2))


if __name__ == "__main__":
    main()
