#!/usr/bin/env python3
"""Analyze MiniCheck faithfulness scores against sigma and sigma_op."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def read_rows(path):
    with Path(path).open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key in ["faithfulness_minicheck", "faithfulness_alignscore", "sigma", "sigma_op"]:
            try:
                row[key] = float(row[key])
            except (TypeError, ValueError):
                row[key] = math.nan
        for key in ["n_markers", "n_op_facts"]:
            try:
                row[key] = int(float(row[key]))
            except (TypeError, ValueError):
                row[key] = None
    return rows


def clean_pairs(rows, x_key, y_key):
    out = []
    for row in rows:
        x = row.get(x_key)
        y = row.get(y_key)
        if x is None or y is None or math.isnan(x) or math.isnan(y):
            continue
        out.append((x, y))
    return out


def pearson(pairs):
    if len(pairs) < 2:
        return None
    xbar = sum(x for x, _ in pairs) / len(pairs)
    ybar = sum(y for _, y in pairs) / len(pairs)
    num = sum((x - xbar) * (y - ybar) for x, y in pairs)
    den_x = math.sqrt(sum((x - xbar) ** 2 for x, _ in pairs))
    den_y = math.sqrt(sum((y - ybar) ** 2 for _, y in pairs))
    return None if den_x == 0 or den_y == 0 else num / (den_x * den_y)


def ranks(values):
    indexed = sorted((value, idx) for idx, value in enumerate(values))
    out = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][0] == indexed[i][0]:
            j += 1
        rank = (i + 1 + j) / 2.0
        for _, idx in indexed[i:j]:
            out[idx] = rank
        i = j
    return out


def spearman(pairs):
    if len(pairs) < 2:
        return None
    rx = ranks([x for x, _ in pairs])
    ry = ranks([y for _, y in pairs])
    return pearson(list(zip(rx, ry)))


def median(values):
    clean = sorted(v for v in values if v is not None and not math.isnan(v))
    if not clean:
        return None
    mid = len(clean) // 2
    if len(clean) % 2:
        return clean[mid]
    return (clean[mid - 1] + clean[mid]) / 2


def summarize(rows):
    by_model = defaultdict(list)
    for row in rows:
        by_model[row["model"]].append(row)
    summary = {}
    for model, model_rows in sorted(by_model.items()):
        f_values = [row["faithfulness_minicheck"] for row in model_rows]
        sigma_values = [row["sigma"] for row in model_rows]
        f_med = median(f_values)
        sigma_med = median(sigma_values)
        high_f = [row for row in model_rows if row["faithfulness_minicheck"] >= 0.9]
        high_f_low_sigma = [
            row
            for row in model_rows
            if row["faithfulness_minicheck"] >= f_med and row["sigma"] < sigma_med
        ]
        confusion = {
            "high_f_high_sigma": 0,
            "high_f_low_sigma": 0,
            "low_f_high_sigma": 0,
            "low_f_low_sigma": 0,
        }
        for row in model_rows:
            f_high = row["faithfulness_minicheck"] >= f_med
            sigma_high = row["sigma"] >= sigma_med
            if f_high and sigma_high:
                confusion["high_f_high_sigma"] += 1
            elif f_high and not sigma_high:
                confusion["high_f_low_sigma"] += 1
            elif not f_high and sigma_high:
                confusion["low_f_high_sigma"] += 1
            else:
                confusion["low_f_low_sigma"] += 1
        summary[model] = {
            "n": len(model_rows),
            "faithfulness_median": f_med,
            "sigma_median": sigma_med,
            "pearson_r_f_sigma": pearson(clean_pairs(model_rows, "faithfulness_minicheck", "sigma")),
            "spearman_rho_f_sigma": spearman(clean_pairs(model_rows, "faithfulness_minicheck", "sigma")),
            "pearson_r_f_sigma_op": pearson(clean_pairs(model_rows, "faithfulness_minicheck", "sigma_op")),
            "spearman_rho_f_sigma_op": spearman(clean_pairs(model_rows, "faithfulness_minicheck", "sigma_op")),
            "median_sigma_in_f_ge_0_9_slice": median([row["sigma"] for row in high_f]),
            "n_f_ge_0_9": len(high_f),
            "median_threshold_confusion": confusion,
            "high_f_low_sigma_count": len(high_f_low_sigma),
        }
    return summary


def plot(rows, path):
    import matplotlib.pyplot as plt

    models = sorted({row["model"] for row in rows})
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharex=True)
    colors = {"gpt-5-mini": "#4C78A8", "deepseek-r1:32b": "#E45756"}
    for model in models:
        subset = [row for row in rows if row["model"] == model]
        axes[0].scatter(
            [row["faithfulness_minicheck"] for row in subset],
            [row["sigma"] for row in subset],
            label=model,
            color=colors.get(model, "#666666"),
            alpha=0.8,
            s=38,
            edgecolor="white",
            linewidth=0.4,
        )
        axes[1].scatter(
            [row["faithfulness_minicheck"] for row in subset],
            [row["sigma_op"] for row in subset],
            label=model,
            color=colors.get(model, "#666666"),
            alpha=0.8,
            s=38,
            edgecolor="white",
            linewidth=0.4,
        )
    axes[0].set_ylabel("Boundary survival (sigma)")
    axes[1].set_ylabel("Operational survival (sigma_op)")
    for ax in axes:
        ax.set_xlabel("MiniCheck faithfulness")
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)
        ax.grid(True, color="#E6E6E6", linewidth=0.7)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")


def write_doc(path, summary, csv_path, json_path, fig_path):
    lines = []
    findings = []
    for model, stats in summary.items():
        lines.append(
            f"- {model}: r(F, sigma_op) = {stats['pearson_r_f_sigma_op']:.3f}; "
            f"r(F, sigma) = {stats['pearson_r_f_sigma']:.3f}; "
            f"median sigma in F >= 0.9 slice = {stats['median_sigma_in_f_ge_0_9_slice']}; "
            f"high-F-low-sigma median cell = {stats['high_f_low_sigma_count']}."
        )
        findings.append(
            f"{model} has {stats['high_f_low_sigma_count']} high-faithfulness/low-sigma rows under median thresholds."
        )
    text = f"""# Faithfulness comparison

## Headline numbers
{chr(10).join(lines)}

## What we ran
- We scored E1 `free_text` handoffs with MiniCheck-Flan-T5-Large, using the upstream transcript as the grounding document and sentence-like handoff chunks as claims.
- AlignScore was not run in this pass; its column is present as NaN for later secondary scoring.

## What we found
- {' '.join(findings)}
- *MiniCheck faithfulness is a grounding score, while sigma measures whether boundary metadata survived the handoff.*

## How to cite this in the paper
- We compare sigma to an off-the-shelf NLI-style faithfulness metric on free-text handoffs. This separates ordinary factual grounding from boundary preservation: a handoff can be supported by the transcript while still losing the audience limits, caveats, and ownership metadata needed for safe downstream use.

## Caveats
- MiniCheck is sentence-level and trained for document-grounded fact checking, not multi-turn handoff safety.
- The generated handoffs are split into sentence-like claims heuristically.
- AlignScore remains a secondary sanity check to run before using a two-metric claim.

## Files produced
- `{csv_path}`: per-handoff faithfulness and sigma table.
- `{json_path}`: per-handoff JSONL with claim probabilities.
- `{fig_path}`: faithfulness-vs-sigma and faithfulness-vs-sigma_op scatter.
"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/faithfulness_scores_e1_free_text.csv")
    parser.add_argument("--json", default="data/faithfulness_summary.json")
    parser.add_argument("--figure", default="figures/faithfulness_vs_sigma_scatter.pdf")
    parser.add_argument("--doc", default="docs/faithfulness_summary.md")
    args = parser.parse_args()

    rows = read_rows(args.csv)
    summary = summarize(rows)
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    plot(rows, args.figure)
    write_doc(args.doc, summary, args.csv, "data/faithfulness_scores_e1_free_text.jsonl", args.figure)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
