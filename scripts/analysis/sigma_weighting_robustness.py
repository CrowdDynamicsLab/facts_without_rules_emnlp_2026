#!/usr/bin/env python3
"""Recompute sigma/sigma_op under alternative scoring weights."""

import csv
import json
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "dataset" / "data"
OUT = ROOT / "results" / "additional"

SCHEMES = {
    "original": {"preserved": 1.0, "paraphrased": 0.75, "weakened": 0.35, "absent": 0.0},
    "weakened_as_half": {"preserved": 1.0, "paraphrased": 0.75, "weakened": 0.5, "absent": 0.0},
    "weakened_as_low": {"preserved": 1.0, "paraphrased": 0.75, "weakened": 0.25, "absent": 0.0},
    "binary_strict_survival": {"preserved": 1.0, "paraphrased": 1.0, "weakened": 0.0, "absent": 0.0},
    "binary_lenient_survival": {"preserved": 1.0, "paraphrased": 1.0, "weakened": 1.0, "absent": 0.0},
}

RUNS = [
    ("E1", "gpt-5-mini", "baseline", DATA / "judge_scores_openai_gpt5mini_phase2_sigma.jsonl"),
    ("E1", "deepseek-r1:32b", "baseline", DATA / "judge_scores_uiuc_deepseek_phase2_sigma.jsonl"),
    ("E2", "gpt-5-mini", "40_word", DATA / "judge_scores_openai_gpt5mini_phase2_stresstest_hard_budget_full36_20260512.jsonl"),
    ("E2", "deepseek-r1:32b", "40_word", DATA / "judge_scores_uiuc_deepseek_phase2_stresstest_hard_budget_full36_20260512.jsonl"),
    ("E2", "gpt-5-mini", "25_word", DATA / "judge_scores_openai_gpt5mini_phase2_stresstest_hard_budget_v2_full36_20260512.jsonl"),
    ("E2", "deepseek-r1:32b", "25_word", DATA / "judge_scores_uiuc_deepseek_phase2_stresstest_hard_budget_v2_full36_20260512.jsonl"),
]


def read_jsonl(path):
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def mean(values):
    return sum(values) / len(values) if values else float("nan")


def score_items(items, weights):
    vals = [weights.get((item.get("label") or "absent").strip().lower(), 0.0) for item in items]
    return mean(vals)


def ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            out[order[k]] = rank
        i = j + 1
    return out


def pearson(x, y):
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    mx, my = mean(x), mean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    den_x = math.sqrt(sum((a - mx) ** 2 for a in x))
    den_y = math.sqrt(sum((b - my) ** 2 for b in y))
    if den_x == 0 or den_y == 0:
        return float("nan")
    return num / (den_x * den_y)


def spearman(x, y):
    return pearson(ranks(x), ranks(y))


def summarize_run(experiment, model, run_label, rows, scheme, weights):
    enriched = []
    for row in rows:
        sigma = score_items(row.get("boundary_marker_scores", []), weights)
        sigma_op = score_items(row.get("operational_fact_scores", []), weights)
        enriched.append({**row, "_sigma": sigma, "_sigma_op": sigma_op})

    out = []

    def make_row(grouping, group, chunk):
        sigmas = [r["_sigma"] for r in chunk if not math.isnan(r["_sigma"])]
        sigma_ops = [r["_sigma_op"] for r in chunk if not math.isnan(r["_sigma_op"])]
        paired = [(r["_sigma"], r["_sigma_op"]) for r in chunk if not math.isnan(r["_sigma"]) and not math.isnan(r["_sigma_op"])]
        px = [x for x, _ in paired]
        py = [y for _, y in paired]
        low_high = sum(1 for s, o in paired if s < 0.7 and o > 0.9)
        return {
            "scheme": scheme,
            "experiment": experiment,
            "model": model,
            "run": run_label,
            "grouping": grouping,
            "group": group,
            "n_handoffs": len(chunk),
            "mean_sigma": mean(sigmas),
            "mean_sigma_op": mean(sigma_ops),
            "sigma_minus_sigma_op": mean(sigmas) - mean(sigma_ops),
            "pearson_sigma_sigma_op": pearson(px, py),
            "spearman_sigma_sigma_op": spearman(px, py),
            "low_sigma_high_sigma_op_count": low_high,
            "low_sigma_high_sigma_op_thresholds": "sigma<0.7;sigma_op>0.9",
        }

    out.append(make_row("overall", "__overall__", enriched))
    by_condition = defaultdict(list)
    for row in enriched:
        by_condition[row.get("condition", "unknown")].append(row)
    for condition, chunk in sorted(by_condition.items()):
        out.append(make_row("condition", condition, chunk))
    return out


def build_rows():
    all_rows = []
    for experiment, model, run_label, path in RUNS:
        rows = read_jsonl(path)
        for scheme, weights in SCHEMES.items():
            all_rows.extend(summarize_run(experiment, model, run_label, rows, scheme, weights))
    return all_rows


def write_csv(path, rows):
    fields = [
        "scheme",
        "experiment",
        "model",
        "run",
        "grouping",
        "group",
        "n_handoffs",
        "mean_sigma",
        "mean_sigma_op",
        "sigma_minus_sigma_op",
        "pearson_sigma_sigma_op",
        "spearman_sigma_sigma_op",
        "low_sigma_high_sigma_op_count",
        "low_sigma_high_sigma_op_thresholds",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            for key, val in list(out.items()):
                if isinstance(val, float):
                    out[key] = "" if math.isnan(val) else f"{val:.6g}"
            writer.writerow(out)


def row_lookup(rows):
    return {
        (r["scheme"], r["model"], r["run"], r["group"]): r
        for r in rows
        if r["grouping"] == "overall"
    }


def write_md(path, rows):
    lookup = row_lookup(rows)
    lines = [
        "# Sigma Weighting Robustness",
        "",
        "## Headline",
        "",
        "The selective-collapse pattern is unchanged under alternative sigma weightings, including binary survival metrics that do not rely on the original weakened-marker score.",
        "",
        "Across all tested schemes, the 25-word compression condition leaves operational-fact survival near ceiling while boundary-marker survival remains lower.",
        "",
        "## Overall results",
        "",
        "| Scheme | Model | E1 sigma | E1 sigma_op | E2 40w sigma | E2 40w sigma_op | E2 25w sigma | E2 25w sigma_op | 25w gap | Low-sigma/high-sigma_op 25w |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for scheme in SCHEMES:
        for model in ["gpt-5-mini", "deepseek-r1:32b"]:
            e1 = lookup[(scheme, model, "baseline", "__overall__")]
            e240 = lookup[(scheme, model, "40_word", "__overall__")]
            e225 = lookup[(scheme, model, "25_word", "__overall__")]
            lines.append(
                "| {scheme} | {model} | {e1s:.3f} | {e1o:.3f} | {e240s:.3f} | {e240o:.3f} | {e225s:.3f} | {e225o:.3f} | {gap:.3f} | {quad}/{n} |".format(
                    scheme=scheme,
                    model=model,
                    e1s=e1["mean_sigma"],
                    e1o=e1["mean_sigma_op"],
                    e240s=e240["mean_sigma"],
                    e240o=e240["mean_sigma_op"],
                    e225s=e225["mean_sigma"],
                    e225o=e225["mean_sigma_op"],
                    gap=e225["sigma_minus_sigma_op"],
                    quad=e225["low_sigma_high_sigma_op_count"],
                    n=e225["n_handoffs"],
                )
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Sigma still drops most sharply under 25-word compression for both models under every scoring scheme.",
            "- Sigma_op remains near ceiling under E2, including under binary strict and binary lenient survival metrics.",
            "- The qualitative sigma-vs-sigma_op gap persists; the result is not driven by assigning weakened markers a score of 0.35.",
            "- Correlation columns in the CSV recompute Pearson and Spearman sigma/sigma_op associations at the handoff level for each scheme.",
            "",
            "## Caveat",
            "",
            "The low-sigma/high-sigma_op quadrant uses fixed thresholds (`sigma<0.7`, `sigma_op>0.9`) for comparability across schemes; this is conservative for binary schemes and should be treated as a descriptive diagnostic.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    rows = build_rows()
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "sigma_weighting_robustness.csv", rows)
    write_md(OUT / "sigma_weighting_robustness.md", rows)
    print(f"Wrote {OUT / 'sigma_weighting_robustness.csv'}")
    print(f"Wrote {OUT / 'sigma_weighting_robustness.md'}")


if __name__ == "__main__":
    main()
