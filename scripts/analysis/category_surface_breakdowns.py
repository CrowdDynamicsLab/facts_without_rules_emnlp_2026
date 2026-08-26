#!/usr/bin/env python3
"""Compute category/surface sigma summaries for E1 and E2."""

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "dataset" / "data"
OUT = ROOT / "results" / "additional"
WEIGHTS = {"preserved": 1.0, "paraphrased": 0.75, "weakened": 0.35, "absent": 0.0}

RUNS = [
    ("E1", "gpt-5-mini", "baseline", DATA / "judge_scores_openai_gpt5mini_phase2_sigma.jsonl"),
    ("E1", "deepseek-r1:32b", "baseline", DATA / "judge_scores_uiuc_deepseek_phase2_sigma.jsonl"),
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


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def bootstrap_ci(values, reps=1000):
    if not values:
        return float("nan"), float("nan")
    # Deterministic bootstrap via a small LCG to avoid extra dependencies.
    seed = 1234567
    means = []
    n = len(values)
    for _ in range(reps):
        sample = []
        for _ in range(n):
            seed = (1103515245 * seed + 12345) % (2 ** 31)
            sample.append(values[seed % n])
        means.append(mean(sample))
    means.sort()
    return means[int(0.025 * reps)], means[int(0.975 * reps)]


def summarize():
    out = []
    for experiment, model, run, path in RUNS:
        rows = read_jsonl(path)
        marker_by_cat = defaultdict(list)
        op_by_surface = defaultdict(list)
        marker_by_surface = defaultdict(list)
        for row in rows:
            surface = row.get("handoff_surface", "unknown")
            for item in row.get("boundary_marker_scores", []):
                score = WEIGHTS.get(item.get("label"), 0.0)
                marker_by_cat[item.get("category", "other")].append(score)
                marker_by_surface[surface].append(score)
            op_scores = [WEIGHTS.get(item.get("label"), 0.0) for item in row.get("operational_fact_scores", [])]
            if op_scores:
                op_by_surface[surface].extend(op_scores)
        for grouping, buckets, metric in [
            ("marker_category", marker_by_cat, "sigma"),
            ("handoff_surface", marker_by_surface, "sigma"),
            ("handoff_surface", op_by_surface, "sigma_op"),
        ]:
            for group, values in sorted(buckets.items()):
                ci = bootstrap_ci(values)
                out.append(
                    {
                        "experiment": experiment,
                        "model": model,
                        "run": run,
                        "grouping": grouping,
                        "group": group,
                        "metric": metric,
                        "n_items": len(values),
                        "mean": mean(values),
                        "ci_low": ci[0],
                        "ci_high": ci[1],
                    }
                )
    return out


def write_csv(path, rows):
    fields = ["experiment", "model", "run", "grouping", "group", "metric", "n_items", "mean", "ci_low", "ci_high"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            for key in ["mean", "ci_low", "ci_high"]:
                out[key] = "" if math.isnan(out[key]) else f"{out[key]:.6g}"
            writer.writerow(out)


def write_md(path, rows):
    lines = [
        "# Category And Surface Breakdowns",
        "",
        "Bootstrap confidence intervals are descriptive item-level intervals over judged marker/fact scores.",
        "",
        "## Most fragile 25-word boundary slices",
        "",
        "| Model | Grouping | Group | Mean sigma | 95% CI | n |",
        "|---|---|---|---:|---:|---:|",
    ]
    fragile = [
        r for r in rows
        if r["experiment"] == "E2" and r["run"] == "25_word" and r["metric"] == "sigma"
    ]
    for r in sorted(fragile, key=lambda x: x["mean"])[:10]:
        lines.append(
            f"| {r['model']} | {r['grouping']} | {r['group']} | {r['mean']:.3f} | [{r['ci_low']:.3f}, {r['ci_high']:.3f}] | {r['n_items']} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "The table identifies which marker categories and handoff surfaces are most fragile under the 25-word compression stress test. Treat these as appendix diagnostics rather than standalone causal claims.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    rows = summarize()
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "category_surface_breakdowns.csv", rows)
    write_md(OUT / "category_surface_breakdowns.md", rows)
    print(f"Wrote {OUT / 'category_surface_breakdowns.csv'}")
    print(f"Wrote {OUT / 'category_surface_breakdowns.md'}")


if __name__ == "__main__":
    main()
