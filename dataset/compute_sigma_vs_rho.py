#!/usr/bin/env python3
"""Compute compression ratio rho and plot sigma/sigma_op trajectories."""

import argparse
import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path


def load_jsonl(path):
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def tokenize(text):
    return re.findall(r"\w+|[^\w\s]", text or "", re.UNICODE)


def transcript_text(scenario):
    parts = []
    for message in scenario["upstream_transcript"]:
        recipients = ", ".join(message["recipients"])
        parts.append(
            "[{phase}] {sender} -> {recipients}: {content}".format(
                phase=message["phase"],
                sender=message["sender"],
                recipients=recipients,
                content=message["content"],
            )
        )
    return "\n".join(parts)


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def percentile(values, pct):
    values = sorted(values)
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * pct
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def bootstrap_ci(scenario_values, key, n_boot, rng):
    vals = [v[key] for v in scenario_values if v.get(key) is not None]
    if not vals:
        return None, None
    boot = []
    for _ in range(n_boot):
        boot.append(mean(rng.choice(vals) for _ in vals))
    return percentile(boot, 0.025), percentile(boot, 0.975)


def collect_rows(dataset, output_paths, judge_paths, experiment_labels):
    scenario_by_id = {s["scenario_id"]: s for s in dataset["scenarios"]}
    combined = []
    for outputs_path, judge_path, experiment in zip(output_paths, judge_paths, experiment_labels):
        outputs = {r["prompt_id"]: r for r in load_jsonl(outputs_path)}
        judges = load_jsonl(judge_path)
        for judge in judges:
            output = outputs[judge["prompt_id"]]
            scenario = scenario_by_id[judge["scenario_id"]]
            upstream_tokens = len(tokenize(transcript_text(scenario)))
            handoff_tokens = len(tokenize(output.get("output", "")))
            rho = 1 - (handoff_tokens / upstream_tokens) if upstream_tokens else None
            overall = judge.get("overall", {})
            combined.append(
                {
                    "experiment": experiment,
                    "prompt_id": judge["prompt_id"],
                    "scenario_id": judge["scenario_id"],
                    "domain": judge.get("domain", output.get("domain", scenario["domain"])),
                    "handoff_surface": judge.get("handoff_surface", output.get("handoff_surface", scenario["handoff_surface"])),
                    "condition": judge.get("condition", output.get("condition", "unknown")),
                    "upstream_tokens": upstream_tokens,
                    "handoff_tokens": handoff_tokens,
                    "rho": rho,
                    "sigma": overall.get("boundary_sigma_numeric"),
                    "sigma_op": overall.get("operational_sigma_numeric"),
                }
            )
    return combined


def summarize(rows, group_key, n_boot, seed):
    rng = random.Random(seed)
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row[group_key]][row["scenario_id"]].append(row)

    summaries = []
    for group, by_scenario in grouped.items():
        scenario_values = []
        for scenario_id, items in by_scenario.items():
            scenario_values.append(
                {
                    "scenario_id": scenario_id,
                    "rho": mean(item["rho"] for item in items),
                    "sigma": mean(item["sigma"] for item in items),
                    "sigma_op": mean(item["sigma_op"] for item in items),
                }
            )
        row = {
            group_key: group,
            "n_scenarios": len(scenario_values),
            "n_handoffs": sum(len(v) for v in by_scenario.values()),
            "mean_rho": mean(v["rho"] for v in scenario_values),
            "mean_sigma": mean(v["sigma"] for v in scenario_values),
            "mean_sigma_op": mean(v["sigma_op"] for v in scenario_values),
        }
        for key in ["rho", "sigma", "sigma_op"]:
            lo, hi = bootstrap_ci(scenario_values, key, n_boot, rng)
            row[key + "_ci95_lo"] = lo
            row[key + "_ci95_hi"] = hi
        summaries.append(row)
    return sorted(summaries, key=lambda r: r["mean_rho"])


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_plot(summary_rows, output):
    def esc(text):
        return (
            str(text)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    def short_label(label):
        return {
            "free_text": "free",
            "compressed_free_text": "bullets",
            "preserve_markers_instruction": "preserve",
            "sectioned_template": "sections",
            "structured_schema": "schema",
            "hard_budget_compressed": "40w",
            "hard_budget_compressed_v2": "25w",
        }.get(label, label)

    width, height = 860, 520
    left, right, top, bottom = 78, 28, 34, 78
    plot_w = width - left - right
    plot_h = height - top - bottom
    x_min = min(r["mean_rho"] for r in summary_rows) - 0.02
    x_max = max(r["mean_rho"] for r in summary_rows) + 0.02
    y_min, y_max = 0.45, 1.03

    def sx(x):
        return left + (x - x_min) / (x_max - x_min) * plot_w

    def sy(y):
        return top + (y_max - y) / (y_max - y_min) * plot_h

    def points(rows, key):
        return " ".join("{:.2f},{:.2f}".format(sx(r["mean_rho"]), sy(r[key])) for r in rows)

    output = output.with_suffix(".svg")
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        '<svg xmlns="[URL]" width="{w}" height="{h}" viewBox="0 0 {w} {h}">'.format(w=width, h=height),
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;font-size:13px;fill:#1f2933}.axis{stroke:#28323c;stroke-width:1.2}.grid{stroke:#d9dee5;stroke-width:1}.sigma{stroke:#1f77b4;fill:none;stroke-width:3}.op{stroke:#d62728;fill:none;stroke-width:3}.ci{stroke-width:1.8;opacity:.45}.label{font-size:12px;fill:#334155}.title{font-size:16px;font-weight:700}</style>',
        '<text class="title" x="{x}" y="22">Boundary vs operational survival under compression</text>'.format(x=left),
    ]

    for y in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        yy = sy(y)
        lines.append('<line class="grid" x1="{:.2f}" y1="{:.2f}" x2="{:.2f}" y2="{:.2f}"/>'.format(left, yy, width - right, yy))
        lines.append('<text x="{:.2f}" y="{:.2f}" text-anchor="end">{:.1f}</text>'.format(left - 8, yy + 4, y))
    for x in [round(x_min + i * (x_max - x_min) / 5, 2) for i in range(6)]:
        xx = sx(x)
        lines.append('<line class="grid" x1="{:.2f}" y1="{:.2f}" x2="{:.2f}" y2="{:.2f}"/>'.format(xx, top, xx, height - bottom))
        lines.append('<text x="{:.2f}" y="{:.2f}" text-anchor="middle">{:.2f}</text>'.format(xx, height - bottom + 22, x))

    lines.append('<line class="axis" x1="{:.2f}" y1="{:.2f}" x2="{:.2f}" y2="{:.2f}"/>'.format(left, height - bottom, width - right, height - bottom))
    lines.append('<line class="axis" x1="{:.2f}" y1="{:.2f}" x2="{:.2f}" y2="{:.2f}"/>'.format(left, top, left, height - bottom))

    for r in summary_rows:
        x = sx(r["mean_rho"])
        lines.append('<line class="ci" stroke="#1f77b4" x1="{:.2f}" y1="{:.2f}" x2="{:.2f}" y2="{:.2f}"/>'.format(x, sy(r["sigma_ci95_lo"]), x, sy(r["sigma_ci95_hi"])))
        lines.append('<line class="ci" stroke="#d62728" x1="{:.2f}" y1="{:.2f}" x2="{:.2f}" y2="{:.2f}"/>'.format(x + 4, sy(r["sigma_op_ci95_lo"]), x + 4, sy(r["sigma_op_ci95_hi"])))

    lines.append('<polyline class="sigma" points="{}"/>'.format(points(summary_rows, "mean_sigma")))
    lines.append('<polyline class="op" points="{}"/>'.format(points(summary_rows, "mean_sigma_op")))
    for r in summary_rows:
        x = sx(r["mean_rho"])
        y1 = sy(r["mean_sigma"])
        y2 = sy(r["mean_sigma_op"])
        lines.append('<circle cx="{:.2f}" cy="{:.2f}" r="4.8" fill="#1f77b4"/>'.format(x, y1))
        lines.append('<rect x="{:.2f}" y="{:.2f}" width="8" height="8" fill="#d62728"/>'.format(x, y2 - 4))
        lines.append('<text class="label" x="{:.2f}" y="{:.2f}">{}</text>'.format(x + 7, y1 + 16, esc(short_label(r["condition"]))))

    lines.append('<text x="{:.2f}" y="{:.2f}" text-anchor="middle">Compression ratio ρ = 1 - |handoff| / |upstream transcript|</text>'.format(left + plot_w / 2, height - 18))
    lines.append('<text transform="translate(20,{:.2f}) rotate(-90)" text-anchor="middle">Mean survival score</text>'.format(top + plot_h / 2))
    lines.append('<line x1="{:.2f}" y1="48" x2="{:.2f}" y2="48" stroke="#1f77b4" stroke-width="3"/><text x="{:.2f}" y="52">Boundary survival σ</text>'.format(width - 245, width - 205, width - 196))
    lines.append('<line x1="{:.2f}" y1="70" x2="{:.2f}" y2="70" stroke="#d62728" stroke-width="3"/><text x="{:.2f}" y="74">Operational survival σ_op</text>'.format(width - 245, width - 205, width - 196))
    lines.append("</svg>")
    output.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("dataset/data/bound_handoff_phase2.json"))
    parser.add_argument("--output-files", nargs="+", type=Path, required=True)
    parser.add_argument("--judge-files", nargs="+", type=Path, required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--per-handoff-output", type=Path, default=Path("dataset/data/sigma_vs_rho_per_handoff.csv"))
    parser.add_argument("--summary-output", type=Path, default=Path("dataset/data/sigma_vs_rho_by_condition.csv"))
    parser.add_argument("--figure-output", type=Path, default=Path("dataset/figures/sigma_vs_rho.png"))
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260512)
    args = parser.parse_args()

    if not (len(args.output_files) == len(args.judge_files) == len(args.labels)):
        raise SystemExit("--output-files, --judge-files, and --labels must have same length")

    dataset = json.loads(args.dataset.read_text())
    rows = collect_rows(dataset, args.output_files, args.judge_files, args.labels)
    write_csv(args.per_handoff_output, rows)
    summary = summarize(rows, "condition", args.n_boot, args.seed)
    write_csv(args.summary_output, summary)
    make_plot(summary, args.figure_output)
    print("Wrote {}".format(args.per_handoff_output))
    print("Wrote {}".format(args.summary_output))
    print("Wrote {}".format(args.figure_output))


if __name__ == "__main__":
    main()
