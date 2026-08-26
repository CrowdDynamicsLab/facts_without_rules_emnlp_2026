#!/usr/bin/env python3
"""Bootstrap confidence intervals over scenarios for BOUND-Handoff results."""

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


def load_jsonl(path):
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def percentile(values, pct):
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * pct
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def bootstrap_scenario_means(rows, value_key, group_keys, n_boot, rng):
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        value = row.get(value_key)
        if value is None:
            continue
        if isinstance(value, bool):
            value = float(value)
        group = tuple(row.get(key, "unknown") for key in group_keys)
        groups[group][row["scenario_id"]].append(float(value))

    out = []
    for group, by_scenario in groups.items():
        scenario_values = [mean(vals) for vals in by_scenario.values()]
        scenario_values = [v for v in scenario_values if v is not None]
        if not scenario_values:
            continue
        boot = []
        for _ in range(n_boot):
            sample = [rng.choice(scenario_values) for _ in scenario_values]
            boot.append(mean(sample))
        out.append(
            {
                "group_values": group,
                "n_scenarios": len(scenario_values),
                "n_rows": sum(len(vals) for vals in by_scenario.values()),
                "mean": mean(scenario_values),
                "ci_low": percentile(boot, 0.025),
                "ci_high": percentile(boot, 0.975),
            }
        )
    return out


def add_summary(summary_rows, experiment, model, metric, grouping, group_keys, rows, value_key, n_boot, rng):
    for result in bootstrap_scenario_means(rows, value_key, group_keys, n_boot, rng):
        group = "__overall__" if not group_keys else " | ".join(str(v) for v in result["group_values"])
        summary_rows.append(
            {
                "experiment": experiment,
                "model": model,
                "metric": metric,
                "grouping": grouping,
                "group": group,
                "n_scenarios": result["n_scenarios"],
                "n_rows": result["n_rows"],
                "mean": result["mean"],
                "ci_low": result["ci_low"],
                "ci_high": result["ci_high"],
            }
        )


def expand_phase2_sigma(rows):
    handoff_rows = []
    marker_rows = []
    for row in rows:
        overall = row.get("overall", {})
        base = {
            "scenario_id": row["scenario_id"],
            "condition": row.get("condition", "unknown"),
            "handoff_surface": row.get("handoff_surface", "unknown"),
            "domain": row.get("domain", "unknown"),
        }
        handoff = dict(base)
        handoff["sigma"] = overall.get("boundary_sigma_numeric")
        handoff["sigma_op"] = overall.get("operational_sigma_numeric")
        handoff_rows.append(handoff)
        for marker in row.get("boundary_marker_scores", []):
            marker_row = dict(base)
            marker_row["marker_category"] = marker.get("category", "unknown")
            marker_row["marker_score"] = marker.get("numeric_score")
            marker_rows.append(marker_row)
    return handoff_rows, marker_rows


def normalize_leakage_rows(rows, preferred_label="final_leakage_heuristic"):
    normalized = []
    for row in rows:
        out = {
            "scenario_id": row["scenario_id"],
            "domain": row.get("domain", "unknown"),
            "condition": row.get("condition", "unknown"),
            "handoff_variant": row.get("handoff_variant") or "unknown",
            "mitigation_condition": row.get("mitigation_condition") or "unknown",
            "pressure_condition": row.get("pressure_condition") or "unknown",
        }
        label = row.get(preferred_label)
        if label is None:
            label = row.get("final_leakage_heuristic")
        out["leakage"] = float(bool(label))
        normalized.append(out)
    return normalized


def maybe_load(path):
    if not path.exists():
        print("Missing {}, skipping".format(path))
        return None
    return load_jsonl(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data", type=Path)
    parser.add_argument("--output", default="data/bootstrap_experiment_summary.csv", type=Path)
    parser.add_argument("--n-boot", default=10000, type=int)
    parser.add_argument("--seed", default=20260511, type=int)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    summary_rows = []

    phase2 = maybe_load(args.data_dir / "judge_scores_openai_gpt5mini_phase2_sigma.jsonl")
    if phase2:
        handoff_rows, marker_rows = expand_phase2_sigma(phase2)
        for metric in ["sigma", "sigma_op"]:
            add_summary(summary_rows, "phase2_sigma", "gpt-5-mini", metric, "overall", [], handoff_rows, metric, args.n_boot, rng)
            add_summary(summary_rows, "phase2_sigma", "gpt-5-mini", metric, "condition", ["condition"], handoff_rows, metric, args.n_boot, rng)
            add_summary(summary_rows, "phase2_sigma", "gpt-5-mini", metric, "handoff_surface", ["handoff_surface"], handoff_rows, metric, args.n_boot, rng)
            add_summary(summary_rows, "phase2_sigma", "gpt-5-mini", metric, "domain", ["domain"], handoff_rows, metric, args.n_boot, rng)
        add_summary(summary_rows, "phase2_sigma", "gpt-5-mini", "sigma_marker_item", "marker_category", ["marker_category"], marker_rows, "marker_score", args.n_boot, rng)

    leakage_specs = [
        (
            "phase2b_downstream_manual",
            "gpt-5-mini",
            "model_outputs_openai_gpt5mini_phase2b_downstream_manual_labeled.jsonl",
            "human_final_leakage",
            ["handoff_variant", "pressure_condition", "domain"],
        ),
        (
            "phase2b_downstream_heuristic",
            "deepseek-r1:32b",
            "eval_uiuc_deepseek_phase2b_downstream_calibrated.jsonl",
            "final_leakage_heuristic",
            ["handoff_variant", "pressure_condition", "domain"],
        ),
        (
            "phase2c_downstream_heuristic",
            "gpt-5-mini",
            "eval_openai_gpt5mini_phase2c_downstream_calibrated.jsonl",
            "final_leakage_heuristic",
            ["handoff_variant", "pressure_condition", "domain"],
        ),
        (
            "phase2c_downstream_heuristic",
            "deepseek-r1:32b",
            "eval_uiuc_deepseek_phase2c_downstream_calibrated.jsonl",
            "final_leakage_heuristic",
            ["handoff_variant", "pressure_condition", "domain"],
        ),
        (
            "phase2c_weak_marker_manual_audit",
            "gpt-5-mini",
            "audit_openai_gpt5mini_phase2c_weak_marker_manual_labeled.jsonl",
            "human_final_leakage",
            ["handoff_variant", "pressure_condition", "domain"],
        ),
        (
            "phase2c_weak_marker_manual_audit",
            "deepseek-r1:32b",
            "audit_uiuc_deepseek_phase2c_weak_marker_manual_labeled.jsonl",
            "human_final_leakage",
            ["handoff_variant", "pressure_condition", "domain"],
        ),
        (
            "exp5_prompt_only",
            "gpt-5-mini",
            "eval_openai_gpt5mini_exp5_mitigation_downstream_calibrated.jsonl",
            "final_leakage_heuristic",
            ["mitigation_condition", "pressure_condition", "domain"],
        ),
        (
            "exp5b_enforced_redaction",
            "gpt-5-mini",
            "eval_openai_gpt5mini_exp5b_mitigation_downstream_calibrated.jsonl",
            "final_leakage_heuristic",
            ["mitigation_condition", "pressure_condition", "domain"],
        ),
    ]

    for experiment, model, filename, label_key, group_fields in leakage_specs:
        rows = maybe_load(args.data_dir / filename)
        if not rows:
            continue
        normalized = normalize_leakage_rows(rows, label_key)
        add_summary(summary_rows, experiment, model, "final_leakage", "overall", [], normalized, "leakage", args.n_boot, rng)
        for field in group_fields:
            add_summary(summary_rows, experiment, model, "final_leakage", field, [field], normalized, "leakage", args.n_boot, rng)

    task_success_specs = [
        (
            "phase2_task_success",
            "gpt-5-mini",
            "task_success_openai_gpt5mini_phase2.jsonl",
            ["condition", "handoff_surface", "domain"],
        ),
        (
            "exp5_task_success",
            "gpt-5-mini",
            "task_success_openai_gpt5mini_exp5.jsonl",
            ["mitigation_condition", "pressure_condition", "domain"],
        ),
        (
            "exp5b_task_success",
            "gpt-5-mini",
            "task_success_openai_gpt5mini_exp5b.jsonl",
            ["mitigation_condition", "pressure_condition", "domain"],
        ),
    ]

    for experiment, model, filename, group_fields in task_success_specs:
        rows = maybe_load(args.data_dir / filename)
        if not rows:
            continue
        for row in rows:
            row["task_success"] = float(row.get("numeric_score", 0.0))
        add_summary(summary_rows, experiment, model, "task_success", "overall", [], rows, "task_success", args.n_boot, rng)
        for field in group_fields:
            add_summary(summary_rows, experiment, model, "task_success", field, [field], rows, "task_success", args.n_boot, rng)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["experiment", "model", "metric", "grouping", "group", "n_scenarios", "n_rows", "mean", "ci_low", "ci_high"]
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print("Wrote {} summary rows to {}".format(len(summary_rows), args.output))


if __name__ == "__main__":
    main()
