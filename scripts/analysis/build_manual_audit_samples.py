#!/usr/bin/env python3
"""Build manual-audit sampling sheets for leakage and sigma validation."""

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "dataset" / "data"
OUT = ROOT / "results" / "additional"


def read_jsonl(path):
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def scenario_lookup():
    data = json.loads((DATA / "bound_handoff_phase2.json").read_text(encoding="utf-8"))
    return {s["scenario_id"]: s for s in data["scenarios"]}


def output_lookup(paths):
    out = {}
    for path in paths:
        for row in read_jsonl(path):
            out[row["prompt_id"]] = row
    return out


def build_leakage_sample():
    scenarios = scenario_lookup()
    sources = [
        ("E6", DATA / "eval_openai_gpt5mini_exp5_mitigation_downstream_calibrated.jsonl"),
        ("E6", DATA / "eval_uiuc_deepseek_exp5_mitigation_downstream_calibrated.jsonl"),
        ("E7", DATA / "eval_openai_gpt5mini_exp5b_mitigation_downstream_calibrated.jsonl"),
        ("E7", DATA / "eval_uiuc_deepseek_exp5b_mitigation_downstream_calibrated.jsonl"),
        ("E8", DATA / "eval_openai_gpt5mini_marker_gradient_calibrated.jsonl"),
        ("E8", DATA / "eval_uiuc_deepseek_marker_gradient_calibrated.jsonl"),
    ]
    outputs = output_lookup([
        DATA / "model_outputs_openai_gpt5mini_exp5_mitigation_downstream.jsonl",
        DATA / "model_outputs_uiuc_deepseek_exp5_mitigation_downstream.jsonl",
        DATA / "model_outputs_openai_gpt5mini_exp5b_mitigation_downstream.jsonl",
        DATA / "model_outputs_uiuc_deepseek_exp5b_mitigation_downstream.jsonl",
        DATA / "model_outputs_openai_gpt5mini_marker_gradient.jsonl",
        DATA / "model_outputs_uiuc_deepseek_marker_gradient.jsonl",
    ])
    rows = []
    wanted = {
        ("E6", "structured_schema_safe"),
        ("E6", "structured_schema_unsafe"),
        ("E7", "structured_schema_safe"),
        ("E7", "structured_schema_unsafe"),
        ("E8", "L1_vague_hedge"),
        ("E8", "L2_soft_suggestion"),
    }
    for experiment, path in sources:
        for row in read_jsonl(path):
            condition = row.get("mitigation_condition") or row.get("handoff_variant") or row.get("condition", "").split("__")[0]
            e8_level = row.get("handoff_variant")
            key = (experiment, e8_level if experiment == "E8" else condition)
            if key not in wanted:
                continue
            if len([r for r in rows if r["experiment"] == experiment and r["condition"] == key[1] and r["model"] == row.get("model")]) >= 8:
                continue
            sc = scenarios[row["scenario_id"]]
            out = outputs.get(row["prompt_id"], {})
            rows.append(
                {
                    "row_id": f"leak_audit_{len(rows)+1:03d}",
                    "experiment": experiment,
                    "model": row.get("model"),
                    "scenario": row.get("scenario_id"),
                    "condition": key[1],
                    "pressure": row.get("pressure_condition"),
                    "output_text": out.get("output", ""),
                    "gold_sensitive_fact": "; ".join(sc.get("gold_disallowed_disclosure", [])),
                    "gold_boundary_marker": "; ".join(m.get("text", "") for m in sc.get("boundary_markers", [])),
                    "heuristic_leak_label": row.get("final_leakage_heuristic"),
                    "human_label": "",
                    "notes": "",
                }
            )
    return rows


def build_sigma_sample():
    rows = []
    for path, model in [
        (DATA / "judge_scores_openai_gpt5mini_phase2_sigma.jsonl", "gpt-5-mini"),
        (DATA / "judge_scores_uiuc_deepseek_phase2_sigma.jsonl", "deepseek-r1:32b"),
        (DATA / "judge_scores_openai_gpt5mini_phase2_stresstest_hard_budget_v2_full36_20260512.jsonl", "gpt-5-mini"),
        (DATA / "judge_scores_uiuc_deepseek_phase2_stresstest_hard_budget_v2_full36_20260512.jsonl", "deepseek-r1:32b"),
    ]:
        for row in read_jsonl(path):
            for item in row.get("boundary_marker_scores", []):
                if len(rows) >= 150:
                    return rows
                rows.append(
                    {
                        "row_id": f"sigma_label_{len(rows)+1:03d}",
                        "model": model,
                        "prompt_id": row.get("prompt_id"),
                        "scenario_id": row.get("scenario_id"),
                        "condition": row.get("condition"),
                        "marker_category": item.get("category"),
                        "gold_marker": item.get("gold_marker"),
                        "judge_label": item.get("label"),
                        "judge_rationale": item.get("rationale"),
                        "human_label": "",
                        "notes": "",
                    }
                )
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    leak_rows = build_leakage_sample()
    write_csv(
        OUT / "leakage_manual_audit_sample.csv",
        leak_rows,
        ["row_id", "experiment", "model", "scenario", "condition", "pressure", "output_text", "gold_sensitive_fact", "gold_boundary_marker", "heuristic_leak_label", "human_label", "notes"],
    )
    sigma_rows = build_sigma_sample()
    write_csv(
        OUT / "sigma_judge_validation_sample_to_label.csv",
        sigma_rows,
        ["row_id", "model", "prompt_id", "scenario_id", "condition", "marker_category", "gold_marker", "judge_label", "judge_rationale", "human_label", "notes"],
    )
    summary = """# Leakage Manual Audit Sample

This file is a blind labeling sheet for targeted paraphrase/category leakage audit rows. `human_label` is intentionally blank.

- Leakage audit rows: {leak_n}
- Sigma validation rows for future labeling: {sigma_n}
- Sampling focus: E6 structured-schema variants, E7 redaction high-risk conditions, and E8 L1/L2 marker-gradient rows.

Use these sheets for an expanded human audit before reporting new agreement estimates.
""".format(leak_n=len(leak_rows), sigma_n=len(sigma_rows))
    (OUT / "leakage_manual_audit_summary.md").write_text(summary, encoding="utf-8")
    print(f"Wrote {OUT / 'leakage_manual_audit_sample.csv'}")
    print(f"Wrote {OUT / 'sigma_judge_validation_sample_to_label.csv'}")
    print(f"Wrote {OUT / 'leakage_manual_audit_summary.md'}")


if __name__ == "__main__":
    main()
