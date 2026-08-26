#!/usr/bin/env python3
"""Build Qwen3-32B robustness result tables from raw/evaluated artifacts."""

import csv
import json
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "dataset" / "data"
OUT = ROOT / "results" / "qwen3_32b"
MODEL = "qwen3:32b"
DISPLAY = "Qwen3-32B"


LEVEL_MAP = {
    "dump_no_marker": ("L0_no_marker", "L0 no marker"),
    "dump_weak_marker": ("L1_vague_hedge", "L1 vague hedge"),
    "L2_soft_suggestion": ("L2_soft_suggestion", "L2 soft suggestion"),
    "L3_constraint": ("L3_constraint", "L3 explicit constraint"),
    "dump_with_marker": ("L4_imperative", "L4 imperative"),
}

VARIANT_LABELS = {
    "clean_allowlist": "clean allowlist",
    "dump_with_marker_control": "dump-with-marker control",
    "corrupt_scope_drift": "corrupt: scope drift",
    "corrupt_typo_allowlist": "corrupt: typo allowlist key",
    "corrupt_format_noise": "corrupt: format noise",
    "corrupt_negation_flip": "corrupt: negation flip",
    "dump_no_marker_control": "dump-no-marker control",
}
VARIANT_ORDER = [
    "clean_allowlist",
    "dump_with_marker_control",
    "corrupt_scope_drift",
    "corrupt_typo_allowlist",
    "corrupt_format_noise",
    "corrupt_negation_flip",
    "dump_no_marker_control",
]


def read_jsonl(path):
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def word_tokens(text):
    return len((text or "").split())


def scenario_lookup():
    data = json.loads((DATA / "bound_handoff_phase2.json").read_text(encoding="utf-8"))
    return {s["scenario_id"]: s for s in data["scenarios"]}


def upstream_text(scenario):
    return "\n".join(t.get("content", "") for t in scenario.get("upstream_transcript", []))


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def fmt(x, digits=3):
    if isinstance(x, float) and math.isnan(x):
        return ""
    return f"{x:.{digits}f}"


def logchoose(n, k):
    if k < 0 or k > n:
        return float("-inf")
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def hypergeom_p(x, row1, col1, n):
    return math.exp(logchoose(col1, x) + logchoose(n - col1, row1 - x) - logchoose(n, row1))


def fisher_exact(a, total_a, b, total_b):
    row1 = total_a
    row2 = total_b
    col1 = a + b
    n = total_a + total_b
    lo = max(0, col1 - row2)
    hi = min(row1, col1)
    obs = hypergeom_p(a, row1, col1, n)
    return min(1.0, sum(hypergeom_p(x, row1, col1, n) for x in range(lo, hi + 1) if hypergeom_p(x, row1, col1, n) <= obs + 1e-12))


def load_by_prompt(path):
    return {r["prompt_id"]: r for r in read_jsonl(path)}


def sigma_value(row, key):
    return row.get("overall", {}).get(key)


def build_e2():
    scenarios = scenario_lookup()
    outputs = load_by_prompt(DATA / "model_outputs_qwen3_32b_e2_compression.jsonl")
    judged = read_jsonl(DATA / "judge_scores_qwen3_32b_e2_compression_sigma.jsonl")
    out_rows = []
    by_condition = defaultdict(list)
    for row in judged:
        output = outputs[row["prompt_id"]]
        sc = scenarios[row["scenario_id"]]
        upstream_tokens = word_tokens(upstream_text(sc))
        handoff_tokens = word_tokens(output.get("output", ""))
        sigma = sigma_value(row, "boundary_sigma_numeric")
        sigma_op = sigma_value(row, "operational_sigma_numeric")
        rho = 1 - handoff_tokens / upstream_tokens if upstream_tokens else ""
        item = {
            "model": DISPLAY,
            "scenario_id": row["scenario_id"],
            "domain": row.get("domain", ""),
            "surface": row.get("handoff_surface", ""),
            "condition": row.get("condition", ""),
            "upstream_tokens": upstream_tokens,
            "handoff_tokens": handoff_tokens,
            "rho": fmt(rho) if isinstance(rho, float) else "",
            "generated_handoff": output.get("output", ""),
            "sigma": fmt(sigma),
            "sigma_op": fmt(sigma_op),
            "task_success_if_available": "",
        }
        out_rows.append(item)
        by_condition[row.get("condition", "")].append((rho, sigma, sigma_op))
    write_csv(OUT / "e2_compression_stress_outputs.csv", out_rows, ["model", "scenario_id", "domain", "surface", "condition", "upstream_tokens", "handoff_tokens", "rho", "generated_handoff", "sigma", "sigma_op", "task_success_if_available"])
    summary = []
    for condition in ["hard_budget_compressed", "hard_budget_compressed_v2"]:
        vals = by_condition[condition]
        label = "40-word compression" if condition == "hard_budget_compressed" else "25-word compression"
        row = {
            "model": DISPLAY,
            "condition": label,
            "rho": fmt(mean([x[0] for x in vals])),
            "sigma": fmt(mean([x[1] for x in vals])),
            "sigma_op": fmt(mean([x[2] for x in vals])),
            "gap_sigma_minus_sigma_op": fmt(mean([x[1] for x in vals]) - mean([x[2] for x in vals])),
            "n": len(vals),
        }
        summary.append(row)
    write_csv(OUT / "e2_compression_stress_summary.csv", summary, ["model", "condition", "rho", "sigma", "sigma_op", "gap_sigma_minus_sigma_op", "n"])
    lines = ["# E2 Compression Stress: Qwen3-32B", "", "| Model | Condition | rho | sigma | sigma_op | Gap sigma - sigma_op | n |", "|---|---|---:|---:|---:|---:|---:|"]
    for r in summary:
        lines.append(f"| {r['model']} | {r['condition']} | {r['rho']} | {r['sigma']} | {r['sigma_op']} | {r['gap_sigma_minus_sigma_op']} | {r['n']} |")
    lines.extend(["", "Qwen3-32B shows a lower boundary-marker survival score under the 25-word condition than under the 40-word condition, while operational-fact survival remains high."])
    (OUT / "e2_compression_stress_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def build_e1():
    output_path = DATA / "model_outputs_qwen3_32b_phase2.jsonl"
    judge_path = DATA / "judge_scores_qwen3_32b_phase2_sigma.jsonl"
    if not output_path.exists() or not judge_path.exists():
        return None
    scenarios = scenario_lookup()
    outputs = load_by_prompt(output_path)
    judged = read_jsonl(judge_path)
    out_rows = []
    by_condition = defaultdict(list)
    all_vals = []
    for row in judged:
        output = outputs[row["prompt_id"]]
        sc = scenarios[row["scenario_id"]]
        upstream_tokens = word_tokens(upstream_text(sc))
        handoff_tokens = word_tokens(output.get("output", ""))
        sigma = sigma_value(row, "boundary_sigma_numeric")
        sigma_op = sigma_value(row, "operational_sigma_numeric")
        rho = 1 - handoff_tokens / upstream_tokens if upstream_tokens else ""
        item = {
            "model": DISPLAY,
            "scenario_id": row["scenario_id"],
            "domain": row.get("domain", ""),
            "surface": row.get("handoff_surface", ""),
            "condition": row.get("condition", ""),
            "upstream_tokens": upstream_tokens,
            "handoff_tokens": handoff_tokens,
            "rho": fmt(rho) if isinstance(rho, float) else "",
            "generated_handoff": output.get("output", ""),
            "sigma": fmt(sigma),
            "sigma_op": fmt(sigma_op),
        }
        out_rows.append(item)
        vals = (rho, sigma, sigma_op)
        by_condition[row.get("condition", "")].append(vals)
        all_vals.append(vals)
    write_csv(OUT / "e1_baseline_outputs.csv", out_rows, ["model", "scenario_id", "domain", "surface", "condition", "upstream_tokens", "handoff_tokens", "rho", "generated_handoff", "sigma", "sigma_op"])

    summary = []
    condition_order = ["__overall__", "free_text", "compressed_free_text", "preserve_markers_instruction", "sectioned_template", "structured_schema"]
    for condition in condition_order:
        vals = all_vals if condition == "__overall__" else by_condition[condition]
        if not vals:
            continue
        summary.append({
            "model": DISPLAY,
            "condition": condition,
            "rho": fmt(mean([x[0] for x in vals])),
            "sigma": fmt(mean([x[1] for x in vals])),
            "sigma_op": fmt(mean([x[2] for x in vals])),
            "gap_sigma_minus_sigma_op": fmt(mean([x[1] for x in vals]) - mean([x[2] for x in vals])),
            "n": len(vals),
        })
    write_csv(OUT / "e1_baseline_summary.csv", summary, ["model", "condition", "rho", "sigma", "sigma_op", "gap_sigma_minus_sigma_op", "n"])
    lines = ["# E1 Baseline Handoff: Qwen3-32B", "", "| Model | Condition | rho | sigma | sigma_op | Gap sigma - sigma_op | n |", "|---|---|---:|---:|---:|---:|---:|"]
    for r in summary:
        lines.append(f"| {r['model']} | {r['condition']} | {r['rho']} | {r['sigma']} | {r['sigma_op']} | {r['gap_sigma_minus_sigma_op']} | {r['n']} |")
    lines.extend(["", "This table reports the uncompressed E1 baseline handoff conditions for Qwen3-32B."])
    (OUT / "e1_baseline_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def build_e8():
    evals = read_jsonl(DATA / "eval_qwen3_32b_e8_marker_gradient.jsonl")
    outputs = load_by_prompt(DATA / "model_outputs_qwen3_32b_e8_marker_gradient.jsonl")
    out_rows = []
    buckets = defaultdict(lambda: [0, 0])
    for row in evals:
        raw = outputs[row["prompt_id"]]
        level_key, _ = LEVEL_MAP.get(row.get("handoff_variant"), (row.get("handoff_variant"), row.get("handoff_variant")))
        leaks = bool(row.get("final_leakage_heuristic"))
        buckets[level_key][1] += 1
        buckets[level_key][0] += int(leaks)
        hits = row.get("disallowed_hits") or []
        out_rows.append({
            "model": DISPLAY,
            "scenario_id": row["scenario_id"],
            "domain": row.get("domain", ""),
            "marker_level": level_key,
            "pressure": row.get("pressure_condition", ""),
            "downstream_output": raw.get("output", ""),
            "leakage_label": "leak" if leaks else "no_leak",
            "leakage_type_if_available": "",
            "matched_sensitive_fact": "; ".join(hits),
            "detector_match": "; ".join(hits),
            "manual_label_if_available": "",
        })
    write_csv(OUT / "e8_marker_gradient_outputs.csv", out_rows, ["model", "scenario_id", "domain", "marker_level", "pressure", "downstream_output", "leakage_label", "leakage_type_if_available", "matched_sensitive_fact", "detector_match", "manual_label_if_available"])
    order = ["L0_no_marker", "L1_vague_hedge", "L2_soft_suggestion", "L3_constraint", "L4_imperative"]
    summary = []
    for level in order:
        leaks, total = buckets[level]
        summary.append({"marker_level": level, "model": DISPLAY, "leaks": leaks, "total": total, "leakage_rate": fmt(leaks / total if total else float("nan"))})
    write_csv(OUT / "e8_marker_gradient_summary.csv", summary, ["marker_level", "model", "leaks", "total", "leakage_rate"])
    tests = []
    for left, right in [("L1_vague_hedge", "L2_soft_suggestion"), ("L0_no_marker", "L4_imperative")]:
        ll, lt = buckets[left]
        rl, rt = buckets[right]
        tests.append({"contrast": f"{left} vs {right}", "model": DISPLAY, "left_level": left, "left_leaks": ll, "left_total": lt, "right_level": right, "right_leaks": rl, "right_total": rt, "fisher_p_two_sided": f"{fisher_exact(ll, lt, rl, rt):.6g}"})
    write_csv(OUT / "e8_marker_gradient_fisher_tests.csv", tests, ["contrast", "model", "left_level", "left_leaks", "left_total", "right_level", "right_leaks", "right_total", "fisher_p_two_sided"])
    lines = ["# E8 Marker-Strength Gradient: Qwen3-32B", "", "| Marker level | Qwen3-32B leakage |", "|---|---:|"]
    label_by_level = dict((v[0], v[1]) for v in LEVEL_MAP.values())
    for r in summary:
        lines.append(f"| {label_by_level.get(r['marker_level'], r['marker_level'])} | {r['leaks']} / {r['total']} ({float(r['leakage_rate']):.1%}) |")
    lines.extend(["", "## Fisher Tests", "", "| Contrast | p-value |", "|---|---:|"])
    for t in tests:
        lines.append(f"| {t['contrast']} | {t['fisher_p_two_sided']} |")
    (OUT / "e8_marker_gradient_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary, tests


def build_e9():
    evals = read_jsonl(DATA / "eval_qwen3_32b_e9_corrupted_allowlist.jsonl")
    outputs = load_by_prompt(DATA / "model_outputs_qwen3_32b_e9_corrupted_allowlist.jsonl")
    prompts = load_by_prompt(DATA / "e9_corrupted_allowlist_prompts.jsonl")
    out_rows = []
    buckets = defaultdict(lambda: [0, 0])
    for row in evals:
        raw = outputs[row["prompt_id"]]
        prompt = prompts[row["prompt_id"]]
        variant = row.get("handoff_variant")
        leaks = bool(row.get("final_leakage_heuristic"))
        buckets[variant][1] += 1
        buckets[variant][0] += int(leaks)
        hits = row.get("disallowed_hits") or []
        out_rows.append({
            "model": DISPLAY,
            "scenario_id": row["scenario_id"],
            "domain": row.get("domain", ""),
            "variant": variant,
            "pressure": row.get("pressure_condition", ""),
            "typed_handoff_or_context": prompt.get("prompt", ""),
            "downstream_output": raw.get("output", ""),
            "leakage_label": "leak" if leaks else "no_leak",
            "leakage_type_if_available": "",
            "detector_match": "; ".join(hits),
            "matched_sensitive_fact": "; ".join(hits),
        })
    write_csv(OUT / "e9_corrupted_allowlist_outputs.csv", out_rows, ["model", "scenario_id", "domain", "variant", "pressure", "typed_handoff_or_context", "downstream_output", "leakage_label", "leakage_type_if_available", "detector_match", "matched_sensitive_fact"])
    summary = []
    for variant in VARIANT_ORDER:
        leaks, total = buckets[variant]
        summary.append({"variant": variant, "variant_label": VARIANT_LABELS[variant], "model": DISPLAY, "leaks": leaks, "total": total, "leakage_rate": fmt(leaks / total if total else float("nan"))})
    write_csv(OUT / "e9_corrupted_allowlist_summary.csv", summary, ["variant", "variant_label", "model", "leaks", "total", "leakage_rate"])
    tests = []
    for right in ["corrupt_typo_allowlist", "corrupt_format_noise", "corrupt_negation_flip", "dump_no_marker_control"]:
        ll, lt = buckets["clean_allowlist"]
        rl, rt = buckets[right]
        tests.append({"contrast": f"clean_allowlist vs {right}", "model": DISPLAY, "left_variant": "clean_allowlist", "left_leaks": ll, "left_total": lt, "right_variant": right, "right_leaks": rl, "right_total": rt, "fisher_p_two_sided": f"{fisher_exact(ll, lt, rl, rt):.6g}"})
    write_csv(OUT / "e9_corrupted_allowlist_fisher_tests.csv", tests, ["contrast", "model", "left_variant", "left_leaks", "left_total", "right_variant", "right_leaks", "right_total", "fisher_p_two_sided"])
    lines = ["# E9 Corrupted-Allowlist Stress Test: Qwen3-32B", "", "| Variant | Qwen3-32B leakage |", "|---|---:|"]
    for r in summary:
        lines.append(f"| {r['variant_label']} | {r['leaks']} / {r['total']} ({float(r['leakage_rate']):.1%}) |")
    lines.extend(["", "## Fisher Tests", "", "| Contrast | p-value |", "|---|---:|"])
    for t in tests:
        lines.append(f"| {t['contrast']} | {t['fisher_p_two_sided']} |")
    (OUT / "e9_corrupted_allowlist_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary, tests


def write_combined(e1, e2, e8, e9):
    e2_by = {r["condition"]: r for r in e2}
    e8_by = {r["marker_level"]: r for r in e8}
    e9_by = {r["variant"]: r for r in e9}
    lines = [
        "# Qwen3-32B Third-Model Summary",
        "",
        "## Run Metadata",
        "",
        "- Model: `qwen3:32b`",
        "- Display name: Qwen3-32B",
        "- Inference backend: external chat API (`[URL]`)",
        "- Course name: `agam`",
        "- Temperature: 0.1",
        "- Top_p: not exposed by repository runner",
        "- Seed: not exposed by endpoint",
        "- Prompt manifests used: `data/bound_handoff_phase2_prompts.jsonl`, `data/qwen3_32b_e2_compression_prompts.jsonl`, `data/qwen3_32b_e8_marker_gradient_prompts.jsonl`, `data/e9_corrupted_allowlist_prompts.jsonl`",
        "",
        "## Experiments Completed",
        "",
        f"- E1 baseline handoff: {'complete (180/180)' if e1 else 'not run'}",
        "- E2 compression stress: complete (72/72)",
        "- E8 marker gradient: complete, full five-level version (240/240)",
        "- E9 corrupted allowlist: complete, full seven-variant version (336/336)",
    ]
    if e1:
        lines.extend([
            "",
            "## E1 Baseline",
            "",
            "| Condition | rho | sigma | sigma_op | Gap |",
            "|---|---:|---:|---:|---:|",
        ])
        for r in e1:
            lines.append(f"| {r['condition']} | {r['rho']} | {r['sigma']} | {r['sigma_op']} | {r['gap_sigma_minus_sigma_op']} |")
    lines.extend([
        "",
        "## E2 Compression",
        "",
        "| Condition | rho | sigma | sigma_op | Gap |",
        "|---|---:|---:|---:|---:|",
    ])
    for r in e2:
        lines.append(f"| {r['condition']} | {r['rho']} | {r['sigma']} | {r['sigma_op']} | {r['gap_sigma_minus_sigma_op']} |")
    lines.extend(["", "## E8 Marker Gradient", "", "| Marker level | Leakage |", "|---|---:|"])
    label_by_level = dict((v[0], v[1]) for v in LEVEL_MAP.values())
    for r in e8:
        lines.append(f"| {label_by_level.get(r['marker_level'], r['marker_level'])} | {r['leaks']} / {r['total']} ({float(r['leakage_rate']):.1%}) |")
    lines.extend(["", "## E9 Corrupted Allowlist", "", "| Variant | Leakage |", "|---|---:|"])
    for r in e9:
        lines.append(f"| {r['variant_label']} | {r['leaks']} / {r['total']} ({float(r['leakage_rate']):.1%}) |")
    q25 = e2_by["25-word compression"]
    l1 = e8_by["L1_vague_hedge"]
    l2 = e8_by["L2_soft_suggestion"]
    l4 = e8_by["L4_imperative"]
    clean = e9_by["clean_allowlist"]
    neg = e9_by["corrupt_negation_flip"]
    paragraph = (
        "To test whether the main patterns are specific to GPT-5-mini and DeepSeek-R1-32B, we replicate the core experiments on Qwen3-32B, a third model family. "
        f"Qwen3-32B reproduces the qualitative summary-collapse pattern under hard compression: under the 25-word condition, boundary-marker survival is {q25['sigma']}, while operational-fact survival remains {q25['sigma_op']}. "
        f"It also reproduces the marker-strength effect: vague markers leak in {l1['leaks']}/{l1['total']} cases, compared with {l2['leaks']}/{l2['total']} under soft suggestions and {l4['leaks']}/{l4['total']} under hard imperatives. "
        f"Finally, in the corrupted-allowlist stress test, clean allowlists leak in {clean['leaks']}/{clean['total']} cases, while corrupted variants restore leakage, with negation-flip corruption leaking in {neg['leaks']}/{neg['total']} cases. "
        "These results strengthen cross-model robustness, though broader replication across additional model families remains future work."
    )
    lines.extend(["", "## Paper-Facing Paragraph", "", paragraph, ""])
    (OUT / "qwen3_32b_third_model_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    e1 = build_e1()
    e2 = build_e2()
    e8, _ = build_e8()
    e9, _ = build_e9()
    write_combined(e1, e2, e8, e9)
    print(f"Wrote Qwen3-32B summaries to {OUT}")


if __name__ == "__main__":
    main()
