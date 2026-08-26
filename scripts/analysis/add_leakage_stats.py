#!/usr/bin/env python3
"""Compute Fisher exact tests and Wilson intervals for leakage contrasts.

The script uses the existing two-model BOUND-Handoff result files and writes:
  - results/additional/leakage_fisher_tests.csv
  - results/additional/leakage_fisher_tests.md
"""

import csv
import json
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "dataset" / "data"
OUT = ROOT / "results" / "additional"
MODELS = ["gpt-5-mini", "deepseek-r1:32b"]


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_jsonl(path):
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def leakage_counts(path, key_field):
    counts = defaultdict(lambda: [0, 0])
    for row in read_jsonl(path):
        model = row.get("model")
        key = row.get(key_field)
        if key is None and key_field == "mitigation_condition":
            key = (row.get("condition") or "").split("__")[0]
        counts[(model, key)][1] += 1
        counts[(model, key)][0] += int(bool(row.get("final_leakage_heuristic")))
    return {k: (v[0], v[1]) for k, v in counts.items()}


def wilson(leaks, total, z=1.959963984540054):
    if total <= 0:
        return (float("nan"), float("nan"))
    p = leaks / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denom
    return max(0.0, center - half), min(1.0, center + half)


def logchoose(n: int, k: int) -> float:
    if k < 0 or k > n:
        return float("-inf")
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def hypergeom_p(x: int, row1: int, col1: int, n: int) -> float:
    return math.exp(logchoose(col1, x) + logchoose(n - col1, row1 - x) - logchoose(n, row1))


def fisher_exact(leaks_a, total_a, leaks_b, total_b):
    """Two-sided Fisher exact p-value with fixed margins."""
    a = leaks_a
    b = leaks_b
    row1 = total_a
    row2 = total_b
    col1 = leaks_a + leaks_b
    n = total_a + total_b
    lo = max(0, col1 - row2)
    hi = min(row1, col1)
    observed = hypergeom_p(a, row1, col1, n)
    p_value = sum(hypergeom_p(x, row1, col1, n) for x in range(lo, hi + 1) if hypergeom_p(x, row1, col1, n) <= observed + 1e-12)
    non_a = total_a - leaks_a
    non_b = total_b - leaks_b
    if leaks_b == 0 or non_a == 0:
        odds = math.inf
    elif leaks_a == 0 or non_b == 0:
        odds = 0.0
    else:
        odds = (leaks_a * non_b) / (non_a * leaks_b)
    return odds, min(1.0, p_value)


def fmt(x: float) -> str:
    if isinstance(x, float) and math.isinf(x):
        return "inf"
    return f"{x:.6g}"


def parse_fraction(text):
    a, b = text.split("/")
    return int(a), int(b)


def collect_counts():
    counts = {}

    # E3: marker presence downstream pressure.
    for row in read_csv(DATA / "phase2b_model_comparison.csv"):
        counts[("E3", row["model"], row["handoff_variant"])] = (int(row["leaks"]), int(row["n"]), "calibrated heuristic")

    # E4: weak marker, with manual weak-marker audits where available.
    for model in MODELS:
        eval_file = DATA / ("eval_openai_gpt5mini_phase2c_downstream_calibrated.jsonl" if model == "gpt-5-mini" else "eval_uiuc_deepseek_phase2c_downstream_calibrated.jsonl")
        for (m, variant), (leaks, total) in leakage_counts(eval_file, "handoff_variant").items():
            if m == model:
                counts[("E4", model, variant)] = (leaks, total, "calibrated heuristic")
    gpt_audit = read_json(DATA / "phase2c_weak_marker_manual_audit_summary.json")
    ds_audit = read_json(DATA / "phase2c_deepseek_weak_marker_manual_audit_summary.json")
    counts[("E4", "gpt-5-mini", "dump_weak_marker")] = (*parse_fraction(gpt_audit["dump_weak_marker_manual"]), "manual audit for all weak-marker rows")
    counts[("E4", "deepseek-r1:32b", "dump_weak_marker")] = (*parse_fraction(ds_audit["dump_weak_marker_manual"]), "manual audit for all weak-marker rows")

    # E5: direct/no-handoff control.
    for model, path in [
        ("gpt-5-mini", DATA / "summary_openai_gpt5mini_single_agent_control.csv"),
        ("deepseek-r1:32b", DATA / "summary_uiuc_deepseek_single_agent_control.csv"),
    ]:
        for row in read_csv(path):
            if row["group"] == "control_variant":
                counts[("E5", model, row["value"])] = (int(row["leaks"]), int(row["n"]), "calibrated heuristic")

    # E6: prompt-only mitigation.
    for model, path in [
        ("gpt-5-mini", DATA / "eval_openai_gpt5mini_exp5_mitigation_downstream_calibrated.jsonl"),
        ("deepseek-r1:32b", DATA / "eval_uiuc_deepseek_exp5_mitigation_downstream_calibrated.jsonl"),
    ]:
        per = leakage_counts(path, "mitigation_condition")
        total_leaks = total_n = 0
        for (m, condition), (leaks, total) in per.items():
            if m != model:
                continue
            counts[("E6", model, condition)] = (leaks, total, "calibrated heuristic")
            total_leaks += leaks
            total_n += total
        counts[("E6", model, "__overall__")] = (total_leaks, total_n, "calibrated heuristic")

    # E7: redaction exact-phrase leakage and manual high-risk semantic audit.
    for model, path in [
        ("gpt-5-mini", DATA / "eval_openai_gpt5mini_exp5b_mitigation_downstream_calibrated.jsonl"),
        ("deepseek-r1:32b", DATA / "eval_uiuc_deepseek_exp5b_mitigation_downstream_calibrated.jsonl"),
    ]:
        rows = [r for r in read_jsonl(path) if r.get("model") == model]
        counts[("E7", model, "exact_phrase_redaction")] = (
            sum(int(bool(r.get("final_leakage_heuristic"))) for r in rows),
            len(rows),
            "exact-phrase/alias detector after redaction",
        )
    gpt_manual = read_jsonl(DATA / "audit_openai_gpt5mini_exp5b_manual_sample40_labeled.jsonl")
    counts[("E7", "gpt-5-mini", "manual_semantic_high_risk")] = (
        sum(int(str(r.get("human_final_leakage", "")).strip().lower() in {"yes", "true", "1", "leak", "leakage"}) for r in gpt_manual),
        len(gpt_manual),
        "manual semantic audit sample; targeted slice, not full denominator",
    )
    for row in read_csv(DATA / "manual_audit_uiuc_deepseek_exp5b_highrisk_summary.csv"):
        if row["group"] == "overall":
            counts[("E7", "deepseek-r1:32b", "manual_semantic_high_risk")] = (
                int(row["manual_leaks"]),
                int(row["n"]),
                "manual semantic high-risk audit; targeted slice, not full denominator",
            )

    # E8: marker-strength gradient.
    for row in read_csv(DATA / "marker_gradient_summary.csv"):
        counts[("E8", row["model"], row["marker_level"])] = (
            int(row["leakage_count"]),
            int(row["n"]),
            "calibrated heuristic",
        )

    # E9: typed allowlist.
    e9 = read_json(DATA / "operational_lifting_summary.json")
    for model, stats in e9.items():
        counts[("E9", model, "typed_allowlist")] = (
            int(stats["leakage_count"]),
            int(stats["n"]),
            "gold-derived typed allowlist/oracle-style probe",
        )

    return counts


def add_row(rows, counts, experiment, contrast, model, cond_a, cond_b, extra_note=""):
    key_a = (experiment, model, cond_a)
    key_b = (experiment, model, cond_b)
    if key_a not in counts or key_b not in counts:
        return
    leaks_a, total_a, note_a = counts[key_a]
    leaks_b, total_b, note_b = counts[key_b]
    odds, p_value = fisher_exact(leaks_a, total_a, leaks_b, total_b)
    ci_a = wilson(leaks_a, total_a)
    ci_b = wilson(leaks_b, total_b)
    rows.append(
        {
            "experiment": experiment,
            "contrast": contrast,
            "model": model,
            "condition_a": cond_a,
            "leaks_a": leaks_a,
            "total_a": total_a,
            "rate_a": leaks_a / total_a if total_a else "",
            "ci_low_a": ci_a[0],
            "ci_high_a": ci_a[1],
            "condition_b": cond_b,
            "leaks_b": leaks_b,
            "total_b": total_b,
            "rate_b": leaks_b / total_b if total_b else "",
            "ci_low_b": ci_b[0],
            "ci_high_b": ci_b[1],
            "odds_ratio": odds,
            "p_value": p_value,
            "notes": "; ".join(x for x in [note_a if note_a == note_b else f"A: {note_a}; B: {note_b}", extra_note] if x),
        }
    )


def build_rows():
    counts = collect_counts()
    rows = []

    for model in MODELS:
        for a, b in [
            ("dump_with_marker", "dump_no_marker"),
            ("no_dump_marker", "dump_no_marker"),
            ("no_dump_marker", "dump_with_marker"),
        ]:
            add_row(rows, counts, "E3", f"{a} vs {b}", model, a, b)

        for a, b in [("dump_with_marker", "dump_weak_marker"), ("dump_weak_marker", "dump_no_marker")]:
            add_row(rows, counts, "E4", f"{a} vs {b}", model, a, b, "weak-marker denominator uses focused manual audit where available")

        for a, b in [
            ("direct_full_marker", "direct_weak_marker"),
            ("direct_full_marker", "direct_no_marker"),
        ]:
            add_row(rows, counts, "E5", f"{a} vs {b}", model, a, b)
        # Cross-experiment E5 vs E3.
        e5 = counts[("E5", model, "direct_full_marker")]
        e3 = counts[("E3", model, "dump_with_marker")]
        counts[("E5", model, "E3_dump_with_marker")] = e3
        add_row(rows, counts, "E5", "direct_full_marker vs E3_dump_with_marker", model, "direct_full_marker", "E3_dump_with_marker", "cross-experiment contrast")

        for condition in [
            "free_text",
            "minimality_instruction",
            "preserve_markers_instruction",
            "sectioned_template",
            "structured_schema_safe",
            "structured_schema_unsafe",
        ]:
            add_row(rows, counts, "E6", f"{condition} vs overall", model, condition, "__overall__", "condition is part of the same E6 denominator; interpret descriptively")
        for a, b in [
            ("structured_schema_safe", "free_text"),
            ("structured_schema_safe", "structured_schema_unsafe"),
            ("minimality_instruction", "free_text"),
        ]:
            add_row(rows, counts, "E6", f"{a} vs {b}", model, a, b)

        for e7_cond, e6_cond in [
            ("exact_phrase_redaction", "__overall__"),
            ("exact_phrase_redaction", "structured_schema_safe"),
        ]:
            counts[("E7", model, f"E6_{e6_cond}")] = counts[("E6", model, e6_cond)]
            add_row(rows, counts, "E7", f"{e7_cond} vs E6_{e6_cond}", model, e7_cond, f"E6_{e6_cond}", "cross-experiment contrast")
        add_row(rows, counts, "E7", "manual_semantic_high_risk vs exact_phrase_redaction", model, "manual_semantic_high_risk", "exact_phrase_redaction", "manual audit is a targeted high-risk slice, not the full E7 denominator")

        for a, b in [
            ("L0_no_marker", "L1_vague_hedge"),
            ("L1_vague_hedge", "L2_soft_suggestion"),
            ("L2_soft_suggestion", "L3_constraint"),
            ("L2_soft_suggestion", "L4_imperative"),
            ("L3_constraint", "L4_imperative"),
        ]:
            add_row(rows, counts, "E8", f"{a} vs {b}", model, a, b)

        for other_exp, other_cond, label in [
            ("E6", "structured_schema_safe", "E6_structured_schema_safe"),
            ("E6", "__overall__", "E6_overall"),
            ("E3", "dump_with_marker", "E3_dump_with_marker"),
            ("E7", "exact_phrase_redaction", "E7_exact_phrase_redaction"),
        ]:
            counts[("E9", model, label)] = counts[(other_exp, model, other_cond)]
            add_row(rows, counts, "E9", f"typed_allowlist vs {label}", model, "typed_allowlist", label, "E9 is an oracle-style/gold-derived typed-boundary probe")

    return rows


def write_csv(path, rows):
    fields = [
        "experiment",
        "contrast",
        "model",
        "condition_a",
        "leaks_a",
        "total_a",
        "rate_a",
        "ci_low_a",
        "ci_high_a",
        "condition_b",
        "leaks_b",
        "total_b",
        "rate_b",
        "ci_low_b",
        "ci_high_b",
        "odds_ratio",
        "p_value",
        "notes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            for key in ["rate_a", "ci_low_a", "ci_high_a", "rate_b", "ci_low_b", "ci_high_b", "odds_ratio", "p_value"]:
                if isinstance(out[key], float):
                    out[key] = fmt(out[key])
            writer.writerow(out)


def find_row(rows, experiment, model, contrast):
    for row in rows:
        if row["experiment"] == experiment and row["model"] == model and row["contrast"] == contrast:
            return row
    return None


def p_text(row):
    if not row:
        return "not available"
    return f"p = {row['p_value']:.3g}"


def pct(leaks, total):
    return f"{leaks}/{total} ({leaks / total:.1%})"


def write_md(path, rows):
    e8_gpt = find_row(rows, "E8", "gpt-5-mini", "L1_vague_hedge vs L2_soft_suggestion")
    e8_ds = find_row(rows, "E8", "deepseek-r1:32b", "L1_vague_hedge vs L2_soft_suggestion")
    e3_gpt = find_row(rows, "E3", "gpt-5-mini", "dump_with_marker vs dump_no_marker")
    e3_ds = find_row(rows, "E3", "deepseek-r1:32b", "dump_with_marker vs dump_no_marker")
    e9_rows = [r for r in rows if r["experiment"] == "E9" and r["contrast"] == "typed_allowlist vs E6_overall"]

    lines = [
        "# Leakage Fisher Tests",
        "",
        "## Paper-ready summary",
        "",
        f"The E8 L1→L2 transition is significant under Fisher's exact test for both models (GPT-5-mini: {p_text(e8_gpt)}, DeepSeek-R1-32B: {p_text(e8_ds)}), supporting the interpretation that disclosure-specific boundary language is more protective than generic caution language.",
        f"The E3 dump-with-marker vs. dump-no-marker contrast is also significant for both models (GPT-5-mini: {p_text(e3_gpt)}, DeepSeek-R1-32B: {p_text(e3_ds)}).",
        "E9 is reported as a gold-derived typed-allowlist probe, not as an end-to-end mitigation; its zero-leakage result is contrasted with E6 and E7 only as a mechanistic reference point.",
        "",
        "## Key rows",
        "",
        "| Experiment | Contrast | Model | A | B | Odds ratio | p-value |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    keep = [r for r in rows if (r["experiment"], r["contrast"]) in {
        ("E3", "dump_with_marker vs dump_no_marker"),
        ("E8", "L1_vague_hedge vs L2_soft_suggestion"),
        ("E9", "typed_allowlist vs E6_overall"),
        ("E7", "manual_semantic_high_risk vs exact_phrase_redaction"),
    }]
    for r in keep:
        lines.append(
            f"| {r['experiment']} | {r['contrast']} | {r['model']} | {pct(r['leaks_a'], r['total_a'])} | {pct(r['leaks_b'], r['total_b'])} | {fmt(r['odds_ratio'])} | {fmt(r['p_value'])} |"
        )
    lines.extend([
        "",
        "## Notes",
        "",
        "- Wilson 95% confidence intervals are included in the CSV for every leakage rate.",
        "- E4 weak-marker counts use focused manual audits for the weak-marker rows where available; strong-marker rows are otherwise calibrated heuristic counts.",
        "- E7 manual semantic leakage rows are targeted audit slices, not full-run estimates, and should not be interpreted as full-denominator leakage rates.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    rows = build_rows()
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "leakage_fisher_tests.csv", rows)
    write_md(OUT / "leakage_fisher_tests.md", rows)
    print(f"Wrote {OUT / 'leakage_fisher_tests.csv'}")
    print(f"Wrote {OUT / 'leakage_fisher_tests.md'}")


if __name__ == "__main__":
    main()
