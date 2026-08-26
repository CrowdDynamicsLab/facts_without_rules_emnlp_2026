#!/usr/bin/env python3
"""Extract concise qualitative examples from existing result files."""

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


def scenario_lookup():
    data = json.loads((DATA / "bound_handoff_phase2.json").read_text(encoding="utf-8"))
    return {s["scenario_id"]: s for s in data["scenarios"]}


def by_prompt(path):
    return {r["prompt_id"]: r for r in read_jsonl(path)}


def truncate(text, n=420):
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[: n - 3] + "..."


def first(xs, default=""):
    if not xs:
        return default
    x = xs[0]
    return x.get("text", "") if isinstance(x, dict) else str(x)


def add_example(examples, kind, scenario, model, condition, handoff_excerpt, downstream_excerpt, why):
    examples.append(
        {
            "kind": kind,
            "scenario_id": scenario["scenario_id"],
            "domain": scenario["domain"],
            "model": model,
            "condition": condition,
            "upstream_operational_fact": first(scenario.get("operational_facts")) or first(scenario.get("gold_allowed_disclosure")),
            "upstream_boundary_marker": "; ".join(m.get("text", "") for m in scenario.get("boundary_markers", [])),
            "gold_sensitive_fact": "; ".join(scenario.get("gold_disallowed_disclosure", [])),
            "handoff_excerpt": truncate(handoff_excerpt),
            "downstream_output_excerpt": truncate(downstream_excerpt),
            "why_it_matters": why,
        }
    )


def build_examples():
    scenarios = scenario_lookup()
    examples = []

    # 1. Summary collapse under 25-word compression.
    stress_judge = read_jsonl(DATA / "judge_scores_openai_gpt5mini_phase2_stresstest_hard_budget_v2_full36_20260512.jsonl")
    stress_out = by_prompt(DATA / "model_outputs_openai_gpt5mini_phase2_stresstest_hard_budget_v2_full36_20260512.jsonl")
    for row in stress_judge:
        sigma = row["overall"]["boundary_sigma_numeric"]
        sigma_op = row["overall"]["operational_sigma_numeric"]
        if sigma is not None and sigma < 0.6 and sigma_op is not None and sigma_op > 0.9:
            out = stress_out.get(row["prompt_id"], {})
            add_example(
                examples,
                "Summary collapse under compression",
                scenarios[row["scenario_id"]],
                "gpt-5-mini",
                row.get("condition"),
                out.get("output", ""),
                "",
                "The operational fact survives a very tight handoff, while boundary metadata is weakened or omitted.",
            )
            break

    # 2. Weak-marker leakage.
    weak_eval = read_jsonl(DATA / "eval_openai_gpt5mini_phase2c_downstream_calibrated.jsonl")
    weak_out = by_prompt(DATA / "model_outputs_openai_gpt5mini_phase2c_downstream.jsonl")
    for row in weak_eval:
        if row.get("handoff_variant") == "dump_weak_marker" and row.get("final_leakage_heuristic"):
            out = weak_out.get(row["prompt_id"], {})
            add_example(
                examples,
                "Weak-marker leakage",
                scenarios[row["scenario_id"]],
                row.get("model"),
                row.get("condition"),
                "Weak marker condition carries sensitive context with generic caution language.",
                out.get("output", ""),
                "A vague caution marker is insufficient under downstream pressure.",
            )
            break

    # 3. Explicit-marker protection.
    strong_eval = read_jsonl(DATA / "eval_openai_gpt5mini_phase2b_downstream_calibrated.jsonl")
    strong_out = by_prompt(DATA / "model_outputs_openai_gpt5mini_phase2b_downstream.jsonl")
    for row in strong_eval:
        if row.get("handoff_variant") == "dump_with_marker" and not row.get("final_leakage_heuristic"):
            out = strong_out.get(row["prompt_id"], {})
            add_example(
                examples,
                "Explicit-marker protection",
                scenarios[row["scenario_id"]],
                row.get("model"),
                row.get("condition"),
                "Strong marker condition carries sensitive context with an explicit do-not-disclose boundary.",
                out.get("output", ""),
                "The sensitive fact is internally present, but an explicit audience-facing boundary prevents final leakage.",
            )
            break

    # 4. Phrase-redaction residual semantic leakage.
    for row in read_jsonl(DATA / "manual_audit_uiuc_deepseek_exp5b_highrisk.jsonl"):
        label = str(row.get("manual_final_leakage", row.get("human_final_leakage", row.get("manual_leakage", "")))).lower()
        if "yes" in label or "true" in label or row.get("manual_leakage") is True:
            add_example(
                examples,
                "Phrase-redaction residual semantic leakage",
                scenarios[row["scenario_id"]],
                "deepseek-r1:32b",
                row.get("condition") or row.get("mitigation_condition"),
                row.get("handoff_output", "Redacted handoff output not stored in this audit row."),
                row.get("output", row.get("output_excerpt", "")),
                "Exact phrase redaction can remove the literal phrase while preserving enough semantic/category information to leak.",
            )
            break

    # 5. Typed allowlist success.
    typed_eval = read_jsonl(DATA / "eval_openai_gpt5mini_operational_lifting_calibrated.jsonl")
    typed_out = by_prompt(DATA / "model_outputs_openai_gpt5mini_operational_lifting.jsonl")
    for row in typed_eval:
        if not row.get("final_leakage_heuristic"):
            out = typed_out.get(row["prompt_id"], {})
            add_example(
                examples,
                "Typed allowlist success",
                scenarios[row["scenario_id"]],
                row.get("model"),
                row.get("condition"),
                "Typed schema gives operational facts a downstream audience allowlist and sensitive facts an empty allowlist.",
                out.get("output", ""),
                "The downstream output uses task-relevant operational information while omitting the sensitive fact.",
            )
            break

    return examples


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def write_md(path, examples):
    lines = ["# Qualitative Examples", ""]
    for ex in examples:
        lines.extend(
            [
                f"### Example: {ex['kind']}",
                "",
                f"- Scenario/domain: `{ex['scenario_id']}` / {ex['domain']}",
                f"- Model/condition: `{ex['model']}` / `{ex['condition']}`",
                f"- Upstream operational fact: {ex['upstream_operational_fact']}",
                f"- Upstream boundary marker: {ex['upstream_boundary_marker']}",
                f"- Gold sensitive fact: {ex['gold_sensitive_fact']}",
                f"- Generated handoff excerpt: {ex['handoff_excerpt']}",
                f"- Downstream output excerpt: {ex['downstream_output_excerpt']}",
                f"- Why it matters: {ex['why_it_matters']}",
                "",
            ]
        )
    lines.append("No corrupted-allowlist behavior example is included because the corrupted-allowlist run was not completed in the existing result set.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    examples = build_examples()
    OUT.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT / "qualitative_examples.jsonl", examples)
    write_md(OUT / "qualitative_examples.md", examples)
    print(f"Wrote {OUT / 'qualitative_examples.jsonl'}")
    print(f"Wrote {OUT / 'qualitative_examples.md'}")


if __name__ == "__main__":
    main()
