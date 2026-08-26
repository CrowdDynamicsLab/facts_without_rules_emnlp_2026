"""Compute ordinal label proportions for sigma and sigma_op judges.
Explodes marker and operational fact labels to item-level rows.
Writes descriptive CSVs and TeX tables for the paper appendix.
Runs sanity checks against existing numeric sigma summaries.
"""

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "dataset" / "data"
OUT_DIR = DATA_DIR / "ordinal"
TEX_DIR = ROOT / "arr_memory_mas" / "tables"

LABELS = ["preserved", "paraphrased", "weakened", "absent"]
LABEL_TO_SCORE = {
    "preserved": 1.0,
    "paraphrased": 0.75,
    "weakened": 0.35,
    "absent": 0.0,
}
E1_CONDITIONS = [
    "free_text",
    "compressed_free_text",
    "preserve_markers_instruction",
    "sectioned_template",
    "structured_schema",
]
E2_CONDITIONS = ["hard_budget_compressed", "hard_budget_compressed_v2"]
DISPLAY_MODELS = {
    "gpt5mini": "GPT-5-mini",
    "deepseek": "DS-R1-32B",
    "qwen3_32b": "Qwen3-32B",
}
MODEL_ORDER = ["GPT-5-mini", "DS-R1-32B", "Qwen3-32B"]
CONDITION_ORDER = E1_CONDITIONS + E2_CONDITIONS

INPUT_FILES = [
    (
        "GPT-5-mini",
        DATA_DIR / "judge_scores_openai_gpt5mini_phase2_sigma.jsonl",
    ),
    (
        "GPT-5-mini",
        DATA_DIR
        / "judge_scores_openai_gpt5mini_phase2_stresstest_hard_budget_full36_20260512.jsonl",
    ),
    (
        "GPT-5-mini",
        DATA_DIR
        / "judge_scores_openai_gpt5mini_phase2_stresstest_hard_budget_v2_full36_20260512.jsonl",
    ),
    (
        "DS-R1-32B",
        DATA_DIR / "judge_scores_uiuc_deepseek_phase2_sigma.jsonl",
    ),
    (
        "DS-R1-32B",
        DATA_DIR
        / "judge_scores_uiuc_deepseek_phase2_stresstest_hard_budget_full36_20260512.jsonl",
    ),
    (
        "DS-R1-32B",
        DATA_DIR
        / "judge_scores_uiuc_deepseek_phase2_stresstest_hard_budget_v2_full36_20260512.jsonl",
    ),
    (
        "Qwen3-32B",
        DATA_DIR / "judge_scores_qwen3_32b_phase2_sigma.jsonl",
    ),
    (
        "Qwen3-32B",
        DATA_DIR / "judge_scores_qwen3_32b_e2_compression_sigma.jsonl",
    ),
]

REFERENCE_SUMMARIES = [
    ("GPT-5-mini", DATA_DIR / "summary_openai_gpt5mini_phase2_sigma_by_condition.csv"),
    (
        "GPT-5-mini",
        DATA_DIR
        / "summary_openai_gpt5mini_phase2_stresstest_hard_budget_full36_sigma_by_condition_20260512.csv",
    ),
    (
        "GPT-5-mini",
        DATA_DIR
        / "summary_openai_gpt5mini_phase2_stresstest_hard_budget_v2_full36_sigma_by_condition_20260512.csv",
    ),
    ("DS-R1-32B", DATA_DIR / "summary_uiuc_deepseek_phase2_sigma_by_condition.csv"),
    (
        "DS-R1-32B",
        DATA_DIR
        / "summary_uiuc_deepseek_phase2_stresstest_hard_budget_full36_sigma_by_condition_20260512.csv",
    ),
    (
        "DS-R1-32B",
        DATA_DIR
        / "summary_uiuc_deepseek_phase2_stresstest_hard_budget_v2_full36_sigma_by_condition_20260512.csv",
    ),
    ("Qwen3-32B", DATA_DIR / "summary_qwen3_32b_phase2_sigma_by_condition.csv"),
    ("Qwen3-32B", DATA_DIR / "summary_qwen3_32b_e2_compression_sigma_by_condition.csv"),
]

PAPER_SUMMARY_FILES = [
    ROOT / "results" / "e1_baseline_summary.csv",
    ROOT / "results" / "qwen3_32b" / "e2_compression_stress_summary.csv",
]
PAPER_CONDITION_ALIASES = {
    "40-word compression": "hard_budget_compressed",
    "25-word compression": "hard_budget_compressed_v2",
}


def load_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path} line {line_no}: {exc}") from exc
    return rows


def condition_sort_key(condition):
    if condition in CONDITION_ORDER:
        return (CONDITION_ORDER.index(condition), condition)
    return (len(CONDITION_ORDER), condition)


def model_sort_key(model):
    if model in MODEL_ORDER:
        return (MODEL_ORDER.index(model), model)
    return (len(MODEL_ORDER), model)


def assert_expected_handoffs(handoff_df):
    counts = (
        handoff_df.groupby(["model", "condition"])["prompt_id"]
        .nunique()
        .reset_index(name="n_handoffs")
        .sort_values(
            by=["model", "condition"],
            key=lambda col: col.map(
                model_sort_key if col.name == "model" else condition_sort_key
            ),
        )
    )
    print("\nCell-count sanity table:")
    print(counts.to_string(index=False))

    bad_rows = counts[(counts["condition"].isin(E1_CONDITIONS + E2_CONDITIONS)) & (counts["n_handoffs"].sub(36).abs() > 2)]
    if not bad_rows.empty:
        raise SystemExit(
            "Unexpected handoff counts; expected 36 +/- 2 per model/condition:\n"
            + bad_rows.to_string(index=False)
        )


def explode_rows():
    handoff_rows = []
    item_rows = []

    for model, path in INPUT_FILES:
        if not path.exists():
            raise FileNotFoundError(path)
        records = load_jsonl(path)
        conditions = sorted({row.get("condition", "") for row in records}, key=condition_sort_key)
        print(
            f"Loaded {path.relative_to(ROOT)}: {len(records)} handoffs, "
            f"conditions = {conditions}"
        )

        for row in records:
            condition = row.get("condition")
            scenario_id = row.get("scenario_id", "")
            prompt_id = row.get("prompt_id", "")
            handoff_rows.append(
                {
                    "model": model,
                    "condition": condition,
                    "scenario_id": scenario_id,
                    "prompt_id": prompt_id,
                }
            )

            for item in row.get("boundary_marker_scores", []):
                label = item.get("label")
                if label not in LABELS:
                    raise ValueError(f"Unexpected marker label {label!r} in {path}")
                item_rows.append(
                    {
                        "model": model,
                        "condition": condition,
                        "scenario_id": scenario_id,
                        "prompt_id": prompt_id,
                        "item_kind": "marker",
                        "item_id": item.get("marker_id", ""),
                        "category": item.get("category", ""),
                        "label": label,
                    }
                )

            for item in row.get("operational_fact_scores", []):
                label = item.get("label")
                if label not in LABELS:
                    raise ValueError(f"Unexpected operational label {label!r} in {path}")
                item_rows.append(
                    {
                        "model": model,
                        "condition": condition,
                        "scenario_id": scenario_id,
                        "prompt_id": prompt_id,
                        "item_kind": "operational",
                        "item_id": item.get("fact_id", ""),
                        "category": "",
                        "label": label,
                    }
                )

    handoff_df = pd.DataFrame(handoff_rows)
    item_df = pd.DataFrame(item_rows)
    return handoff_df, item_df


def proportions_for(group_cols, item_df):
    counts = (
        item_df.groupby(group_cols + ["label"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=LABELS, fill_value=0)
    )
    totals = counts.sum(axis=1)
    props = counts.div(totals, axis=0)

    out = pd.DataFrame(index=counts.index)
    out["n_items"] = totals
    for label in LABELS:
        out[f"p_{label}"] = props[label]
    out = out.reset_index()
    return sort_output(out)


def sort_output(df):
    sort_cols = [col for col in ["model", "condition", "item_kind"] if col in df.columns]
    if not sort_cols:
        return df
    return df.sort_values(
        by=sort_cols,
        key=lambda col: col.map(
            model_sort_key if col.name == "model" else condition_sort_key
        )
        if col.name in {"model", "condition"}
        else col,
    ).reset_index(drop=True)


def rounded_for_csv(df):
    out = df.copy()
    for col in [f"p_{label}" for label in LABELS]:
        out[col] = out[col].round(3)
    return out


def verify_proportion_sums(df, label):
    prop_cols = [f"p_{name}" for name in LABELS]
    sums = df[prop_cols].sum(axis=1)
    bad = df[(sums - 1.0).abs() > 1e-6]
    if not bad.empty:
        raise SystemExit(f"Proportion-sum check failed for {label}:\n{bad.to_string(index=False)}")
    print(f"PASS proportion sums: {label}")


def numeric_by_cell(item_df):
    scored = item_df.copy()
    scored["score"] = scored["label"].map(LABEL_TO_SCORE)
    return (
        scored.groupby(["model", "condition", "prompt_id", "item_kind"])["score"]
        .mean()
        .reset_index(name="handoff_score")
        .groupby(["model", "condition", "item_kind"])["handoff_score"]
        .mean()
        .reset_index(name="mean_score")
    )


def load_reference_values():
    refs = {}
    for model, path in REFERENCE_SUMMARIES:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for _, row in df[df["condition"] != "__overall__"].iterrows():
            condition = row["condition"]
            refs[(model, condition, "marker")] = (float(row["mean_sigma"]), path)
            refs[(model, condition, "operational")] = (float(row["mean_sigma_op"]), path)
    return refs


def load_paper_values():
    refs = {}
    for path in PAPER_SUMMARY_FILES:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df = df[df["condition"] != "__overall__"].copy()
        df["condition"] = df["condition"].replace(PAPER_CONDITION_ALIASES)
        for _, row in df.iterrows():
            model = row["model"]
            condition = row["condition"]
            refs[(model, condition, "marker")] = (float(row["sigma"]), path)
            refs[(model, condition, "operational")] = (float(row["sigma_op"]), path)
    return refs


def verify_numeric_sigma(item_df):
    observed = numeric_by_cell(item_df)
    summary_refs = load_reference_values()
    paper_refs = load_paper_values()
    failures = []

    for _, row in observed.iterrows():
        key = (row["model"], row["condition"], row["item_kind"])
        obs = float(row["mean_score"])
        if key in summary_refs:
            ref, path = summary_refs[key]
            if abs(obs - ref) > 0.005:
                failures.append((key, obs, ref, path))
        if key in paper_refs:
            ref, path = paper_refs[key]
            if abs(obs - ref) > 0.005:
                failures.append((key, obs, ref, path))

    if failures:
        lines = [
            f"{model} {condition} {kind}: observed={obs:.6f} ref={ref:.6f} source={path.relative_to(ROOT)}"
            for (model, condition, kind), obs, ref, path in failures
        ]
        raise SystemExit("FAIL numeric sigma cross-check:\n" + "\n".join(lines))
    print("PASS numeric sigma cross-check against existing per-handoff summary CSVs")


def verify_selective_fragility(by_condition):
    primary_models = ["GPT-5-mini", "DS-R1-32B"]
    condition = "hard_budget_compressed_v2"
    pivot = by_condition[
        (by_condition["model"].isin(primary_models))
        & (by_condition["condition"] == condition)
    ].pivot(index="model", columns="item_kind", values="p_preserved")
    bad = []
    for model in primary_models:
        gap = float(pivot.loc[model, "operational"] - pivot.loc[model, "marker"])
        if gap < 0.30:
            bad.append((model, gap))
    if bad:
        raise SystemExit(
            "FAIL ordinal selective-fragility check: "
            + ", ".join(f"{model} gap={gap:.3f}" for model, gap in bad)
        )
    print("PASS ordinal selective-fragility check for primary-model 25-word cells")


def fmt_prop(value):
    return f"{value:.2f}"


def latex_escape(value):
    return (
        value.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("&", "\\&")
        .replace("%", "\\%")
    )


def row_values(by_condition, model, condition):
    rows = {}
    for kind in ["marker", "operational"]:
        match = by_condition[
            (by_condition["model"] == model)
            & (by_condition["condition"] == condition)
            & (by_condition["item_kind"] == kind)
        ]
        if len(match) != 1:
            raise ValueError(f"Expected one row for {model}/{condition}/{kind}, found {len(match)}")
        rows[kind] = match.iloc[0]
    return rows


def tex_number_cells(row):
    return " & ".join(fmt_prop(float(row[f"p_{label}"])) for label in LABELS)


def build_25_word_tex(by_condition):
    marker_n = int(
        by_condition[
            (by_condition["condition"] == "hard_budget_compressed_v2")
            & (by_condition["item_kind"] == "marker")
        ]["n_items"].iloc[0]
    )
    op_n = int(
        by_condition[
            (by_condition["condition"] == "hard_budget_compressed_v2")
            & (by_condition["item_kind"] == "operational")
        ]["n_items"].iloc[0]
    )
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{@{}lcccccccc@{}}",
        "\\toprule",
        f"& \\multicolumn{{4}}{{c}}{{markers ($n={marker_n}$)}} "
        f"& \\multicolumn{{4}}{{c}}{{operational ($n={op_n}$)}} \\\\",
        "\\cmidrule(lr){2-5}\\cmidrule(l){6-9}",
        "\\textbf{Model} & pres & para & weak & abs & pres & para & weak & abs \\\\",
        "\\midrule",
    ]
    for model in MODEL_ORDER:
        rows = row_values(by_condition, model, "hard_budget_compressed_v2")
        marker = rows["marker"]
        operational = rows["operational"]
        lines.append(
            f"{model} "
            f"& {tex_number_cells(marker)} "
            f"& {tex_number_cells(operational)} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            (
                "\\caption{Ordinal label proportions under 25-word compression. "
                "Reported directly from the four-level judge rubric without the numeric "
                "mapping in Eq.~\\ref{eq:judge_rubric}. "
                "The marker $\\to$ operational shift in mass toward the \\emph{preserved} "
                "bucket is the same selective-fragility signature that the "
                "$\\sigma$-$\\sigma_\\mathrm{op}$ gap reports in the body.}"
            ),
            "\\label{tab:ordinal_label_proportions_25word}",
            "\\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def build_baseline_tex(by_condition):
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{@{}llcccccccc@{}}",
        "\\toprule",
        "&& \\multicolumn{4}{c}{markers ($n=72$)} & "
        "\\multicolumn{4}{c}{operational ($n=37$)} \\\\",
        "\\cmidrule(lr){3-6}\\cmidrule(l){7-10}",
        "\\textbf{Model} & \\textbf{Condition} & pres & para & weak & abs & pres & para & weak & abs \\\\",
        "\\midrule",
    ]
    for model_index, model in enumerate(MODEL_ORDER):
        if model_index:
            lines.append("\\addlinespace")
        for condition in E1_CONDITIONS:
            rows = row_values(by_condition, model, condition)
            marker = rows["marker"]
            operational = rows["operational"]
            lines.append(
                f"{model} & \\texttt{{{latex_escape(condition)}}} "
                f"& {tex_number_cells(marker)} "
                f"& {tex_number_cells(operational)} \\\\"
            )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            (
                "\\caption{E1 baseline ordinal label proportions by condition. "
                "Proportions are reported directly from the four-level judge rubric "
                "without collapsing labels into binary outcomes.}"
            ),
            "\\label{tab:ordinal_label_proportions_baseline}",
            "\\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def print_condition_summaries(by_condition):
    for model in MODEL_ORDER:
        model_rows = by_condition[by_condition["model"] == model]
        for condition in sorted(model_rows["condition"].unique(), key=condition_sort_key):
            for item_kind, display in [("marker", "marker:"), ("operational", "op:    ")]:
                row = model_rows[
                    (model_rows["condition"] == condition)
                    & (model_rows["item_kind"] == item_kind)
                ].iloc[0]
                print(
                    f"{model:<11}  {condition:<36}  {display}  "
                    f"pres={row['p_preserved']:.2f} "
                    f"para={row['p_paraphrased']:.2f} "
                    f"weak={row['p_weakened']:.2f} "
                    f"abs={row['p_absent']:.2f}  "
                    f"(n={int(row['n_items'])})"
                )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TEX_DIR.mkdir(parents=True, exist_ok=True)

    handoff_df, item_df = explode_rows()
    assert_expected_handoffs(handoff_df)

    by_condition = proportions_for(["model", "condition", "item_kind"], item_df)
    overall = proportions_for(["model", "item_kind"], item_df)

    verify_proportion_sums(by_condition, "by condition")
    verify_proportion_sums(overall, "overall")
    verify_numeric_sigma(item_df)
    verify_selective_fragility(by_condition)

    rounded_for_csv(by_condition).to_csv(
        OUT_DIR / "ordinal_label_proportions_by_condition.csv",
        index=False,
        float_format="%.3f",
    )
    rounded_for_csv(overall).to_csv(
        OUT_DIR / "ordinal_label_proportions_overall.csv",
        index=False,
        float_format="%.3f",
    )
    sort_output(item_df).to_csv(OUT_DIR / "ordinal_labels_per_handoff_long.csv", index=False)

    (TEX_DIR / "ordinal_label_proportions.tex").write_text(
        build_25_word_tex(by_condition), encoding="utf-8"
    )
    (TEX_DIR / "ordinal_label_proportions_baseline.tex").write_text(
        build_baseline_tex(by_condition), encoding="utf-8"
    )

    print_condition_summaries(by_condition)
    print(f"Wrote {OUT_DIR / 'ordinal_label_proportions_by_condition.csv'}")
    print(f"Wrote {OUT_DIR / 'ordinal_label_proportions_overall.csv'}")
    print(f"Wrote {OUT_DIR / 'ordinal_labels_per_handoff_long.csv'}")
    print(f"Wrote {TEX_DIR / 'ordinal_label_proportions.tex'}")
    print(f"Wrote {TEX_DIR / 'ordinal_label_proportions_baseline.tex'}")


if __name__ == "__main__":
    main()
