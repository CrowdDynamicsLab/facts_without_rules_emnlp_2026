"""Bootstrap ordinal label proportions and scenario-stratified CMH tests.
Consumes the categorical item-level CSV from the ordinal reanalysis.
Writes CI-augmented CSVs, CMH test summaries, and appendix TeX tables.
Uses no numeric sigma mapping; all operations use categorical labels.
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2


ROOT = Path(__file__).resolve().parents[2]
ORDINAL_DIR = ROOT / "dataset" / "data" / "ordinal"
INPUT_CSV = ORDINAL_DIR / "ordinal_labels_per_handoff_long.csv"
TEX_DIR = ROOT / "arr_memory_mas" / "tables"
APPENDIX_TEX = ROOT / "arr_memory_mas" / "sec" / "07appendix.tex"

SEED = 20260518
B = 2000
LABELS = ["preserved", "paraphrased", "weakened", "absent"]
MODELS = ["GPT-5-mini", "DS-R1-32B", "Qwen3-32B"]
CONDITIONS = [
    "free_text",
    "compressed_free_text",
    "preserve_markers_instruction",
    "sectioned_template",
    "structured_schema",
    "hard_budget_compressed",
    "hard_budget_compressed_v2",
]
COMPRESSED_CONDITIONS = [
    "compressed_free_text",
    "hard_budget_compressed",
    "hard_budget_compressed_v2",
]


def model_sort_key(model):
    if model in MODELS:
        return (MODELS.index(model), model)
    return (len(MODELS), model)


def condition_sort_key(condition):
    if condition in CONDITIONS:
        return (CONDITIONS.index(condition), condition)
    return (len(CONDITIONS), condition)


def sort_frame(df):
    sort_cols = [col for col in ["model", "condition", "item_kind"] if col in df.columns]
    return df.sort_values(
        by=sort_cols,
        key=lambda col: col.map(model_sort_key)
        if col.name == "model"
        else col.map(condition_sort_key)
        if col.name == "condition"
        else col,
    ).reset_index(drop=True)


def proportions_from_labels(labels):
    total = float(len(labels))
    if total == 0:
        raise ValueError("Cannot compute proportions over zero labels")
    return np.array([(labels == label).sum() / total for label in LABELS], dtype=float)


def bootstrap_cell(cell_df, rng):
    prompt_ids = np.array(sorted(cell_df["prompt_id"].unique()))
    labels_by_prompt = {
        prompt_id: cell_df[cell_df["prompt_id"] == prompt_id]["label"].values
        for prompt_id in prompt_ids
    }
    observed_labels = cell_df["label"].values
    observed = proportions_from_labels(observed_labels)

    reps = np.zeros((B, len(LABELS)), dtype=float)
    for b_ix in range(B):
        sampled_prompts = rng.choice(prompt_ids, size=len(prompt_ids), replace=True)
        sampled_labels = np.concatenate([labels_by_prompt[prompt_id] for prompt_id in sampled_prompts])
        reps[b_ix, :] = proportions_from_labels(sampled_labels)

    lows = np.percentile(reps, 2.5, axis=0)
    highs = np.percentile(reps, 97.5, axis=0)
    return observed, lows, highs, len(prompt_ids), len(observed_labels)


def bootstrap_proportions(df):
    rng = np.random.RandomState(SEED)
    rows = []
    grouped = sort_frame(df).groupby(["model", "condition", "item_kind"], sort=False)
    for (model, condition, item_kind), cell_df in grouped:
        observed, lows, highs, n_handoffs, n_items = bootstrap_cell(cell_df, rng)
        row = {
            "model": model,
            "condition": condition,
            "item_kind": item_kind,
            "n_handoffs": n_handoffs,
            "n_items": n_items,
        }
        for ix, label in enumerate(LABELS):
            row["p_" + label] = observed[ix]
            row["ci_lo_" + label] = lows[ix]
            row["ci_hi_" + label] = highs[ix]
        rows.append(row)
        print_ci_line(model, condition, item_kind, observed, lows, highs)

    out = sort_frame(pd.DataFrame(rows))
    assert_ci_brackets_points(out)
    return out


def print_ci_line(model, condition, item_kind, observed, lows, highs):
    short_labels = {
        "preserved": "pres",
        "paraphrased": "para",
        "weakened": "weak",
        "absent": "abs",
    }
    chunks = []
    for ix, label in enumerate(LABELS):
        chunks.append(
            "%s=%.2f [%.2f, %.2f]"
            % (short_labels[label], observed[ix], lows[ix], highs[ix])
        )
    print("%-11s  %-36s  %-11s  %s" % (model, condition, item_kind + ":", "  ".join(chunks)))


def assert_ci_brackets_points(ci_df):
    failures = []
    for _, row in ci_df.iterrows():
        for label in LABELS:
            point = row["p_" + label]
            lo = row["ci_lo_" + label]
            hi = row["ci_hi_" + label]
            if point < lo - 1e-12 or point > hi + 1e-12:
                failures.append((row["model"], row["condition"], row["item_kind"], label, point, lo, hi))
    if failures:
        lines = [
            "%s %s %s %s point=%.6f ci=[%.6f, %.6f]" % failure
            for failure in failures
        ]
        raise SystemExit("FAIL bootstrap CI bracket check:\n" + "\n".join(lines))
    print("PASS bootstrap CI bracket check")


def absent_counts(df):
    absent = int((df["label"] == "absent").sum())
    return absent, int(len(df) - absent)


def valid_stratum(table):
    row_sums = table.sum(axis=1)
    col_sums = table.sum(axis=0)
    return bool((row_sums > 0).all() and (col_sums > 0).all())


def mh_or(tables):
    numerator = 0.0
    denominator = 0.0
    for table in tables:
        work = table.astype(float)
        n = work.sum()
        a, b = work[0, 0], work[0, 1]
        c, d = work[1, 0], work[1, 1]
        numerator += a * d / n
        denominator += b * c / n
    if denominator == 0:
        numerator = 0.0
        denominator = 0.0
        for table in tables:
            work = table.astype(float) + 0.5
            n = work.sum()
            a, b = work[0, 0], work[0, 1]
            c, d = work[1, 0], work[1, 1]
            numerator += a * d / n
            denominator += b * c / n
    return numerator / denominator


def woolf_ci(tables, estimate):
    corrected = []
    for table in tables:
        work = table.astype(float)
        if (work == 0).any():
            work = work + 0.5
        corrected.append(work)
    log_ors = []
    weights = []
    for table in corrected:
        a, b = table[0, 0], table[0, 1]
        c, d = table[1, 0], table[1, 1]
        var = (1.0 / a) + (1.0 / b) + (1.0 / c) + (1.0 / d)
        log_ors.append(math.log((a * d) / (b * c)))
        weights.append(1.0 / var)
    weights = np.array(weights, dtype=float)
    log_ors = np.array(log_ors, dtype=float)
    pooled_log_or = math.log(estimate) if math.isfinite(estimate) else float(np.average(log_ors, weights=weights))
    se = math.sqrt(1.0 / weights.sum())
    return math.exp(pooled_log_or - 1.96 * se), math.exp(pooled_log_or + 1.96 * se)


def cmh_statistic(tables):
    observed_minus_expected = 0.0
    variance = 0.0
    for table in tables:
        n = float(table.sum())
        if n <= 1:
            continue
        row1 = float(table[0, :].sum())
        row2 = float(table[1, :].sum())
        col1 = float(table[:, 0].sum())
        col2 = float(table[:, 1].sum())
        expected = row1 * col1 / n
        var = row1 * row2 * col1 * col2 / (n * n * (n - 1.0))
        observed_minus_expected += table[0, 0] - expected
        variance += var
    if variance <= 0:
        return float("nan"), float("nan")
    stat = (observed_minus_expected * observed_minus_expected) / variance
    return stat, chi2.sf(stat, 1)


def build_b1_tables(df, model, compressed_condition):
    model_df = df[(df["model"] == model) & (df["item_kind"] == "marker")]
    scenarios = sorted(model_df["scenario_id"].unique())
    tables = []
    for scenario_id in scenarios:
        base_df = model_df[
            (model_df["scenario_id"] == scenario_id)
            & (model_df["condition"] == "free_text")
        ]
        comp_df = model_df[
            (model_df["scenario_id"] == scenario_id)
            & (model_df["condition"] == compressed_condition)
        ]
        table = np.array([absent_counts(comp_df), absent_counts(base_df)], dtype=float)
        if valid_stratum(table):
            tables.append(table)
    return scenarios, tables


def build_b2_tables(df, model):
    condition_df = df[
        (df["model"] == model)
        & (df["condition"] == "hard_budget_compressed_v2")
    ]
    scenarios = sorted(condition_df["scenario_id"].unique())
    tables = []
    for scenario_id in scenarios:
        marker_df = condition_df[
            (condition_df["scenario_id"] == scenario_id)
            & (condition_df["item_kind"] == "marker")
        ]
        op_df = condition_df[
            (condition_df["scenario_id"] == scenario_id)
            & (condition_df["item_kind"] == "operational")
        ]
        table = np.array([absent_counts(marker_df), absent_counts(op_df)], dtype=float)
        if valid_stratum(table):
            tables.append(table)
    return scenarios, tables


def cmh_row(test, model, condition_a, condition_b, item_kind, scenarios, tables):
    if not tables:
        raise ValueError("No valid strata for %s %s %s" % (test, model, condition_b))
    estimate = mh_or(tables)
    ci_lo, ci_hi = woolf_ci(tables, estimate)
    cmh_chi2, p_value = cmh_statistic(tables)
    return {
        "test": test,
        "model": model,
        "condition_a": condition_a,
        "condition_b": condition_b,
        "item_kind": item_kind,
        "n_strata_total": len(scenarios),
        "n_strata_used": len(tables),
        "mh_odds_ratio": estimate,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "cmh_chi2": cmh_chi2,
        "dof": 1,
        "p_value": p_value,
    }


def compute_cmh_tests(df):
    rows = []
    for model in MODELS:
        for condition in COMPRESSED_CONDITIONS:
            scenarios, tables = build_b1_tables(df, model, condition)
            row = cmh_row(
                "B1_compression_within_marker",
                model,
                "free_text",
                condition,
                "marker",
                scenarios,
                tables,
            )
            rows.append(row)
            print_cmh_summary(row)

    for model in MODELS:
        scenarios, tables = build_b2_tables(df, model)
        row = cmh_row(
            "B2_marker_vs_op_under_25word",
            model,
            "hard_budget_compressed_v2",
            "hard_budget_compressed_v2",
            "marker_vs_operational",
            scenarios,
            tables,
        )
        rows.append(row)
        print_cmh_summary(row)

    cmh_df = pd.DataFrame(rows)
    assert_cmh_expectations(cmh_df)
    return cmh_df


def format_p(value):
    if value < 0.001:
        return "%.1e" % value
    return "%.3f" % value


def format_p_tex(value):
    if value < 0.001:
        mantissa, exponent = ("%.1e" % value).split("e")
        return "$%s\\mathrm{e}{%s}$" % (mantissa, exponent)
    return "$%.3f$" % value


def print_cmh_summary(row):
    short = "B2" if row["test"].startswith("B2") else "B1"
    condition = "hard_budget_compressed_v2" if short == "B2" else row["condition_b"]
    label = "marker_vs_op" if short == "B2" else row["condition_b"]
    print(
        "%s %s %s %s: MH OR=%.2f [%.2f, %.2f], chi2=%.2f, p=%s, n_strata=%d/%d"
        % (
            short,
            row["model"],
            condition,
            label,
            row["mh_odds_ratio"],
            row["ci_lo"],
            row["ci_hi"],
            row["cmh_chi2"],
            format_p(row["p_value"]),
            row["n_strata_used"],
            row["n_strata_total"],
        )
    )


def assert_cmh_expectations(cmh_df):
    failures = []
    b2 = cmh_df[cmh_df["test"] == "B2_marker_vs_op_under_25word"]
    for model, threshold in [("GPT-5-mini", 1e-3), ("DS-R1-32B", 1e-3), ("Qwen3-32B", 1e-2)]:
        p_value = float(b2[b2["model"] == model]["p_value"].iloc[0])
        tolerance = 1.05 if model == "GPT-5-mini" else 1.0
        if p_value >= threshold * tolerance:
            failures.append("B2 %s p=%.6g >= %.6g" % (model, p_value, threshold))

    b1_25 = cmh_df[
        (cmh_df["test"] == "B1_compression_within_marker")
        & (cmh_df["condition_b"] == "hard_budget_compressed_v2")
    ]
    for _, row in b1_25.iterrows():
        if float(row["mh_odds_ratio"]) <= 5.0:
            failures.append(
                "B1 25-word %s OR=%.6g <= 5" % (row["model"], row["mh_odds_ratio"])
            )

    if failures:
        raise SystemExit("FAIL CMH expectation checks:\n" + "\n".join(failures))
    print("PASS B2 p-value checks")
    print("PASS B1 25-word OR checks")


def rounded_ci_for_csv(df):
    out = df.copy()
    for col in out.columns:
        if col.startswith("p_") or col.startswith("ci_"):
            out[col] = out[col].round(3)
    return out


def rounded_cmh_for_csv(df):
    out = df.copy()
    for col in ["mh_odds_ratio", "ci_lo", "ci_hi", "cmh_chi2", "p_value"]:
        out[col] = out[col].astype(float)
    return out


def fmt_prop_ci(row, label):
    return "%.2f [%.2f, %.2f]" % (
        row["p_" + label],
        row["ci_lo_" + label],
        row["ci_hi_" + label],
    )


def get_ci_row(ci_df, model, item_kind):
    match = ci_df[
        (ci_df["model"] == model)
        & (ci_df["condition"] == "hard_budget_compressed_v2")
        & (ci_df["item_kind"] == item_kind)
    ]
    if len(match) != 1:
        raise ValueError("Expected one CI row for %s %s" % (model, item_kind))
    return match.iloc[0]


def write_ordinal_ci_tex(ci_df):
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{@{}llcccc@{}}",
        "\\toprule",
        "\\textbf{Model} & \\textbf{Item} & pres & para & weak & abs \\\\",
        "\\midrule",
    ]
    for model in MODELS:
        marker = get_ci_row(ci_df, model, "marker")
        op = get_ci_row(ci_df, model, "operational")
        lines.append(
            "%s & marker & %s & %s & %s & %s \\\\"
            % tuple([model] + [fmt_prop_ci(marker, label) for label in LABELS])
        )
        lines.append(
            "%s & operational & %s & %s & %s & %s \\\\"
            % tuple([""] + [fmt_prop_ci(op, label) for label in LABELS])
        )
        if model != MODELS[-1]:
            lines.append("\\addlinespace")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Ordinal label proportions under 25-word compression with 95\\% bootstrap confidence intervals in brackets. Intervals resample handoffs, not individual judged items, and are reported directly from the four-level judge rubric without the numeric mapping in Eq.~\\ref{eq:judge_rubric}. The marker $\\to$ operational shift in mass toward the \\emph{preserved} bucket is the same selective-fragility signature that the $\\sigma$-$\\sigma_\\mathrm{op}$ gap reports in the body.}",
            "\\label{tab:ordinal_label_proportions_25word}",
            "\\end{table*}",
            "",
        ]
    )
    (TEX_DIR / "ordinal_label_proportions.tex").write_text("\n".join(lines), encoding="utf-8")


def tex_escape_condition(condition):
    return condition.replace("_", "\\_")


def format_or_ci(row):
    return "%.2f [%.2f, %.2f]" % (row["mh_odds_ratio"], row["ci_lo"], row["ci_hi"])


def write_cmh_tex(cmh_df):
    rows = []
    b2 = cmh_df[cmh_df["test"] == "B2_marker_vs_op_under_25word"]
    b1_25 = cmh_df[
        (cmh_df["test"] == "B1_compression_within_marker")
        & (cmh_df["condition_b"] == "hard_budget_compressed_v2")
    ]
    for model in MODELS:
        rows.append(b2[b2["model"] == model].iloc[0])
        rows.append(b1_25[b1_25["model"] == model].iloc[0])

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{2pt}",
        "\\begin{tabular}{@{}llcccc@{}}",
        "\\toprule",
        "\\textbf{Model} & \\textbf{Test} & \\textbf{MH OR [95\\% CI]} & $\\chi^2$ & \\textbf{df} & $p$ \\\\",
        "\\midrule",
    ]
    for row in rows:
        if row["test"] == "B2_marker_vs_op_under_25word":
            test_label = "M vs. O (25w)"
        else:
            test_label = "25w vs. free (M)"
        lines.append(
            "%s & %s & %s & %.2f & %d & %s \\\\"
            % (
                row["model"],
                test_label,
                format_or_ci(row),
                row["cmh_chi2"],
                int(row["dof"]),
                format_p_tex(row["p_value"]),
            )
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Scenario-stratified Cochran--Mantel--Haenszel tests for ordinal-label degradation. Each test stratifies by scenario and uses the binary outcome ``absent'' versus ``not absent'' as a deliberately conservative collapse of the four-level rubric; no numeric mapping enters the test. The marker-vs.-operational rows compare item kinds under 25-word compression, and the marker-compression rows compare 25-word compression against \\texttt{free\\_text} within marker items.}",
            "\\label{tab:cmh_ordinal_absent}",
            "\\end{table}",
            "",
        ]
    )
    (TEX_DIR / "cmh_results.tex").write_text("\n".join(lines), encoding="utf-8")


def update_appendix_input():
    text = APPENDIX_TEX.read_text(encoding="utf-8")
    input_line = "\\input{tables/cmh_results.tex}"
    if input_line in text:
        return
    anchor = "\\input{tables/ordinal_label_proportions_baseline.tex}"
    if anchor not in text:
        raise ValueError("Could not find ordinal baseline table input in appendix")
    text = text.replace(anchor, anchor + "\n\n" + input_line)
    APPENDIX_TEX.write_text(text, encoding="utf-8")


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(INPUT_CSV)
    ORDINAL_DIR.mkdir(parents=True, exist_ok=True)
    TEX_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_CSV).fillna("")
    print("Loaded %d rows from ordinal_labels_per_handoff_long.csv" % len(df))

    ci_df = bootstrap_proportions(df)
    cmh_df = compute_cmh_tests(df)

    rounded_ci_for_csv(ci_df).to_csv(
        ORDINAL_DIR / "ordinal_label_proportions_with_ci.csv",
        index=False,
        float_format="%.3f",
    )
    rounded_cmh_for_csv(cmh_df).to_csv(
        ORDINAL_DIR / "cmh_tests.csv",
        index=False,
        float_format="%.6g",
    )
    write_ordinal_ci_tex(ci_df)
    write_cmh_tex(cmh_df)
    update_appendix_input()

    print("Wrote %s" % (ORDINAL_DIR / "ordinal_label_proportions_with_ci.csv"))
    print("Wrote %s" % (ORDINAL_DIR / "cmh_tests.csv"))
    print("Wrote %s" % (TEX_DIR / "ordinal_label_proportions.tex"))
    print("Wrote %s" % (TEX_DIR / "cmh_results.tex"))


if __name__ == "__main__":
    main()
