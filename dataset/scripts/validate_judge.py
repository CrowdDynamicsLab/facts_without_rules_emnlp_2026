#!/usr/bin/env python3
"""Prepare and analyze a 50-row human audit of the Phase 2 sigma judge."""

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path


LABELS = ["preserved", "paraphrased", "weakened", "absent"]
CONDITIONS = [
    "free_text",
    "compressed_free_text",
    "preserve_markers_instruction",
    "sectioned_template",
    "structured_schema",
]
CATEGORIES = ["audience", "ownership", "hedge", "caveat"]


def read_jsonl(path):
    rows = []
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def write_csv(path, rows, fieldnames):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def load_outputs(path):
    return {row["prompt_id"]: row for row in read_jsonl(path)}


def build_marker_rows(judge_rows, output_rows):
    rows = []
    for judge in judge_rows:
        output = output_rows.get(judge["prompt_id"], {})
        for marker in judge.get("boundary_marker_scores", []):
            rows.append(
                {
                    "prompt_id": judge["prompt_id"],
                    "scenario_id": judge["scenario_id"],
                    "condition": judge["condition"],
                    "marker_id": marker.get("marker_id", ""),
                    "marker_category": marker.get("category", ""),
                    "gold_marker_text": marker.get("gold_marker", ""),
                    "generated_handoff_text": output.get("output", ""),
                    "judge_label": marker.get("label", ""),
                    "human_label": "",
                }
            )
    return rows


def stratified_sample(rows, n, seed):
    rng = random.Random(seed)
    by_cell = defaultdict(list)
    for row in rows:
        by_cell[(row["condition"], row["marker_category"])].append(row)
    for cell_rows in by_cell.values():
        cell_rows.sort(key=lambda r: (r["scenario_id"], r["prompt_id"], r["marker_id"]))
        rng.shuffle(cell_rows)

    cells = [(condition, category) for condition in CONDITIONS for category in CATEGORIES]
    sample = []
    used = set()

    # Two per cell gives 40 rows across the 5 x 4 stratification.
    for cell in cells:
        for row in by_cell.get(cell, [])[:2]:
            key = (row["prompt_id"], row["marker_id"])
            if key not in used:
                sample.append(row)
                used.add(key)

    # Add the remaining 10 in a deterministic round-robin over cells.
    while len(sample) < n:
        added = False
        for cell in cells:
            if len(sample) >= n:
                break
            for row in by_cell.get(cell, []):
                key = (row["prompt_id"], row["marker_id"])
                if key in used:
                    continue
                sample.append(row)
                used.add(key)
                added = True
                break
        if not added:
            break

    sample.sort(key=lambda r: (r["condition"], r["marker_category"], r["scenario_id"], r["marker_id"]))
    return sample


def cohen_kappa(pairs):
    total = len(pairs)
    if total == 0:
        return None
    observed = sum(1 for a, b in pairs if a == b) / total
    left = Counter(a for a, _ in pairs)
    right = Counter(b for _, b in pairs)
    expected = sum((left[label] / total) * (right[label] / total) for label in LABELS)
    if math.isclose(1.0, expected):
        return 1.0 if math.isclose(observed, 1.0) else None
    return (observed - expected) / (1.0 - expected)


def agreement_summary(rows):
    labeled = [
        row
        for row in rows
        if row.get("human_label") in LABELS and row.get("judge_label") in LABELS
    ]
    pairs = [(row["judge_label"], row["human_label"]) for row in labeled]
    confusion = {j: {h: 0 for h in LABELS} for j in LABELS}
    for judge_label, human_label in pairs:
        confusion[judge_label][human_label] += 1

    by_category = {}
    for category in CATEGORIES:
        cat_rows = [row for row in labeled if row.get("marker_category") == category]
        cat_pairs = [(row["judge_label"], row["human_label"]) for row in cat_rows]
        by_category[category] = {
            "n": len(cat_rows),
            "cohen_kappa": cohen_kappa(cat_pairs),
            "accuracy": (
                sum(1 for j, h in cat_pairs if j == h) / len(cat_pairs) if cat_pairs else None
            ),
        }

    return {
        "status": "complete" if len(labeled) == len(rows) and rows else "awaiting_human_labels",
        "n_rows": len(rows),
        "n_labeled": len(labeled),
        "cohen_kappa_overall": cohen_kappa(pairs),
        "accuracy_overall": (
            sum(1 for j, h in pairs if j == h) / len(pairs) if pairs else None
        ),
        "per_marker_category": by_category,
        "confusion_matrix_judge_rows_human_columns": confusion,
    }


def write_summary_doc(path, summary, sample_path, labeled_path, agreement_path):
    kappa = summary.get("cohen_kappa_overall")
    accuracy = summary.get("accuracy_overall")
    status = summary.get("status")
    if kappa is None:
        headline = "Human labels pending; agreement statistics not yet computed."
        finding = (
            "The audit sheet has been generated and spot-check rows are available. "
            "Once human labels are filled, rerun the script to compute Cohen's kappa, "
            "accuracy, and the confusion matrix."
        )
    else:
        headline = f"Overall Cohen's kappa = {kappa:.3f}; accuracy = {accuracy:.3f}."
        weak = sorted(
            summary["per_marker_category"].items(),
            key=lambda item: -1 if item[1]["cohen_kappa"] is None else item[1]["cohen_kappa"],
        )[0]
        finding = (
            f"The judge-human comparison is {status}; the weakest category by kappa is "
            f"{weak[0]} ({weak[1]['cohen_kappa']:.3f})."
        )

    text = f"""# Judge validation

## Headline numbers
- {headline}
- Labeled rows: {summary.get("n_labeled", 0)} / {summary.get("n_rows", 0)}.

## What we ran
- We prepared a deterministic 50-marker stratified audit sample from the GPT-5-mini Phase 2 E1 judge outputs.
- The sample balances the five handoff conditions and four marker categories, with 2-3 rows per cell.
- The audit sheet includes the generated handoff text, gold marker, and judge label; `human_label` is blank for manual annotation.

## What we found
- {finding}

## How to cite this in the paper
- We validate the boundary-survival judge on a 50-marker stratified human audit spanning all handoff conditions and marker categories. Agreement is reported with Cohen's kappa, per-category accuracy, and a four-label confusion matrix over preserved, paraphrased, weakened, and absent labels.

## Caveats
- The human labels are not produced by this script; Yian must fill `human_label` before final agreement numbers are meaningful.
- The sample validates the GPT-5-mini E1 judge directly; DeepSeek uses the same rubric but is not separately human-audited here.

## Files produced
- `{sample_path}`: audit CSV for human labeling.
- `{labeled_path}`: JSONL mirror of the sampled rows; fill or replace `human_label` before rerunning.
- `{agreement_path}`: agreement summary JSON.
"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge", default="data/judge_scores_openai_gpt5mini_phase2_sigma.jsonl")
    parser.add_argument("--outputs", default="data/model_outputs_openai_gpt5mini_phase2.jsonl")
    parser.add_argument("--audit-csv", default="data/judge_validation_sample50_audit_sheet.csv")
    parser.add_argument("--labeled-jsonl", default="data/judge_validation_sample50_labeled.jsonl")
    parser.add_argument("--summary-json", default="data/judge_validation_agreement_summary.json")
    parser.add_argument("--doc", default="docs/judge_validation_summary.md")
    parser.add_argument("--seed", type=int, default=20260514)
    parser.add_argument("--n", type=int, default=50)
    args = parser.parse_args()

    judge_rows = read_jsonl(args.judge)
    output_rows = load_outputs(args.outputs)
    marker_rows = build_marker_rows(judge_rows, output_rows)
    sample = stratified_sample(marker_rows, args.n, args.seed)

    fieldnames = [
        "prompt_id",
        "scenario_id",
        "condition",
        "marker_category",
        "gold_marker_text",
        "generated_handoff_text",
        "judge_label",
        "human_label",
    ]
    write_csv(args.audit_csv, sample, fieldnames)

    labeled_path = Path(args.labeled_jsonl)
    if labeled_path.exists():
        labeled_rows = read_jsonl(labeled_path)
    else:
        labeled_rows = sample
        write_jsonl(labeled_path, labeled_rows)

    summary = agreement_summary(labeled_rows)
    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_json).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_summary_doc(args.doc, summary, args.audit_csv, args.labeled_jsonl, args.summary_json)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
