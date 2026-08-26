#!/usr/bin/env python3
"""Merge completed annotation CSVs with internal keys and compute agreement.

This script accepts one or more completed annotator CSVs for the same packet.
Example:
  python3 scripts/merge_annotation_results.py \
    --packet sigma \
    --internal-key annotation_packets/sigma_marker_survival_internal_key.csv \
    --annotator-csvs annotator1.csv annotator2.csv \
    --output annotation_packets/sigma_merged_annotations.csv \
    --summary annotation_packets/annotation_summary.md
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


SIGMA_LABELS = ["absent", "weakened", "paraphrased", "preserved"]
LEAK_LABELS = ["no_leak", "leak"]
LEAK_TYPES = ["none", "exact", "paraphrase", "category"]


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def cohen_kappa(a, b, labels):
    pairs = [(x, y) for x, y in zip(a, b) if x and y]
    if not pairs:
        return None
    total = len(pairs)
    observed = sum(1 for x, y in pairs if x == y) / total
    ca = Counter(x for x, _ in pairs)
    cb = Counter(y for _, y in pairs)
    expected = sum((ca[l] / total) * (cb[l] / total) for l in labels)
    if expected == 1:
        return 1.0
    return (observed - expected) / (1 - expected)


def raw_agreement(a, b):
    pairs = [(x, y) for x, y in zip(a, b) if x and y]
    if not pairs:
        return None
    return sum(1 for x, y in pairs if x == y) / len(pairs)


def merge(args):
    key_rows = {r["annotation_id"]: r for r in read_csv(args.internal_key)}
    annotator_tables = []
    for idx, path in enumerate(args.annotator_csvs, 1):
        rows = read_csv(path)
        annotator_tables.append((f"annotator_{idx}", {r["annotation_id"]: r for r in rows}))

    if args.packet == "sigma":
        label_col = "label"
        allowed = set(SIGMA_LABELS)
        labels = SIGMA_LABELS
    else:
        label_col = "leakage_label"
        allowed = set(LEAK_LABELS)
        labels = LEAK_LABELS

    merged = []
    errors = []
    for annotation_id in sorted(key_rows):
        out = dict(key_rows[annotation_id])
        for annotator, table in annotator_tables:
            row = table.get(annotation_id)
            value = (row.get(label_col, "") if row else "").strip()
            if value and value not in allowed:
                errors.append(f"{annotator}:{annotation_id} invalid {label_col}={value}")
            out[f"{annotator}_{label_col}"] = value
            if args.packet != "sigma":
                leak_type = (row.get("leakage_type", "") if row else "").strip()
                if leak_type and leak_type not in LEAK_TYPES:
                    errors.append(f"{annotator}:{annotation_id} invalid leakage_type={leak_type}")
                out[f"{annotator}_leakage_type"] = leak_type
        merged.append(out)
    if errors:
        for err in errors:
            print("ERROR:", err)
        raise SystemExit(1)
    return merged, labels, label_col


def write_summary(path, args, merged, labels, label_col):
    lines = ["# Annotation Summary", ""]
    lines.append(f"Packet: `{args.packet}`")
    lines.append(f"Items: {len(merged)}")
    lines.append(f"Annotators: {len(args.annotator_csvs)}")
    lines.append("")
    if len(args.annotator_csvs) >= 2:
        a = [r[f"annotator_1_{label_col}"] for r in merged]
        b = [r[f"annotator_2_{label_col}"] for r in merged]
        lines.append(f"Raw agreement: {raw_agreement(a, b):.3f}")
        lines.append(f"Cohen's kappa: {cohen_kappa(a, b, labels):.3f}")
        lines.append("")
    for idx in range(1, len(args.annotator_csvs) + 1):
        col = f"annotator_{idx}_{label_col}"
        counts = Counter(r[col] for r in merged if r.get(col))
        lines.append(f"Annotator {idx} label counts: `{dict(counts)}`")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", choices=["sigma", "e4", "e7"], required=True)
    parser.add_argument("--internal-key", required=True)
    parser.add_argument("--annotator-csvs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    merged, labels, label_col = merge(args)
    fields = list(merged[0].keys()) if merged else []
    write_csv(args.output, merged, fields)
    write_summary(args.summary, args, merged, labels, label_col)
    print(f"Wrote {args.output}")
    print(f"Wrote {args.summary}")


if __name__ == "__main__":
    main()
