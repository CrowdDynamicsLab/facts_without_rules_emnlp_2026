#!/usr/bin/env python3
"""Run MiniCheck faithfulness scoring for E1 free-text handoffs."""

import argparse
import csv
import json
import math
import re
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text())


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
    with output.open("w", encoding="utf-8") as f:
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


def sentence_chunks(text):
    chunks = []
    for line in (text or "").splitlines():
        line = re.sub(r"^[\s>*\-•\d.)]+", "", line).strip()
        if not line:
            continue
        pieces = re.split(r"(?<=[.!?])\s+", line)
        for piece in pieces:
            piece = " ".join(piece.split()).strip()
            if len(piece.split()) >= 3:
                chunks.append(piece)
    return chunks or [" ".join((text or "").split())]


def scenario_documents(path):
    data = read_json(path)
    docs = {}
    for scenario in data["scenarios"]:
        parts = []
        for turn in scenario.get("upstream_transcript", []):
            sender = turn.get("sender", "Speaker")
            content = turn.get("content", "")
            parts.append(f"{sender}: {content}")
        docs[scenario["scenario_id"]] = "\n".join(parts)
    return docs


def sigma_lookup(paths):
    lookup = {}
    for path in paths:
        if not Path(path).exists():
            continue
        for row in read_jsonl(path):
            lookup[row["prompt_id"]] = {
                "sigma": row.get("overall", {}).get("boundary_sigma_numeric"),
                "sigma_op": row.get("overall", {}).get("operational_sigma_numeric"),
                "n_markers": len(row.get("boundary_marker_scores", [])),
                "n_op_facts": len(row.get("operational_fact_scores", [])),
            }
    return lookup


def model_outputs(paths):
    rows = []
    for path in paths:
        if not Path(path).exists():
            continue
        for row in read_jsonl(path):
            if row.get("condition") == "free_text":
                rows.append(row)
    rows.sort(key=lambda r: (r.get("model", ""), r["scenario_id"], r["prompt_id"]))
    return rows


def model_slug(model):
    if model == "gpt-5-mini":
        return "openai_gpt5mini"
    if model == "deepseek-r1:32b":
        return "uiuc_deepseek"
    return re.sub(r"[^a-zA-Z0-9]+", "_", model).strip("_").lower()


def score_rows(rows, docs, sigmas, cache_dir, batch_size, device):
    import sys
    import nltk

    # Delta py311 can expose apex without fused_layer_norm_cuda; hide it so
    # transformers falls back to standard T5 layer norm.
    sys.modules["apex"] = None
    sys.modules["apex.normalization"] = None

    # Newer NLTK releases require both tokenizer resources.
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

    from minicheck.minicheck import MiniCheck

    # This MiniCheck version uses device_map="auto" internally.
    scorer = MiniCheck(model_name="flan-t5-large", cache_dir=cache_dir)
    claim_records = []
    for idx, row in enumerate(rows):
        for claim in sentence_chunks(row.get("output", "")):
            claim_records.append((idx, docs[row["scenario_id"]], claim))

    all_probs = [[] for _ in rows]
    all_labels = [[] for _ in rows]
    for start in range(0, len(claim_records), batch_size):
        batch = claim_records[start : start + batch_size]
        pred_label, raw_prob, _, _ = scorer.score(
            docs=[item[1] for item in batch],
            claims=[item[2] for item in batch],
        )
        for (row_idx, _, _), label, prob in zip(batch, pred_label, raw_prob):
            all_labels[row_idx].append(int(label))
            all_probs[row_idx].append(float(prob))

    out = []
    for row, labels, probs in zip(rows, all_labels, all_probs):
        sigma = sigmas.get(row["prompt_id"], {})
        score = sum(probs) / len(probs) if probs else math.nan
        out.append(
            {
                "prompt_id": row["prompt_id"],
                "scenario_id": row["scenario_id"],
                "model": row.get("model", ""),
                "faithfulness_minicheck": score,
                "faithfulness_alignscore": math.nan,
                "sigma": sigma.get("sigma"),
                "sigma_op": sigma.get("sigma_op"),
                "n_markers": sigma.get("n_markers"),
                "n_op_facts": sigma.get("n_op_facts"),
                "n_claims": len(labels),
                "minicheck_supported_fraction": sum(labels) / len(labels) if labels else math.nan,
                "minicheck_claim_probs": probs,
            }
        )
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", default="data/bound_handoff_phase2.json")
    parser.add_argument(
        "--outputs",
        nargs="+",
        default=[
            "data/model_outputs_openai_gpt5mini_phase2.jsonl",
            "data/model_outputs_uiuc_deepseek_phase2.jsonl",
        ],
    )
    parser.add_argument(
        "--judges",
        nargs="+",
        default=[
            "data/judge_scores_openai_gpt5mini_phase2_sigma.jsonl",
            "data/judge_scores_uiuc_deepseek_phase2_sigma.jsonl",
        ],
    )
    parser.add_argument("--csv", default="data/faithfulness_scores_e1_free_text.csv")
    parser.add_argument("--jsonl", default="data/faithfulness_scores_e1_free_text.jsonl")
    parser.add_argument("--cache-dir", default="./ckpts")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    docs = scenario_documents(args.scenarios)
    sigmas = sigma_lookup(args.judges)
    rows = model_outputs(args.outputs)
    if args.limit:
        rows = rows[: args.limit]
    scored = score_rows(rows, docs, sigmas, args.cache_dir, args.batch_size, args.device)

    fields = [
        "prompt_id",
        "scenario_id",
        "model",
        "faithfulness_minicheck",
        "faithfulness_alignscore",
        "sigma",
        "sigma_op",
        "n_markers",
        "n_op_facts",
    ]
    write_csv(args.csv, scored, fields)
    write_jsonl(args.jsonl, scored)
    print(json.dumps({"rows": len(scored), "csv": args.csv, "jsonl": args.jsonl}, indent=2))


if __name__ == "__main__":
    main()
