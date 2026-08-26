#!/usr/bin/env python3
"""Create redacted Experiment 5 handoffs for the enforced-redaction 5B run."""

import argparse
import json
import re
from pathlib import Path

from evaluate_downstream_outputs import scenario_leakage_phrases


REDACTION_TOKEN = "[REDACTED_LOCAL_CONTEXT]"


def load_jsonl(path):
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def phrase_pattern(phrase):
    tokens = re.findall(r"[a-z0-9]+", phrase.lower())
    if not tokens:
        return None
    return re.compile(r"\b" + r"[\W_]+".join(re.escape(token) for token in tokens) + r"\b", re.IGNORECASE)


def redact_text(text, phrases):
    redacted = text or ""
    hits = []
    # Replace longer phrases first so short aliases do not fragment longer evidence.
    for phrase in sorted(set(phrases), key=len, reverse=True):
        pattern = phrase_pattern(phrase)
        if not pattern:
            continue
        redacted, count = pattern.subn(REDACTION_TOKEN, redacted)
        if count:
            hits.append({"phrase": phrase, "count": count})
    return redacted, hits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("handoff_outputs", type=Path)
    parser.add_argument("--output", default="data/model_outputs_openai_gpt5mini_exp5b_mitigation_handoffs_redacted.jsonl")
    args = parser.parse_args()

    dataset = json.loads(args.dataset.read_text())
    scenario_by_id = {scenario["scenario_id"]: scenario for scenario in dataset["scenarios"]}
    rows = load_jsonl(args.handoff_outputs)

    redacted_rows = []
    total_hits = 0
    rows_with_hits = 0
    by_condition = {}

    for row in rows:
        scenario = scenario_by_id[row["scenario_id"]]
        phrases = scenario_leakage_phrases(scenario)
        output, hits = redact_text(row.get("output", ""), phrases)
        total_hits += sum(hit["count"] for hit in hits)
        rows_with_hits += bool(hits)

        condition = row.get("mitigation_condition") or row.get("condition") or "unknown"
        stats = by_condition.setdefault(condition, {"rows": 0, "rows_with_hits": 0, "hit_count": 0})
        stats["rows"] += 1
        stats["rows_with_hits"] += bool(hits)
        stats["hit_count"] += sum(hit["count"] for hit in hits)

        redacted = dict(row)
        redacted["output_original"] = row.get("output", "")
        redacted["output"] = output
        redacted["redaction_token"] = REDACTION_TOKEN
        redacted["redaction_hits"] = hits
        redacted["redaction_hit_count"] = sum(hit["count"] for hit in hits)
        redacted["redaction_enforced"] = True
        redacted_rows.append(redacted)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for row in redacted_rows:
            f.write(json.dumps(row) + "\n")

    print("Read {} handoff outputs".format(len(rows)))
    print("Rows with redactions: {}".format(rows_with_hits))
    print("Total redaction replacements: {}".format(total_hits))
    print("By condition:")
    for condition in sorted(by_condition):
        stats = by_condition[condition]
        print(
            "  {condition}: {rows_with_hits}/{rows} rows, {hit_count} replacements".format(
                condition=condition,
                **stats,
            )
        )
    print("Wrote {}".format(output_path))


if __name__ == "__main__":
    main()
