#!/usr/bin/env python3
"""Build a manual audit sheet for BOUND-Handoff model outputs.

The sheet is JSONL so annotations can be merged back into later scripts without
spreadsheet parsing. It pre-fills gold facts, heuristic labels, and blank fields
for human labels.
"""

import argparse
import json
from pathlib import Path

from evaluate_handoff_outputs import evaluate_record


def load_jsonl(path):
    records = []
    with path.open() as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("outputs", type=Path)
    parser.add_argument("--output", default="data/bound_handoff_audit_sheet.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    dataset = json.loads(args.dataset.read_text())
    scenario_by_id = {scenario["scenario_id"]: scenario for scenario in dataset["scenarios"]}
    outputs = load_jsonl(args.outputs)
    if args.limit:
        outputs = outputs[: args.limit]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for record in outputs:
            scenario = scenario_by_id[record["scenario_id"]]
            heuristic = evaluate_record(record, scenario_by_id)
            audit_record = {
                "prompt_id": record["prompt_id"],
                "scenario_id": record["scenario_id"],
                "condition": record["condition"],
                "model": record.get("model", "unknown"),
                "domain": scenario["domain"],
                "handoff_surface": scenario["handoff_surface"],
                "output": record["output"],
                "gold_allowed_disclosure": scenario["gold_allowed_disclosure"],
                "gold_disallowed_disclosure": scenario["gold_disallowed_disclosure"],
                "boundary_markers": scenario["boundary_markers"],
                "heuristic": heuristic,
                "human_marker_survival": {},
                "human_leakage": None,
                "human_operational_fact_survival": None,
                "human_context_dump": None,
                "auditor_notes": "",
            }
            f.write(json.dumps(audit_record) + "\n")

    print("Wrote {} audit records to {}".format(len(outputs), output_path))


if __name__ == "__main__":
    main()
