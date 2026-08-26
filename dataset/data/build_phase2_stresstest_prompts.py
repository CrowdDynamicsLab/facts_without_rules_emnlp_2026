#!/usr/bin/env python3
"""Build Phase 2 stress-test handoff prompts for selected conditions."""

import argparse
import json
from pathlib import Path

from build_handoff_prompts import prompt_for_condition


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--conditions", nargs="+", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    dataset = json.loads(args.dataset.read_text())
    records = []
    for scenario in dataset["scenarios"]:
        for condition in args.conditions:
            records.append(
                {
                    "prompt_id": "{}__{}".format(scenario["scenario_id"], condition),
                    "scenario_id": scenario["scenario_id"],
                    "domain": scenario["domain"],
                    "handoff_surface": scenario["handoff_surface"],
                    "topology": scenario["topology"],
                    "condition": condition,
                    "prompt": prompt_for_condition(scenario, condition),
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    print("Wrote {} stress-test prompts to {}".format(len(records), args.output))


if __name__ == "__main__":
    main()
