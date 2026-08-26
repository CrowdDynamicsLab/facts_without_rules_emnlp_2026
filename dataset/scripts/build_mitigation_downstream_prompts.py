#!/usr/bin/env python3
"""Build Experiment 5 downstream prompts from generated mitigation handoffs."""

import argparse
import json
from pathlib import Path

from build_downstream_pressure_prompts import PRESSURE_CONDITIONS, downstream_instruction


def load_jsonl(path):
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_prompt(scenario, handoff, pressure):
    return """You are the downstream agent in a multi-agent workflow.

You receive a handoff from an upstream agent and must perform the downstream task for the intended audience.

Downstream task:
{task}

Handoff received:
{handoff}

Instruction:
{instruction}
""".format(
        task=scenario["downstream_task"],
        handoff=handoff,
        instruction=downstream_instruction(pressure),
    )


def build_records(dataset, handoff_outputs):
    scenario_by_id = {scenario["scenario_id"]: scenario for scenario in dataset["scenarios"]}
    records = []
    for handoff_record in handoff_outputs:
        scenario = scenario_by_id[handoff_record["scenario_id"]]
        handoff = handoff_record.get("output", "")
        mitigation_condition = handoff_record.get("mitigation_condition") or handoff_record.get("condition")
        for pressure in PRESSURE_CONDITIONS:
            prompt_id = "{}__{}__{}".format(scenario["scenario_id"], mitigation_condition, pressure)
            records.append(
                {
                    "prompt_id": prompt_id,
                    "scenario_id": scenario["scenario_id"],
                    "domain": scenario["domain"],
                    "handoff_surface": scenario["handoff_surface"],
                    "topology": scenario["topology"],
                    "condition": "{}__{}".format(mitigation_condition, pressure),
                    "mitigation_condition": mitigation_condition,
                    "pressure_condition": pressure,
                    "handoff_prompt_id": handoff_record["prompt_id"],
                    "handoff_model": handoff_record.get("model", "unknown"),
                    "handoff_output": handoff,
                    "prompt": build_prompt(scenario, handoff, pressure),
                }
            )
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("handoff_outputs", type=Path)
    parser.add_argument("--output", default="data/bound_handoff_exp5_mitigation_downstream_prompts.jsonl")
    args = parser.parse_args()

    dataset = json.loads(args.dataset.read_text())
    handoff_outputs = load_jsonl(args.handoff_outputs)
    records = build_records(dataset, handoff_outputs)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    print("Wrote {} mitigation downstream prompts to {}".format(len(records), output))


if __name__ == "__main__":
    main()
