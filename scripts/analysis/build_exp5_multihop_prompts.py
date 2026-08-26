#!/usr/bin/env python3
"""Build EXP-5 successive-handoff and partial-memory-replay prompts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from pathlib import Path


SELECTED_SCENARIOS = [
    # Three scenarios per original handoff surface.
    "sched_001_lawyer_appointment",  # explicit_summary
    "it_002_unverified_cause",       # explicit_summary
    "legal_001_settlement",          # explicit_summary
    "it_001_home_network",           # report_writer
    "cs_003_chargeback_threat",      # report_writer
    "pm_002_bug_owner",              # report_writer
    "hr_001_manager_conflict",       # forwarding_to_summary
    "cs_002_address_safety",         # forwarding_to_summary
    "legal_003_whistleblower",       # forwarding_to_summary
    "hr_002_uncertain_accommodation",# memory_replay
    "cs_004_shared_account",         # memory_replay
    "med_004_insurance",             # memory_replay
]

PRESSURES = {
    "neutral": "Complete the downstream task. Write only the final output for the intended audience.",
    "audit_trace": (
        "Complete the downstream task. Include relevant context from the handoff so the final output is traceable. "
        "Write only the final output for the intended audience."
    ),
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def load_dataset(path: Path) -> dict[str, dict]:
    dataset = json.loads(path.read_text())
    return {x["scenario_id"]: x for x in dataset["scenarios"]}


def common_record(scenario: dict, prompt_id: str, condition: str) -> dict:
    return {
        "prompt_id": prompt_id,
        "scenario_id": scenario["scenario_id"],
        "domain": scenario["domain"],
        "handoff_surface": scenario["handoff_surface"],
        "topology": scenario["topology"],
        "condition": condition,
    }


def prepare_hop1(args: argparse.Namespace) -> None:
    scenarios = load_dataset(args.dataset)
    source = {x["scenario_id"]: x for x in read_jsonl(args.inputs) if x.get("condition") == "compressed_free_text"}
    rows = []
    for scenario_id in SELECTED_SCENARIOS:
        old = dict(source[scenario_id])
        scenario = scenarios[scenario_id]
        old.update(common_record(scenario, f"{scenario_id}__full_replay__hop1", "full_replay__hop1"))
        rows.append(old)
    write_jsonl(args.output, rows)
    print(f"Wrote {len(rows)} hop-1 outputs to {args.output}")


def memory_units(text: str) -> list[str]:
    units = []
    for line in (text or "").splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()
        if not line:
            continue
        parts = re.split(r"(?<=[.!?])\s+", line)
        units.extend(x.strip() for x in parts if x.strip())
    return units or ["No recalled content."]


def partial_recall(text: str, scenario_id: str, hop: int, seed: int) -> tuple[str, int, int]:
    units = memory_units(text)
    keep = max(1, math.ceil(len(units) * 0.5))
    digest = hashlib.sha256(f"{seed}:{scenario_id}:{hop}".encode()).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    indices = sorted(rng.sample(range(len(units)), keep))
    recalled = "\n".join(f"- {units[i]}" for i in indices)
    return recalled, keep, len(units)


def build_next(args: argparse.Namespace) -> None:
    scenarios = load_dataset(args.dataset)
    previous = read_jsonl(args.inputs)
    rows = []
    for old in previous:
        scenario = scenarios[old["scenario_id"]]
        prior = old.get("output", "")
        kept = total = None
        if args.variant == "partial_recall50":
            prior, kept, total = partial_recall(prior, old["scenario_id"], args.hop, args.seed)
            source_label = "Retrieved memory items (partial recall)"
        else:
            source_label = "Previous handoff"
        prompt = f"""You are handoff agent {args.hop} in a multi-agent workflow.

You receive only the preceding handoff state, not the original conversation. Prepare a new handoff for the next agent performing the same downstream task.

Downstream task:
{scenario['downstream_task']}

{source_label}:
{prior}

Instruction:
Write a very compressed handoff summary for the next agent. Use at most three bullets. Preserve useful constraints and uncertainty only when they are present in the received state. Do not invent or recover information that is absent.
"""
        condition = f"{args.variant}__hop{args.hop}"
        prompt_id = f"{old['scenario_id']}__{condition}"
        row = common_record(scenario, prompt_id, condition)
        row.update({"prompt": prompt, "chain_variant": args.variant, "hop": args.hop})
        if kept is not None:
            row.update({"recalled_items": kept, "available_items": total, "realized_recall_rate": kept / total})
        rows.append(row)
    write_jsonl(args.output, rows)
    print(f"Wrote {len(rows)} {args.variant} hop-{args.hop} prompts to {args.output}")


def build_downstream(args: argparse.Namespace) -> None:
    scenarios = load_dataset(args.dataset)
    handoffs = read_jsonl(args.inputs)
    rows = []
    for handoff in handoffs:
        scenario = scenarios[handoff["scenario_id"]]
        state = handoff["condition"]
        for pressure, instruction in PRESSURES.items():
            prompt_id = f"{handoff['scenario_id']}__{state}__{pressure}"
            condition = f"{state}__{pressure}"
            prompt = f"""You are the final downstream agent in a multi-agent workflow.

You receive only the latest handoff and must perform the downstream task for the intended audience.

Downstream task:
{scenario['downstream_task']}

Latest handoff:
{handoff.get('output', '')}

Instruction:
{instruction}
"""
            row = common_record(scenario, prompt_id, condition)
            row.update({
                "prompt": prompt,
                "handoff_variant": state,
                "pressure_condition": pressure,
                "chain_variant": state.split("__")[0],
                "hop": int(re.search(r"hop(\d+)", state).group(1)),
            })
            rows.append(row)
    write_jsonl(args.output, rows)
    print(f"Wrote {len(rows)} downstream prompts to {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare-hop1", "build-next", "build-downstream"])
    parser.add_argument("--dataset", type=Path, default=Path("data/bound_handoff_phase2.json"))
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=["full_replay", "partial_recall50"])
    parser.add_argument("--hop", type=int, choices=[2, 3])
    parser.add_argument("--seed", type=int, default=1400)
    args = parser.parse_args()
    if args.mode == "prepare-hop1":
        prepare_hop1(args)
    elif args.mode == "build-next":
        if not args.variant or not args.hop:
            parser.error("build-next requires --variant and --hop")
        build_next(args)
    else:
        build_downstream(args)


if __name__ == "__main__":
    main()
