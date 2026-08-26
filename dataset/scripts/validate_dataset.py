#!/usr/bin/env python3
"""Validate BOUND-Handoff scenario JSON."""

import argparse
import json
import sys
from pathlib import Path
from typing import List


REQUIRED_SCENARIO_FIELDS = {
    "scenario_id",
    "domain",
    "handoff_surface",
    "topology",
    "upstream_transcript",
    "downstream_task",
    "operational_facts",
    "boundary_markers",
    "local_context_drop",
    "sensitive_facts",
    "gold_allowed_disclosure",
    "gold_disallowed_disclosure",
    "handoff_conditions",
    "evaluation_notes",
}

VALID_MARKER_TYPES = {
    "audience_constraint",
    "ownership_claim",
    "hedge",
    "disclosure_caveat",
}


def fail(errors: List[str], scenario_id: str, message: str) -> None:
    errors.append(f"{scenario_id}: {message}")


def validate_scenario(scenario: dict) -> List[str]:
    errors = []  # type: List[str]
    scenario_id = scenario.get("scenario_id", "<missing scenario_id>")

    missing = REQUIRED_SCENARIO_FIELDS - set(scenario)
    if missing:
        fail(errors, scenario_id, f"missing fields: {sorted(missing)}")
        return errors

    if not scenario["upstream_transcript"]:
        fail(errors, scenario_id, "upstream_transcript is empty")
    if not scenario["downstream_task"].strip():
        fail(errors, scenario_id, "downstream_task is empty")
    if len(scenario["operational_facts"]) < 1:
        fail(errors, scenario_id, "needs at least one operational fact")
    if len(scenario["boundary_markers"]) < 2:
        fail(errors, scenario_id, "needs at least two boundary markers")
    if len(scenario["local_context_drop"]) < 1:
        fail(errors, scenario_id, "needs at least one local_context_drop fact")
    if len(scenario["sensitive_facts"]) < 1:
        fail(errors, scenario_id, "needs at least one sensitive fact id")

    fact_ids = {item["fact_id"] for item in scenario["operational_facts"]}
    drop_ids = {item["fact_id"] for item in scenario["local_context_drop"]}
    all_fact_ids = fact_ids | drop_ids

    for sensitive_id in scenario["sensitive_facts"]:
        if sensitive_id not in drop_ids:
            fail(errors, scenario_id, f"sensitive fact {sensitive_id} is not in local_context_drop")

    marker_types = set()
    for marker in scenario["boundary_markers"]:
        marker_type = marker.get("type")
        marker_types.add(marker_type)
        if marker_type not in VALID_MARKER_TYPES:
            fail(errors, scenario_id, f"invalid marker type {marker_type!r}")
        if not marker.get("applies_to_fact_ids"):
            fail(errors, scenario_id, f"marker {marker.get('marker_id')} has no applies_to_fact_ids")
        for fact_id in marker.get("applies_to_fact_ids", []):
            if fact_id not in all_fact_ids:
                fail(errors, scenario_id, f"marker {marker.get('marker_id')} points to unknown fact {fact_id}")

    if "audience_constraint" not in marker_types and "disclosure_caveat" not in marker_types:
        fail(errors, scenario_id, "needs an audience constraint or disclosure caveat")

    if not scenario["gold_allowed_disclosure"]:
        fail(errors, scenario_id, "gold_allowed_disclosure is empty")
    if not scenario["gold_disallowed_disclosure"]:
        fail(errors, scenario_id, "gold_disallowed_disclosure is empty")

    if "structured_schema" not in scenario["handoff_conditions"]:
        fail(errors, scenario_id, "handoff_conditions must include structured_schema")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()

    data = json.loads(args.dataset.read_text())
    scenarios = data.get("scenarios", [])
    errors = []  # type: List[str]

    if not scenarios:
        errors.append("dataset has no scenarios")

    seen_ids = set()
    for scenario in scenarios:
        scenario_id = scenario.get("scenario_id")
        if scenario_id in seen_ids:
            errors.append(f"{scenario_id}: duplicate scenario_id")
        seen_ids.add(scenario_id)
        errors.extend(validate_scenario(scenario))

    if errors:
        print("Validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    domains = sorted({scenario["domain"] for scenario in scenarios})
    surfaces = sorted({scenario["handoff_surface"] for scenario in scenarios})
    print(f"Validation passed: {len(scenarios)} scenarios")
    print(f"Domains: {', '.join(domains)}")
    print(f"Handoff surfaces: {', '.join(surfaces)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
