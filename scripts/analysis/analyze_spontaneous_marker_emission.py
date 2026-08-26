#!/usr/bin/env python3
"""Analyze spontaneous-marker emission and prepare blinded human calibration."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path


TIERS = ["low", "medium", "high"]
TIER_SCORE = {x: i for i, x in enumerate(TIERS)}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def quantile(values: list[float], q: float) -> float:
    values = sorted(values)
    return values[min(len(values) - 1, max(0, int(q * len(values))))]


def bootstrap_ci(rows: list[dict], key: str, rng: random.Random, reps: int) -> tuple[float, float]:
    by_scenario = defaultdict(list)
    for row in rows:
        by_scenario[row["scenario_id"]].append(float(row[key]))
    ids = sorted(by_scenario)
    estimates = []
    for _ in range(reps):
        sample = [rng.choice(ids) for _ in ids]
        values = [v for sid in sample for v in by_scenario[sid]]
        estimates.append(sum(values) / len(values))
    return quantile(estimates, .025), quantile(estimates, .975)


def slope(scores: list[int], values: list[float]) -> float:
    xm, ym = sum(scores) / len(scores), sum(values) / len(values)
    denom = sum((x - xm) ** 2 for x in scores)
    return sum((x - xm) * (y - ym) for x, y in zip(scores, values)) / denom


def exact_scenario_trend(rows: list[dict]) -> dict:
    by_scenario = defaultdict(list)
    tier_by_scenario = {}
    for row in rows:
        by_scenario[row["scenario_id"]].append(float(row["primary_emission"]))
        tier_by_scenario[row["scenario_id"]] = row["sensitivity"]
    ids = sorted(by_scenario)
    values = [sum(by_scenario[x]) / len(by_scenario[x]) for x in ids]
    observed_scores = [TIER_SCORE[tier_by_scenario[x]] for x in ids]
    observed = slope(observed_scores, values)
    indices = set(range(len(ids)))
    null = []
    for low in itertools.combinations(range(len(ids)), 4):
        remaining = sorted(indices - set(low))
        for medium in itertools.combinations(remaining, 4):
            scores = [2] * len(ids)
            for i in low: scores[i] = 0
            for i in medium: scores[i] = 1
            null.append(slope(scores, values))
    p = sum(abs(x) >= abs(observed) - 1e-12 for x in null) / len(null)
    return {"scenario_level_slope_per_tier": observed, "exact_two_sided_p": p,
            "direction": "increasing" if observed > 0 else "decreasing" if observed < 0 else "flat",
            "n_scenarios": len(ids), "n_exact_permutations": len(null)}


def kappa(pairs: list[tuple[int, int]]) -> float | None:
    if not pairs:
        return None
    n = len(pairs); observed = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    expected = sum((ca[x] / n) * (cb[x] / n) for x in [0, 1])
    if math.isclose(expected, 1):
        return 1.0 if math.isclose(observed, 1) else None
    return (observed - expected) / (1 - expected)


def build_or_read_packet(args: argparse.Namespace, outputs: list[dict], judged: list[dict], scenarios: dict) -> dict:
    output_map = {(x["prompt_id"], x["model"]): x for x in outputs}
    judge_map = {(x["prompt_id"], x["evaluated_model"]): x for x in judged}
    if not args.audit_csv.exists():
        candidates = []
        for key, output in output_map.items():
            judge = judge_map[key]
            scenario = scenarios[output["scenario_id"]]
            protected = scenario["gold_disallowed_disclosure"][0]
            candidates.append({"_key": key, "scenario_id": output["scenario_id"],
                               "condition": output["condition"], "protected_fact": protected,
                               "generated_handoff": output.get("output", "")})
        rng = random.Random(1400)
        rng.shuffle(candidates)
        selected = candidates[:40]
        packet, mapping = [], []
        for index, row in enumerate(selected, 1):
            blind_id = f"SM{index:03d}"
            packet.append({"blind_id": blind_id, "protected_fact": row["protected_fact"],
                           "generated_handoff": row["generated_handoff"],
                           "author_a_emitted": "", "author_a_governs": "",
                           "author_b_emitted": "", "author_b_governs": "",
                           "reconciled_emitted": "", "reconciled_governs": ""})
            mapping.append({"blind_id": blind_id, "prompt_id": row["_key"][0], "model": row["_key"][1]})
        write_csv(args.audit_csv, packet)
        args.audit_mapping.write_text(json.dumps(mapping, indent=2))

    packet = list(csv.DictReader(args.audit_csv.open()))
    mapping = {x["blind_id"]: x for x in json.loads(args.audit_mapping.read_text())}
    complete = all(row[field] in {"0", "1"} for row in packet for field in
                   ["author_a_emitted", "author_a_governs", "author_b_emitted", "author_b_governs",
                    "reconciled_emitted", "reconciled_governs"])
    result = {"status": "complete" if complete else "awaiting_two_author_labels",
              "n": len(packet), "n_complete": 0}
    if complete:
        result["n_complete"] = len(packet)
        for target in ["emitted", "governs"]:
            aa = [(int(r[f"author_a_{target}"]), int(r[f"author_b_{target}"])) for r in packet]
            jh = []
            for row in packet:
                meta = mapping[row["blind_id"]]
                judge = judge_map[(meta["prompt_id"], meta["model"])]
                judge_value = judge["emitted"] if target == "emitted" else judge["governs_sensitive"]
                jh.append((int(judge_value), int(row[f"reconciled_{target}"])))
            result[f"author_kappa_{target}"] = kappa(aa)
            result[f"judge_human_kappa_{target}"] = kappa(jh)
            result[f"judge_human_accuracy_{target}"] = sum(a == b for a, b in jh) / len(jh)
    args.validation_json.write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/bound_handoff_phase2.json"))
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--judged", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/spontaneous"))
    parser.add_argument("--audit-csv", type=Path, default=Path("data/spontaneous/emission_validation_blind40.csv"))
    parser.add_argument("--audit-mapping", type=Path, default=Path("data/spontaneous/emission_validation_blind40_mapping.json"))
    parser.add_argument("--validation-json", type=Path, default=Path("data/spontaneous/emission_judge_validation.json"))
    parser.add_argument("--reps", type=int, default=10000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = json.loads(args.dataset.read_text())
    scenarios = {x["scenario_id"]: x for x in dataset["scenarios"]}
    outputs, judged = read_jsonl(args.outputs), read_jsonl(args.judged)
    assert len(outputs) == len(judged) == 48
    assert {(x["prompt_id"], x["model"]) for x in outputs} == {(x["prompt_id"], x["evaluated_model"]) for x in judged}
    for row in judged:
        row["primary_emission"] = int(row["emitted"] and row["governs_sensitive"])
        row["silent_redaction"] = int(not row["sensitive_fact_carried"] and not row["primary_emission"])
        row["carried_ungoverned"] = int(row["sensitive_fact_carried"] and not row["primary_emission"])

    summary = []
    rng = random.Random(1400)
    for model in sorted({x["evaluated_model"] for x in judged}):
        for tier in TIERS:
            group = [x for x in judged if x["evaluated_model"] == model and x["sensitivity"] == tier]
            record = {"model": model, "sensitivity": tier, "n_handoffs": len(group),
                      "n_scenarios": len({x['scenario_id'] for x in group})}
            for key in ["primary_emission", "silent_redaction", "carried_ungoverned", "sensitive_fact_carried"]:
                value = sum(x[key] for x in group) / len(group)
                lo, hi = bootstrap_ci(group, key, rng, args.reps)
                record.update({key + "_rate": value, key + "_ci_low": lo, key + "_ci_high": hi})
            governing = [x for x in group if x["primary_emission"]]
            strengths = Counter(x["strength"] for x in governing)
            record.update({"governing_emissions": len(governing),
                           "L1_fraction": strengths["L1_vague"] / len(governing) if governing else None,
                           "L2_fraction": strengths["L2_soft"] / len(governing) if governing else None,
                           "L3_fraction": strengths["L3_explicit"] / len(governing) if governing else None})
            assert record["primary_emission_ci_low"] <= record["primary_emission_rate"] <= record["primary_emission_ci_high"]
            summary.append(record)

    trends = {}
    for model in sorted({x["evaluated_model"] for x in judged}):
        trends[model] = exact_scenario_trend([x for x in judged if x["evaluated_model"] == model])
    validation = build_or_read_packet(args, outputs, judged, scenarios)
    write_csv(args.output_dir / "summary_spontaneous_emission_by_tier.csv", summary)
    result = {"status": "provisional_pending_human_validation" if validation["status"] != "complete" else "complete",
              "bootstrap_unit": "scenario", "bootstrap_replicates": args.reps,
              "trend_test": "exact scenario-level ordinal-tier permutation test",
              "summary": summary, "trends": trends, "validation": validation}
    (args.output_dir / "summary_spontaneous_emission_by_tier.json").write_text(json.dumps(result, indent=2))
    for row in summary:
        print(row["model"], row["sensitivity"], f"emission={row['primary_emission_rate']:.3f}",
              f"CI=[{row['primary_emission_ci_low']:.3f},{row['primary_emission_ci_high']:.3f}]")
    for model, trend in trends.items():
        print(model, "trend", trend)
    print("emission judge validation:", validation)
    print(f"Wrote analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
