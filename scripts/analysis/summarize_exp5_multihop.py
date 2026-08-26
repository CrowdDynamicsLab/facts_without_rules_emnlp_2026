#!/usr/bin/env python3
"""Summarize EXP-5 hop-wise survival and downstream leakage."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


STATE_RE = re.compile(r"(full_replay|partial_recall50)__hop([123])")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def state(record: dict) -> tuple[str, int]:
    match = STATE_RE.search(record.get("condition", "") or record.get("prompt_id", ""))
    if not match:
        raise ValueError(f"Cannot parse chain state: {record.get('prompt_id')}")
    return match.group(1), int(match.group(2))


def pressure(record: dict) -> str:
    value = record.get("pressure_condition")
    if value in {"neutral", "audit_trace"}:
        return value
    return record["condition"].rsplit("__", 1)[-1]


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    return values[min(len(values) - 1, max(0, int(q * len(values))))]


def bootstrap(values_by_scenario: dict[str, list[float]], seed: int, reps: int) -> tuple[float, float]:
    ids = sorted(values_by_scenario)
    rng = random.Random(seed)
    estimates = []
    for _ in range(reps):
        sampled = [rng.choice(ids) for _ in ids]
        values = [v for sid in sampled for v in values_by_scenario[sid]]
        estimates.append(sum(values) / len(values))
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def paired_bootstrap(delta_by_scenario: dict[str, float], seed: int, reps: int) -> tuple[float, float]:
    ids = sorted(delta_by_scenario)
    rng = random.Random(seed)
    estimates = []
    for _ in range(reps):
        sampled = [rng.choice(ids) for _ in ids]
        estimates.append(sum(delta_by_scenario[x] for x in sampled) / len(sampled))
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", action="append", nargs=4, metavar=("MODEL", "SIGMA", "EXACT", "SEMANTIC"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reps", type=int, default=10000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    survival_rows, delta_rows, leakage_rows = [], [], []
    for model_index, (model, sigma_path, exact_path, semantic_path) in enumerate(args.spec):
        sigma = read_jsonl(Path(sigma_path))
        exact = read_jsonl(Path(exact_path))
        semantic = read_jsonl(Path(semantic_path))

        grouped = defaultdict(list)
        for row in sigma:
            grouped[state(row)].append(row)
        state_maps = {}
        for (variant, hop), rows in sorted(grouped.items()):
            b = defaultdict(list); op = defaultdict(list)
            for row in rows:
                b[row["scenario_id"]].append(float(row["overall"]["boundary_sigma_numeric"]))
                op[row["scenario_id"]].append(float(row["overall"]["operational_sigma_numeric"]))
            blo, bhi = bootstrap(b, 1400 + model_index * 100 + hop, args.reps)
            olo, ohi = bootstrap(op, 2400 + model_index * 100 + hop, args.reps)
            bmean = sum(v for x in b.values() for v in x) / sum(map(len, b.values()))
            omean = sum(v for x in op.values() for v in x) / sum(map(len, op.values()))
            survival_rows.append({"model": model, "variant": variant, "hop": hop, "n": len(rows),
                                  "sigma_b": bmean, "sigma_b_ci_low": blo, "sigma_b_ci_high": bhi,
                                  "sigma_op": omean, "sigma_op_ci_low": olo, "sigma_op_ci_high": ohi})
            state_maps[(variant, hop)] = {
                "b": {k: sum(v) / len(v) for k, v in b.items()},
                "op": {k: sum(v) / len(v) for k, v in op.items()},
            }

        for variant in ["full_replay", "partial_recall50"]:
            states = {1: state_maps[("full_replay", 1)],
                      2: state_maps[(variant, 2)], 3: state_maps[(variant, 3)]}
            for start, end in [(1, 2), (2, 3), (1, 3)]:
                for metric in ["b", "op"]:
                    ids = sorted(set(states[start][metric]) & set(states[end][metric]))
                    deltas = {sid: states[end][metric][sid] - states[start][metric][sid] for sid in ids}
                    mean = sum(deltas.values()) / len(deltas)
                    lo, hi = paired_bootstrap(deltas, 3400 + model_index * 1000 + end * 100 + (0 if metric == "b" else 1), args.reps)
                    delta_rows.append({"model": model, "variant": variant, "metric": "sigma_b" if metric == "b" else "sigma_op",
                                       "contrast": f"hop{end}-hop{start}", "n_scenarios": len(ids),
                                       "paired_delta": mean, "ci_low": lo, "ci_high": hi})

        exact_map = {x["prompt_id"]: x for x in exact}
        semantic_map = {x["prompt_id"]: x for x in semantic}
        assert set(exact_map) == set(semantic_map), (len(exact_map), len(semantic_map))
        lgroups = defaultdict(list)
        for prompt_id, erow in exact_map.items():
            variant, hop = state(erow)
            pair = (erow, semantic_map[prompt_id])
            lgroups[(variant, hop, pressure(erow))].append(pair)
            lgroups[(variant, hop, "pooled")].append(pair)
        for (variant, hop, press), pairs in sorted(lgroups.items()):
            exact_by = defaultdict(list); semantic_by = defaultdict(list)
            types = Counter()
            for erow, srow in pairs:
                exact_by[erow["scenario_id"]].append(float(erow["final_leakage_heuristic"]))
                semantic_by[erow["scenario_id"]].append(float(srow["semantic_final_leakage"]))
                types[srow["semantic_leakage_type"]] += 1
            elo, ehi = bootstrap(exact_by, 4400 + model_index * 1000 + hop * 10, args.reps)
            slo, shi = bootstrap(semantic_by, 5400 + model_index * 1000 + hop * 10, args.reps)
            evals = [v for x in exact_by.values() for v in x]
            svals = [v for x in semantic_by.values() for v in x]
            leakage_rows.append({"model": model, "variant": variant, "hop": hop, "pressure": press, "n": len(pairs),
                                 "exact_leaks": int(sum(evals)), "exact_rate": sum(evals) / len(evals),
                                 "exact_ci_low": elo, "exact_ci_high": ehi,
                                 "semantic_leaks": int(sum(svals)), "semantic_rate": sum(svals) / len(svals),
                                 "semantic_ci_low": slo, "semantic_ci_high": shi,
                                 "semantic_exact": types["exact"], "semantic_paraphrase": types["paraphrase"],
                                 "semantic_category_inference": types["category_inference"]})

    write_csv(args.output_dir / "summary_exp5_multihop_survival.csv", survival_rows)
    write_csv(args.output_dir / "summary_exp5_multihop_paired_deltas.csv", delta_rows)
    write_csv(args.output_dir / "summary_exp5_multihop_leakage.csv", leakage_rows)
    (args.output_dir / "summary_exp5_multihop.json").write_text(json.dumps({
        "bootstrap_unit": "scenario", "bootstrap_replicates": args.reps,
        "survival": survival_rows, "paired_deltas": delta_rows, "leakage": leakage_rows,
    }, indent=2))
    print(f"Wrote EXP-5 summaries to {args.output_dir}")


if __name__ == "__main__":
    main()
