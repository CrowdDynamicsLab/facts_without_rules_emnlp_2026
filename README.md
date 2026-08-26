# Facts Without Rules: Boundary Metadata Collapse in Multi-Agent LLM Handoffs

Code for the EMNLP 2026 Findings paper *Facts Without Rules: Boundary Metadata
Collapse in Multi-Agent LLM Handoffs*.

Multi-agent LLM systems coordinate by compressing an upstream interaction into a
handoff artifact that the next agent treats as shared state. We show that this
compression step is a structural source of privacy leakage: summaries keep the
operational facts a downstream agent needs to act, while selectively dropping the
boundary metadata that governs how those facts may be used — audience
constraints, ownership claims, hedges, disclosure caveats. We call this failure
mode **summary collapse**.

The repository contains the BOUND-HANDOFF pipeline: prompt builders, model
runners, survival judges, leakage detectors, and the analysis scripts behind the
tables and figures in the paper.

## What the measurements are

| Symbol | Meaning |
|---|---|
| `σ_b` | Boundary-marker survival. Mean over upstream boundary markers of a four-level judge score: preserved (1.0), paraphrased (0.75), weakened (0.35), absent (0.0). |
| `σ_op` | Operational-fact survival. The same score over task-relevant facts — times, entities, decisions, owners, constraints. |
| `ρ` | Compression ratio, `1 - |h|/|u|`, over token lengths of the upstream transcript and the handoff artifact. |

The central finding is that `σ_b` and `σ_op` decouple: under compression `σ_b`
falls sharply while `σ_op` stays near ceiling.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Set your model endpoint credential in the environment. No secret values are
stored in this repository.

```bash
export MODEL_KEY="..."
```

## Repository layout

```
dataset/scripts/     prompt builders, model runners, downstream evaluators,
                     sigma judges, task-success judges, experiment summarizers
dataset/data/        expected location for benchmark manifests and prompt files
scripts/analysis/    statistical tests, robustness checks, annotation tooling,
                     qualitative example extraction, E9 corruption, summaries
RUNBOOK.txt          representative end-to-end commands
```

## Pipeline

The four stages run in order. Replace `MODEL_NAME` and `JUDGE_MODEL` with your
configured model identifiers.

**1. Build prompts.** Each experiment family has its own builder:

```bash
python3 dataset/scripts/build_handoff_prompts.py              # E1, E2
python3 dataset/scripts/build_downstream_pressure_prompts.py  # E3
python3 dataset/scripts/build_marker_gradient_prompts.py      # E8
python3 dataset/scripts/build_single_agent_control_prompts.py # E5
python3 dataset/scripts/build_mitigation_handoff_prompts.py   # E6
python3 dataset/scripts/build_operational_lifting_prompts.py  # E9
python3 scripts/analysis/build_e9_corrupted_allowlist_prompts.py
```

**2. Run the model.**

```bash
python3 dataset/scripts/run_openai_prompts.py \
    --prompts dataset/data/bound_handoff_phase2_prompts.jsonl \
    --output  dataset/data/model_outputs_MODEL_phase2.jsonl \
    --model MODEL_NAME --temperature 0.1
```

**3. Score survival and leakage.**

```bash
python3 dataset/scripts/judge_phase2_sigma.py \
    --dataset dataset/data/bound_handoff_phase2.json \
    --outputs dataset/data/model_outputs_MODEL_phase2.jsonl \
    --judge-output dataset/data/judge_scores_MODEL_phase2_sigma.jsonl \
    --judge-model JUDGE_MODEL --reasoning-effort minimal

python3 dataset/scripts/evaluate_downstream_outputs.py \
    dataset/data/bound_handoff_phase2.json \
    dataset/data/model_outputs_MODEL_downstream.jsonl \
    --output dataset/data/eval_MODEL_downstream.jsonl
```

Leakage is scored by two complementary detectors: a scenario-specific
exact/alias matcher, reported as a reproducible high-precision baseline, and a
calibrated semantic judge with higher recall on paraphrase and category-level
disclosure. Both are reported throughout; neither is treated as ground truth.

**4. Analyze.**

```bash
python3 dataset/scripts/compute_sigma_vs_rho.py
python3 dataset/scripts/bootstrap_experiment_summaries.py
python3 dataset/scripts/compute_ordinal_label_proportions.py   # raw four-level label distribution
python3 dataset/scripts/compute_ordinal_bootstrap_and_cmh.py   # bootstrap CIs + CMH tests
python3 scripts/analysis/add_leakage_stats.py
python3 scripts/analysis/sigma_weighting_robustness.py
python3 scripts/analysis/category_surface_breakdowns.py
python3 scripts/analysis/summarize_qwen3_32b.py
```

`RUNBOOK.txt` has the full argument lists.

## Additional controls

Four further experiments address alternative explanations for the main result.

**Multi-hop.** Whether boundary loss accumulates across successive handoffs,
comparing full replay against partial memory recall:

```bash
python3 scripts/analysis/build_exp5_multihop_prompts.py
python3 scripts/analysis/summarize_exp5_multihop.py
python3 scripts/analysis/plot_exp5_multihop.py
```

**Spontaneous marker emission.** Whether agents add governing language on their
own when given neutral prompts containing no privacy cue:

```bash
python3 scripts/analysis/build_spontaneous_marker_prompts.py
python3 scripts/analysis/judge_spontaneous_marker_emission.py
python3 scripts/analysis/analyze_spontaneous_marker_emission.py
python3 scripts/analysis/plot_spontaneous_marker_probe.py
```

**Graded noisy allowlist.** How E9's protection degrades under imperfect
boundary extraction at 5/10/20/30% audience-label error:

```bash
python3 scripts/analysis/plot_exp4_graded_allowlist_curve.py
```

**Corrupted allowlist.** Whether protection comes from typed structure or from
the correctness of the boundary field:

```bash
python3 scripts/analysis/build_e9_corrupted_allowlist_prompts.py
```

## Model runners

`dataset/scripts/run_openai_prompts.py` is the generic hosted-endpoint runner.
`dataset/scripts/run_uiuc_chat_prompts.py` is the institution-specific runner
used for the open-weight models; it reads its credential from an environment
variable named by `--api-key-env` and stores no secret values.

## Human annotation

`scripts/build_annotation_packets.py` and `scripts/merge_annotation_results.py`
produce and consolidate the manual audit packets;
`scripts/annotation_instructions.md` is the instruction sheet given to
annotators. Detector calibration and judge validation were performed by two of
the authors, blind to condition. The E7 residual-leakage audit was additionally
scored by two annotators external to the author team, who worked independently
and did not reconcile; we report the intersection of their flags as a
conservative confirmed-leak floor.

## Models

Experiments run on GPT-5-mini and DeepSeek-R1-32B as primary models, with
Qwen3-32B as a targeted third-model replication and a fixed-snapshot flagship
model as a capability control. All runs use temperature 0.1 with matched
prompts across models.

## Data

This bundle contains code only. No generated result files, model outputs, or
judge scores are included. Place benchmark manifests and prompt files under
`dataset/data/` before running.

## Notes on reproduction

Results depend on hosted model endpoints that change over time. Absolute numbers
will not reproduce exactly against different model snapshots; the qualitative
finding to check is the `σ_b`/`σ_op` decoupling and the sharp L1→L2 transition in
the marker-strength gradient.

