#!/usr/bin/env python3
"""Plot handoff-level sigma versus sigma_op decoupling.

Reads the precomputed per-handoff CSV, makes a two-panel scatter figure
for GPT-5-mini and DeepSeek-R1-32B, annotates Pearson r and the
low-sigma/high-sigma_op decoupling cell, then copies the same PDF into
the paper figure directory.
"""

from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INPUT_CSV = Path("dataset/data/sigma_decoupling_per_handoff.csv")
WORKING_PDF = Path("dataset/figures/sigma_vs_sigma_op_scatter.pdf")
PAPER_PDF = Path("emnlp_2026/figures/sigma_vs_sigma_op_scatter.pdf")

MODEL_ORDER = ["gpt-5-mini", "deepseek-r1:32b"]
MODEL_TITLES = {
    "gpt-5-mini": "(a) GPT-5-mini",
    "deepseek-r1:32b": "(b) DeepSeek-R1-32B",
}
CONDITION_ORDER = [
    "free_text",
    "compressed_free_text",
    "preserve_markers_instruction",
    "sectioned_template",
    "structured_schema",
]
CONDITION_COLORS = {
    "free_text": "#4C78A8",
    "compressed_free_text": "#F58518",
    "preserve_markers_instruction": "#54A24B",
    "sectioned_template": "#B279A2",
    "structured_schema": "#E45756",
}
TICKS = [0.0, 0.25, 0.5, 0.75, 1.0]


def pearson_r(x_values, y_values):
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2 or np.isclose(np.std(x), 0) or np.isclose(np.std(y), 0):
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def load_data():
    df = pd.read_csv(INPUT_CSV)
    required = {
        "prompt_id",
        "scenario_id",
        "model",
        "condition",
        "surface",
        "sigma_handoff",
        "sigma_op_handoff",
        "n_markers",
        "n_op_facts",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError("Missing required columns: {}".format(", ".join(missing)))

    counts = df["model"].value_counts().to_dict()
    gpt_n = int(counts.get("gpt-5-mini", 0))
    deepseek_n = int(counts.get("deepseek-r1:32b", 0))
    print("Read {} rows. GPT n={}, DeepSeek n={}.".format(len(df), gpt_n, deepseek_n))
    if len(df) != 360 or gpt_n != 180 or deepseek_n != 180:
        raise ValueError("Expected 360 rows total with 180 rows per model.")
    return df


def jitter(values, amplitude=0.01):
    values = np.asarray(values, dtype=float)
    return np.clip(values + np.random.uniform(-amplitude, amplitude, size=len(values)), 0.0, 1.05)


def plot_model(ax, data, model):
    model_df = data[data["model"] == model]
    ax.set_axisbelow(True)
    ax.axline((0, 0), slope=1, color="#BBBBBB", linestyle="--", linewidth=1.0, zorder=0)
    ax.axvspan(0.9, 1.05, ymin=0.0, ymax=0.7 / 1.05, color="lightcoral", alpha=0.10, zorder=0)

    for condition in CONDITION_ORDER:
        subset = model_df[model_df["condition"] == condition]
        if subset.empty:
            continue
        ax.scatter(
            jitter(subset["sigma_op_handoff"].to_numpy()),
            jitter(subset["sigma_handoff"].to_numpy()),
            marker="o",
            s=28,
            alpha=0.5,
            color=CONDITION_COLORS[condition],
            label=condition,
            linewidths=0,
        )

    r = pearson_r(model_df["sigma_op_handoff"], model_df["sigma_handoff"])
    decoupled = model_df[
        (model_df["sigma_handoff"] < 0.7) & (model_df["sigma_op_handoff"] > 0.9)
    ]
    decoupled_n = len(decoupled)

    ax.text(
        1.02,
        0.05,
        "Pearson r = {r:.3f}\ndecoupling cell: {n}/{total}".format(
            r=r, n=decoupled_n, total=len(model_df)
        ),
        ha="right",
        va="bottom",
        fontsize=9,
    )
    ax.text(
        1.02,
        0.68,
        "decoupling: $\\sigma$ low,\n$\\sigma_{\\mathrm{op}}$ high",
        ha="right",
        va="top",
        fontsize=9,
        color="#8F4A4A",
    )
    ax.set_title(MODEL_TITLES[model], fontsize=10)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(TICKS)
    ax.set_yticks(TICKS)
    ax.tick_params(axis="both", labelsize=8)
    print(
        "{}: Pearson r = {:.3f}; decoupling cell: {}/{}".format(
            model, r, decoupled_n, len(model_df)
        )
    )
    return r, decoupled_n


def make_figure(df):
    np.random.seed(20260515)
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.0), dpi=300, sharex=True, sharey=True)
    handles = []
    labels = []
    for ax, model in zip(axes, MODEL_ORDER):
        plot_model(ax, df, model)
        if not handles:
            handles, labels = ax.get_legend_handles_labels()

    for ax in axes:
        ax.set_xlabel("$\\sigma_{\\mathrm{op}}$ (operational-fact survival)")
    axes[0].set_ylabel("$\\sigma$ (boundary-marker survival)")

    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=3,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0.12, 1, 1))

    WORKING_PDF.parent.mkdir(parents=True, exist_ok=True)
    PAPER_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(WORKING_PDF, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    shutil.copyfile(WORKING_PDF, PAPER_PDF)
    print("Wrote {}".format(WORKING_PDF))
    print("Wrote {}".format(PAPER_PDF))


def main():
    df = load_data()
    make_figure(df)


if __name__ == "__main__":
    main()
