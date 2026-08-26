#!/usr/bin/env python3
"""Plot provisional spontaneous-marker emission and strength results."""

from __future__ import annotations

import csv
from pathlib import Path

from reportlab.graphics import renderPDF, renderSVG
from reportlab.graphics.shapes import Circle, Drawing, Line, PolyLine, Rect, String
from reportlab.lib.colors import HexColor, white


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "rebuttal" / "summary_spontaneous_emission_by_tier.csv"
OUTPUT = ROOT / "rebuttal" / "spontaneous_marker_probe"
TIERS = ["low", "medium", "high"]
MODELS = ["gpt-5-mini", "deepseek-r1:32b"]
COLORS = {"gpt-5-mini": HexColor("#0072B2"), "deepseek-r1:32b": HexColor("#D55E00")}
LABELS = {"gpt-5-mini": "GPT-5-mini", "deepseek-r1:32b": "DeepSeek-R1-32B"}


def text(d: Drawing, x: float, y: float, value: str, size: float = 8,
         anchor: str = "middle", bold: bool = False) -> None:
    d.add(String(x, y, value, fontName="Helvetica-Bold" if bold else "Helvetica",
                 fontSize=size, textAnchor=anchor, fillColor=HexColor("#222222")))


def load() -> dict[tuple[str, str], dict[str, str]]:
    with INPUT.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {(r["model"], r["sensitivity"]): r for r in rows}


def axes(d: Drawing, left: float, bottom: float, width: float, height: float,
         ylabel: str, panel: str) -> tuple:
    grey, dark = HexColor("#D9D9D9"), HexColor("#333333")
    ymap = lambda v: bottom + float(v) * height
    xmap = lambda i: left + (i + .5) * width / 3
    for tick in [0, .2, .4, .6, .8, 1.0]:
        y = ymap(tick)
        d.add(Line(left, y, left + width, y, strokeColor=grey, strokeWidth=.5))
        text(d, left - 7, y - 2.5, f"{tick:.1f}", 7, "end")
    d.add(Line(left, bottom, left, bottom + height, strokeColor=dark, strokeWidth=.7))
    d.add(Line(left, bottom, left + width, bottom, strokeColor=dark, strokeWidth=.7))
    for i, tier in enumerate(TIERS):
        x = xmap(i)
        d.add(Line(x, bottom, x, bottom - 3, strokeColor=dark, strokeWidth=.6))
        text(d, x, bottom - 13, tier.capitalize(), 7)
    text(d, left + width / 2, bottom - 27, "Sensitivity tier (domain proxy)", 8)
    text(d, left, bottom + height + 7, ylabel, 8, "start")
    text(d, left + 6, bottom + 8, panel, 8, "start", True)
    return xmap, ymap


def main() -> None:
    data = load()
    d = Drawing(525, 205)
    d.add(Rect(0, 0, d.width, d.height, fillColor=white, strokeColor=None))
    xmap, ymap = axes(d, 42, 38, 205, 126, "Governing-marker emission rate", "(a) Emission")
    for model in MODELS:
        values = [data[(model, tier)] for tier in TIERS]
        points = []
        for i, row in enumerate(values):
            x, y = xmap(i), ymap(float(row["primary_emission_rate"]))
            lo, hi = ymap(float(row["primary_emission_ci_low"])), ymap(float(row["primary_emission_ci_high"]))
            points.extend([x, y])
            d.add(Line(x, lo, x, hi, strokeColor=COLORS[model], strokeWidth=.8))
            d.add(Line(x - 2.5, lo, x + 2.5, lo, strokeColor=COLORS[model], strokeWidth=.8))
            d.add(Line(x - 2.5, hi, x + 2.5, hi, strokeColor=COLORS[model], strokeWidth=.8))
        line = PolyLine(points, strokeColor=COLORS[model], strokeWidth=1.5, fillColor=None)
        if model.startswith("deepseek"):
            line.strokeDashArray = [5, 3]
        d.add(line)
        for i in range(0, len(points), 2):
            x, y = points[i], points[i + 1]
            if model == "gpt-5-mini":
                d.add(Circle(x, y, 3, fillColor=COLORS[model], strokeColor=COLORS[model]))
            else:
                d.add(Rect(x - 3, y - 3, 6, 6, fillColor=COLORS[model], strokeColor=COLORS[model]))

    xmap2, ymap2 = axes(d, 310, 38, 205, 126, "Strength mix among governing emissions", "(b) Strength")
    strength_colors = {"L1": HexColor("#B3CDE3"), "L2": HexColor("#6497B1"), "L3": HexColor("#005B96")}
    bar_w = 24
    for i, tier in enumerate(TIERS):
        for offset, model in [(-14, "gpt-5-mini"), (14, "deepseek-r1:32b")]:
            row = data[(model, tier)]
            x = xmap2(i) + offset - bar_w / 2
            n = int(row["governing_emissions"])
            short = "G" if model == "gpt-5-mini" else "D"
            if n == 0:
                text(d, x + bar_w / 2, ymap2(.04), f"{short} n=0", 6)
                continue
            bottom = ymap2(0)
            for key, label in [("L1_fraction", "L1"), ("L2_fraction", "L2"), ("L3_fraction", "L3")]:
                value = float(row[key] or 0)
                height = value * 126
                if height:
                    d.add(Rect(x, bottom, bar_w, height, fillColor=strength_colors[label],
                               strokeColor=COLORS[model], strokeWidth=1.1))
                bottom += height
            label_y = ymap2(1.0) + (4 if model == "gpt-5-mini" else 11)
            text(d, x + bar_w / 2, label_y, f"{short} n={n}", 6)

    # Shared legends, dual-encoded for model and strength.
    for idx, model in enumerate(MODELS):
        x, y = 70 + idx * 135, 188
        line = Line(x, y, x + 20, y, strokeColor=COLORS[model], strokeWidth=1.5)
        if model.startswith("deepseek"): line.strokeDashArray = [5, 3]
        d.add(line)
        if model == "gpt-5-mini": d.add(Circle(x + 10, y, 3, fillColor=COLORS[model], strokeColor=COLORS[model]))
        else: d.add(Rect(x + 7, y - 3, 6, 6, fillColor=COLORS[model], strokeColor=COLORS[model]))
        text(d, x + 27, y - 2.5, LABELS[model], 7, "start")
    for idx, label in enumerate(["L1", "L2", "L3"]):
        x, y = 355 + idx * 48, 188
        d.add(Rect(x, y - 4, 9, 9, fillColor=strength_colors[label], strokeColor=None))
        text(d, x + 13, y - 2.5, label, 7, "start")

    renderPDF.drawToFile(d, str(OUTPUT.with_suffix(".pdf")))
    renderSVG.drawToFile(d, str(OUTPUT.with_suffix(".svg")))


if __name__ == "__main__":
    main()
