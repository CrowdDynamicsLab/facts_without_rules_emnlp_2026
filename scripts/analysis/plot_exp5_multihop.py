#!/usr/bin/env python3
"""Create the paper-ready EXP-5 multi-hop survival/leakage figure."""

from __future__ import annotations

import csv
from pathlib import Path

from reportlab.graphics import renderPDF, renderSVG
from reportlab.graphics.shapes import Circle, Drawing, Line, PolyLine, Rect, String
from reportlab.lib.colors import HexColor, white


ROOT = Path(__file__).resolve().parents[1]
SURVIVAL = ROOT / "rebuttal" / "summary_exp5_multihop_survival.csv"
LEAKAGE = ROOT / "rebuttal" / "summary_exp5_multihop_leakage.csv"
OUTPUT = ROOT / "rebuttal" / "exp5_multihop_survival_leakage"

MODEL_COLORS = {"GPT-5-mini": HexColor("#0072B2"), "DeepSeek-R1-32B": HexColor("#D55E00")}
VARIANT_STYLE = {
    "full_replay": {"dash": None, "marker": "circle", "label": "full replay"},
    "partial_recall50": {"dash": [5, 3], "marker": "square", "label": "partial recall"},
}


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def text(d: Drawing, x: float, y: float, value: str, size: float = 8,
         anchor: str = "middle", bold: bool = False) -> None:
    d.add(String(x, y, value, fontName="Helvetica-Bold" if bold else "Helvetica",
                 fontSize=size, textAnchor=anchor, fillColor=HexColor("#222222")))


def series_data(survival: list[dict], leakage: list[dict], model: str, variant: str,
                metric: str) -> list[tuple[float, float, float, float]]:
    if metric in {"sigma_b", "sigma_op"}:
        table = survival
        low, high = f"{metric}_ci_low", f"{metric}_ci_high"
    else:
        table = [r for r in leakage if r["pressure"] == "pooled"]
        low, high = "semantic_ci_low", "semantic_ci_high"
    selected = []
    for row in table:
        if row["model"] != model:
            continue
        row_variant = row["variant"]
        hop = int(row["hop"])
        if hop == 1 and variant == "partial_recall50" and row_variant == "full_replay":
            pass
        elif row_variant != variant:
            continue
        value_key = metric if metric in {"sigma_b", "sigma_op"} else "semantic_rate"
        selected.append((hop, float(row[value_key]), float(row[low]), float(row[high])))
    return sorted(selected)


def panel(d: Drawing, left: float, bottom: float, width: float, height: float,
          survival: list[dict], leakage: list[dict], metric: str, label: str, ylabel: str) -> None:
    grey, dark = HexColor("#D9D9D9"), HexColor("#333333")
    xmap = lambda hop: left + (hop - 1) * width / 2
    ymap = lambda value: bottom + value * height
    for tick in [0, .2, .4, .6, .8, 1.0]:
        y = ymap(tick)
        d.add(Line(left, y, left + width, y, strokeColor=grey, strokeWidth=.5))
        text(d, left - 7, y - 2.5, f"{tick:.1f}", 7, "end")
    d.add(Line(left, bottom, left, bottom + height, strokeColor=dark, strokeWidth=.7))
    d.add(Line(left, bottom, left + width, bottom, strokeColor=dark, strokeWidth=.7))
    for hop in [1, 2, 3]:
        x = xmap(hop)
        d.add(Line(x, bottom, x, bottom - 3, strokeColor=dark, strokeWidth=.6))
        text(d, x, bottom - 13, str(hop), 7)
    text(d, left + width / 2, bottom - 27, "Handoff hop", 8)
    text(d, left, bottom + height + 7, ylabel, 8, "start")
    text(d, left + 6, bottom + 8, label, 8, "start", True)

    for model, color in MODEL_COLORS.items():
        for variant, style in VARIANT_STYLE.items():
            values = series_data(survival, leakage, model, variant, metric)
            points = []
            for hop, value, lo, hi in values:
                x, y = xmap(hop), ymap(value)
                points.extend([x, y])
                d.add(Line(x, ymap(lo), x, ymap(hi), strokeColor=color, strokeWidth=.7))
                d.add(Line(x - 2, ymap(lo), x + 2, ymap(lo), strokeColor=color, strokeWidth=.7))
                d.add(Line(x - 2, ymap(hi), x + 2, ymap(hi), strokeColor=color, strokeWidth=.7))
            line = PolyLine(points, strokeColor=color, strokeWidth=1.4, fillColor=None)
            if style["dash"]:
                line.strokeDashArray = style["dash"]
            d.add(line)
            for i in range(0, len(points), 2):
                x, y = points[i], points[i + 1]
                if style["marker"] == "circle":
                    d.add(Circle(x, y, 2.8, fillColor=color, strokeColor=color))
                else:
                    d.add(Rect(x - 2.8, y - 2.8, 5.6, 5.6, fillColor=color, strokeColor=color))


def legend(d: Drawing, x: float, y: float) -> None:
    offset = 0
    for model, color in MODEL_COLORS.items():
        for variant, style in VARIANT_STYLE.items():
            xx = x + offset * 113
            line = Line(xx, y, xx + 20, y, strokeColor=color, strokeWidth=1.4)
            if style["dash"]:
                line.strokeDashArray = style["dash"]
            d.add(line)
            if style["marker"] == "circle":
                d.add(Circle(xx + 10, y, 2.8, fillColor=color, strokeColor=color))
            else:
                d.add(Rect(xx + 7.2, y - 2.8, 5.6, 5.6, fillColor=color, strokeColor=color))
            short = "GPT" if model == "GPT-5-mini" else "DeepSeek"
            text(d, xx + 26, y - 2.5, f"{short}, {style['label']}", 7, "start")
            offset += 1


def main() -> None:
    survival, leakage = rows(SURVIVAL), rows(LEAKAGE)
    d = Drawing(720, 205)
    d.add(Rect(0, 0, d.width, d.height, fillColor=white, strokeColor=None))
    legend(d, 112, 188)
    panel(d, 42, 38, 180, 126, survival, leakage, "sigma_b", "(a) Boundary survival", "Boundary survival (sigma_b)")
    panel(d, 286, 38, 180, 126, survival, leakage, "sigma_op", "(b) Operational survival", "Operational survival (sigma_op)")
    panel(d, 530, 38, 180, 126, survival, leakage, "semantic", "(c) Final leakage", "Semantic leakage rate")
    renderPDF.drawToFile(d, str(OUTPUT.with_suffix(".pdf")))
    renderSVG.drawToFile(d, str(OUTPUT.with_suffix(".svg")))


if __name__ == "__main__":
    main()
