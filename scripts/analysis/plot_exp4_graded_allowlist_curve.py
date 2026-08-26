#!/usr/bin/env python3
"""Plot the EXP-4 graded noisy-allowlist privacy--utility curve."""

from __future__ import annotations

import csv
from pathlib import Path

from reportlab.graphics import renderPDF, renderSVG
from reportlab.graphics.shapes import Circle, Drawing, Line, PolyLine, Rect, String
from reportlab.lib.colors import HexColor, white


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "rebuttal" / "summary_exp4_graded_allowlist_privacy_utility_curve_20260713.csv"
OUTPUT_STEM = ROOT / "rebuttal" / "exp4_graded_allowlist_privacy_utility_curve"

STYLES = {
    "GPT-5-mini": {"color": HexColor("#0072B2"), "marker": "circle", "dash": None},
    "DeepSeek-R1-32B": {"color": HexColor("#D55E00"), "marker": "square", "dash": [5, 3]},
}


def load_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with INPUT.open(newline="") as handle:
        for raw in csv.DictReader(handle):
            rows.append(
                {
                    "model": raw["model"],
                    "rate": float(raw["nominal_error_rate_pct"]),
                    "leak": float(raw["exact_leak_rate"]),
                    "leak_low": float(raw["leak_ci_low"]),
                    "leak_high": float(raw["leak_ci_high"]),
                    "utility": float(raw["task_success_mean"]),
                    "utility_low": float(raw["utility_ci_low"]),
                    "utility_high": float(raw["utility_ci_high"]),
                }
            )
    return rows


def add_text(d: Drawing, x: float, y: float, text: str, *, size: float = 8,
             anchor: str = "middle", bold: bool = False) -> None:
    d.add(String(x, y, text, fontName="Helvetica-Bold" if bold else "Helvetica",
                 fontSize=size, textAnchor=anchor, fillColor=HexColor("#222222")))


def draw_panel(d: Drawing, rows: list[dict[str, object]], left: float, bottom: float,
               width: float, height: float, metric: str, low_key: str, high_key: str,
               ymin: float, ymax: float, yticks: list[float], panel_label: str,
               ylabel: str, label_at_top: bool) -> None:
    grey = HexColor("#D9D9D9")
    dark = HexColor("#333333")
    xmap = lambda v: left + (float(v) / 30.0) * width
    ymap = lambda v: bottom + ((float(v) - ymin) / (ymax - ymin)) * height

    for tick in yticks:
        y = ymap(tick)
        d.add(Line(left, y, left + width, y, strokeColor=grey, strokeWidth=0.5))
        add_text(d, left - 7, y - 2.5, f"{tick:.2f}" if metric == "utility" else f"{tick:.1f}",
                 size=7, anchor="end")
    d.add(Line(left, bottom, left, bottom + height, strokeColor=dark, strokeWidth=0.7))
    d.add(Line(left, bottom, left + width, bottom, strokeColor=dark, strokeWidth=0.7))

    for tick in [0, 5, 10, 20, 30]:
        x = xmap(tick)
        d.add(Line(x, bottom, x, bottom - 3, strokeColor=dark, strokeWidth=0.6))
        add_text(d, x, bottom - 13, str(tick), size=7)
    add_text(d, left + width / 2, bottom - 27, "Allowlist label error rate (%)", size=8)
    add_text(d, left, bottom + height + 6, ylabel, size=8, anchor="start")

    for model, style in STYLES.items():
        subset = sorted((r for r in rows if r["model"] == model), key=lambda r: r["rate"])
        points = []
        for r in subset:
            x, y = xmap(r["rate"]), ymap(r[metric])
            lo, hi = ymap(r[low_key]), ymap(r[high_key])
            d.add(Line(x, lo, x, hi, strokeColor=style["color"], strokeWidth=0.8))
            d.add(Line(x - 2.5, lo, x + 2.5, lo, strokeColor=style["color"], strokeWidth=0.8))
            d.add(Line(x - 2.5, hi, x + 2.5, hi, strokeColor=style["color"], strokeWidth=0.8))
            points.extend([x, y])
        line = PolyLine(points, strokeColor=style["color"], strokeWidth=1.5, fillColor=None)
        if style["dash"]:
            line.strokeDashArray = style["dash"]
        d.add(line)
        for i in range(0, len(points), 2):
            x, y = points[i], points[i + 1]
            if style["marker"] == "circle":
                d.add(Circle(x, y, 3, fillColor=style["color"], strokeColor=style["color"]))
            else:
                d.add(Rect(x - 3, y - 3, 6, 6, fillColor=style["color"], strokeColor=style["color"]))

    label_y = bottom + height - 11 if label_at_top else bottom + 8
    add_text(d, left + 7, label_y, panel_label, size=8, anchor="start", bold=True)


def add_legend(d: Drawing, x: float, y: float) -> None:
    for offset, (model, style) in enumerate(STYLES.items()):
        yy = y - offset * 14
        line = Line(x, yy, x + 20, yy, strokeColor=style["color"], strokeWidth=1.5)
        if style["dash"]:
            line.strokeDashArray = style["dash"]
        d.add(line)
        if style["marker"] == "circle":
            d.add(Circle(x + 10, yy, 3, fillColor=style["color"], strokeColor=style["color"]))
        else:
            d.add(Rect(x + 7, yy - 3, 6, 6, fillColor=style["color"], strokeColor=style["color"]))
        add_text(d, x + 27, yy - 2.5, model, size=7, anchor="start")


def main() -> None:
    rows = load_rows()
    d = Drawing(475, 175)
    d.add(Rect(0, 0, d.width, d.height, fillColor=white, strokeColor=None))
    draw_panel(d, rows, 48, 39, 170, 116, "leak", "leak_low", "leak_high",
               -0.02, 0.50, [0, .1, .2, .3, .4, .5], "(a) Privacy risk",
               "Exact leakage rate", True)
    draw_panel(d, rows, 293, 39, 170, 116, "utility", "utility_low", "utility_high",
               .75, 1.02, [.75, .80, .85, .90, .95, 1.00], "(b) Utility",
               "Judged task success", False)
    add_legend(d, 61, 128)
    renderPDF.drawToFile(d, str(OUTPUT_STEM.with_suffix(".pdf")))
    renderSVG.drawToFile(d, str(OUTPUT_STEM.with_suffix(".svg")))


if __name__ == "__main__":
    main()
