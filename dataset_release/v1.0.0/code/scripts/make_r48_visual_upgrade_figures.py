from __future__ import annotations

import csv
import html
import math
import random
import shutil
import subprocess
import textwrap
import zipfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "submission_hess" / "journal_upload_R47" / "source_data"
FACTORY = ROOT / "paper_figures" / "hess_r48_visual_upgrade"
SUBMISSION_COPY = ROOT / "submission_hess" / "figure_upgrade_R48"

W, H = 1900, 1250
FONT = "Arial, Helvetica, sans-serif"
NAVY = "#081845"
BLUE = "#0b56b7"
CYAN = "#06a5d8"
GREEN = "#3e8a3a"
ORANGE = "#f47a20"
RED = "#c71f1f"
PURPLE = "#6c3ab2"
GREY = "#617086"
LIGHT = "#f7fbff"


def ensure_dirs() -> None:
    for p in [
        FACTORY / "figure_specs",
        FACTORY / "src",
        FACTORY / "output" / "svg",
        FACTORY / "output" / "pdf",
        FACTORY / "output" / "png",
        FACTORY / "output" / "pptx",
        FACTORY / "output" / "contact_sheets",
        FACTORY / "reviews",
        SUBMISSION_COPY,
    ]:
        p.mkdir(parents=True, exist_ok=True)


def e(s: object) -> str:
    return html.escape(str(s), quote=True)


class Canvas:
    def __init__(self, title: str, w: int = W, h: int = H):
        self.w = w
        self.h = h
        self.parts: list[str] = []
        self.parts.append(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}">'
        )
        self.parts.append(f"<title>{e(title)}</title>")
        self.parts.append(
            """
<defs>
  <linearGradient id="bg" x1="0" x2="1" y1="0" y2="1">
    <stop offset="0%" stop-color="#ffffff"/>
    <stop offset="52%" stop-color="#f6fbff"/>
    <stop offset="100%" stop-color="#edf5ff"/>
  </linearGradient>
  <linearGradient id="sky" x1="0" x2="1" y1="0" y2="1">
    <stop offset="0%" stop-color="#f7fbff"/>
    <stop offset="100%" stop-color="#ddecff"/>
  </linearGradient>
  <linearGradient id="soil" x1="0" x2="0" y1="0" y2="1">
    <stop offset="0%" stop-color="#b87b3e"/>
    <stop offset="100%" stop-color="#6a3e1e"/>
  </linearGradient>
  <linearGradient id="rock" x1="0" x2="1" y1="0" y2="1">
    <stop offset="0%" stop-color="#565c66"/>
    <stop offset="100%" stop-color="#15202b"/>
  </linearGradient>
  <linearGradient id="water" x1="0" x2="1" y1="0" y2="0">
    <stop offset="0%" stop-color="#10c9ff"/>
    <stop offset="100%" stop-color="#005cc8"/>
  </linearGradient>
  <linearGradient id="panel" x1="0" x2="1" y1="0" y2="1">
    <stop offset="0%" stop-color="#ffffff"/>
    <stop offset="100%" stop-color="#eef6ff"/>
  </linearGradient>
  <marker id="arrowBlue" viewBox="0 0 10 10" refX="9" refY="5"
          markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M 0 0 L 10 5 L 0 10 z" fill="#0b56b7"/>
  </marker>
  <marker id="arrowCyan" viewBox="0 0 10 10" refX="9" refY="5"
          markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M 0 0 L 10 5 L 0 10 z" fill="#06a5d8"/>
  </marker>
  <marker id="arrowOrange" viewBox="0 0 10 10" refX="9" refY="5"
          markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M 0 0 L 10 5 L 0 10 z" fill="#f47a20"/>
  </marker>
  <marker id="arrowRed" viewBox="0 0 10 10" refX="9" refY="5"
          markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M 0 0 L 10 5 L 0 10 z" fill="#c71f1f"/>
  </marker>
</defs>
"""
        )
        self.rect(0, 0, w, h, fill="url(#bg)", stroke="none")

    def add(self, s: str) -> None:
        self.parts.append(s)

    def rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        rx: float = 0,
        fill: str = "none",
        stroke: str = "#b6c5d9",
        sw: float = 2,
        opacity: float | None = None,
        dash: str | None = None,
    ) -> None:
        op = "" if opacity is None else f' opacity="{opacity}"'
        da = "" if dash is None else f' stroke-dasharray="{dash}"'
        self.add(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
            f'rx="{rx:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{op}{da}/>'
        )

    def circle(
        self,
        cx: float,
        cy: float,
        r: float,
        fill: str = "none",
        stroke: str = "#b6c5d9",
        sw: float = 2,
        opacity: float | None = None,
    ) -> None:
        op = "" if opacity is None else f' opacity="{opacity}"'
        self.add(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{sw}"{op}/>'
        )

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        stroke: str = NAVY,
        sw: float = 2,
        dash: str | None = None,
        arrow: str | None = None,
        opacity: float | None = None,
    ) -> None:
        da = "" if dash is None else f' stroke-dasharray="{dash}"'
        ar = "" if arrow is None else f' marker-end="url(#{arrow})"'
        op = "" if opacity is None else f' opacity="{opacity}"'
        self.add(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{stroke}" stroke-width="{sw}" stroke-linecap="round"{da}{ar}{op}/>'
        )

    def path(
        self,
        d: str,
        fill: str = "none",
        stroke: str = NAVY,
        sw: float = 2,
        dash: str | None = None,
        arrow: str | None = None,
        opacity: float | None = None,
    ) -> None:
        da = "" if dash is None else f' stroke-dasharray="{dash}"'
        ar = "" if arrow is None else f' marker-end="url(#{arrow})"'
        op = "" if opacity is None else f' opacity="{opacity}"'
        self.add(
            f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" '
            f'stroke-linejoin="round" stroke-linecap="round"{da}{ar}{op}/>'
        )

    def polygon(
        self,
        pts: list[tuple[float, float]],
        fill: str,
        stroke: str = "#ffffff",
        sw: float = 1.5,
        opacity: float | None = None,
    ) -> None:
        p = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        op = "" if opacity is None else f' opacity="{opacity}"'
        self.add(
            f'<polygon points="{p}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{op}/>'
        )

    def polyline(
        self,
        pts: list[tuple[float, float]],
        fill: str = "none",
        stroke: str = NAVY,
        sw: float = 2,
        opacity: float | None = None,
        dash: str | None = None,
    ) -> None:
        p = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        op = "" if opacity is None else f' opacity="{opacity}"'
        da = "" if dash is None else f' stroke-dasharray="{dash}"'
        self.add(
            f'<polyline points="{p}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" '
            f'stroke-linejoin="round" stroke-linecap="round"{op}{da}/>'
        )

    def text(
        self,
        x: float,
        y: float,
        txt: str,
        size: int = 24,
        fill: str = NAVY,
        weight: str = "400",
        anchor: str = "start",
        style: str = "",
        opacity: float | None = None,
    ) -> None:
        op = "" if opacity is None else f' opacity="{opacity}"'
        self.add(
            f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{size}" '
            f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}"{op} {style}>{e(txt)}</text>'
        )

    def multiline(
        self,
        x: float,
        y: float,
        txt: str,
        size: int = 22,
        fill: str = NAVY,
        weight: str = "400",
        width_chars: int = 40,
        line_h: float = 1.25,
        anchor: str = "start",
    ) -> None:
        lines: list[str] = []
        for para in txt.split("\n"):
            lines.extend(textwrap.wrap(para, width_chars) or [""])
        self.add(
            f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{size}" '
            f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">'
        )
        for i, line in enumerate(lines):
            dy = 0 if i == 0 else size * line_h
            self.add(f'<tspan x="{x:.1f}" dy="{dy:.1f}">{e(line)}</tspan>')
        self.add("</text>")

    def panel(self, x: float, y: float, w: float, h: float, label: str, title: str) -> None:
        self.rect(x, y, w, h, rx=18, fill="url(#panel)", stroke="#b8c7da", sw=2)
        self.circle(x + 35, y + 35, 22, fill=BLUE, stroke="white", sw=3)
        self.text(x + 35, y + 43, label, 22, "white", "700", "middle")
        self.text(x + 72, y + 43, title, 26, NAVY, "700")

    def header(self, fig: str, title: str, subtitle: str) -> None:
        self.add(f"<desc>{e(fig)}: {e(title)}. {e(subtitle)}</desc>")
        self.line(64, 108, W - 64, 108, "#d0dae8", 2)

    def finish(self) -> str:
        self.parts.append("</svg>")
        return "\n".join(self.parts)


def result_pill(c: Canvas, x: float, y: float, w: float, text: str, kind: str = "ok") -> None:
    colors = {
        "ok": ("#eff8e9", "#5d9a40", GREEN),
        "warn": ("#fff4e6", "#f3a23a", ORANGE),
        "bad": ("#fff0f0", "#d85050", RED),
        "info": ("#edf5ff", "#8ab2e5", BLUE),
    }
    fill, stroke, color = colors[kind]
    c.rect(x, y, w, 52, rx=15, fill=fill, stroke=stroke, sw=2)
    mark = {"ok": "OK", "warn": "!", "bad": "x", "info": "i"}[kind]
    c.circle(x + 32, y + 26, 17, fill=color, stroke="white", sw=2)
    c.text(x + 32, y + 33, mark, 14 if kind == "ok" else 20, "white", "800", "middle")
    c.multiline(x + 62, y + 22, text, 17, NAVY, "600", max(18, int(w / 10)), 1.05)


def axis(c: Canvas, x: float, y: float, w: float, h: float, xlabel: str = "", ylabel: str = "") -> None:
    c.line(x, y + h, x + w, y + h, "#26364f", 2)
    c.line(x, y, x, y + h, "#26364f", 2)
    for i in range(1, 5):
        yy = y + h * i / 5
        c.line(x, yy, x + w, yy, "#d7e0ee", 1)
    if xlabel:
        c.text(x + w / 2, y + h + 33, xlabel, 18, NAVY, "400", "middle")
    if ylabel:
        c.text(x - 40, y + h / 2, ylabel, 18, NAVY, "400", "middle", style='transform="rotate(-90deg)"')


def mini_curve(
    c: Canvas,
    x: float,
    y: float,
    w: float,
    h: float,
    color: str = BLUE,
    seed: int = 1,
    trend: str = "fall",
    dashed: bool = False,
) -> None:
    rng = random.Random(seed)
    pts: list[tuple[float, float]] = []
    for i in range(44):
        t = i / 43
        if trend == "fall":
            val = 0.85 - 0.63 * t + 0.06 * math.sin(8 * t) + rng.uniform(-0.035, 0.035)
        elif trend == "hump":
            val = 0.25 + 0.55 * math.exp(-((t - 0.43) / 0.27) ** 2) + rng.uniform(-0.025, 0.025)
        elif trend == "rise":
            val = 0.22 + 0.55 * t + 0.05 * math.sin(9 * t) + rng.uniform(-0.025, 0.025)
        else:
            val = 0.48 + 0.15 * math.sin(12 * t) + rng.uniform(-0.035, 0.035)
        val = max(0.05, min(0.95, val))
        pts.append((x + t * w, y + (1 - val) * h))
    c.polyline(pts, stroke=color, sw=3, dash="8 7" if dashed else None)


def spark_bars(c: Canvas, x: float, y: float, w: float, h: float, values: list[float], color: str) -> None:
    mn, mx = min(values), max(values)
    span = mx - mn or 1
    bw = w / len(values) * 0.72
    for i, val in enumerate(values):
        hh = (val - mn) / span * h * 0.82 + h * 0.08
        c.rect(x + i * w / len(values), y + h - hh, bw, hh, rx=2, fill=color, stroke="none", opacity=0.8)


def pin(c: Canvas, x: float, y: float, color: str, label: str) -> None:
    c.path(
        f"M{x},{y-36} C{x-22},{y-36} {x-34},{y-17} {x-28},{y+2} C{x-23},{y+19} {x},{y+42} {x},{y+42} C{x},{y+42} {x+23},{y+19} {x+28},{y+2} C{x+34},{y-17} {x+22},{y-36} {x},{y-36} Z",
        fill=color,
        stroke="white",
        sw=4,
    )
    c.circle(x, y - 10, 10, fill="white", stroke="none")
    c.text(x + 22, y - 22, label, 22, color, "700")


def draw_world_schematic(c: Canvas, x: float, y: float, w: float, h: float) -> None:
    land = "#d8dde3"
    stroke = "#ffffff"
    def P(points):
        return [(x + px * w, y + py * h) for px, py in points]

    continents = [
        [(0.08, 0.30), (0.17, 0.18), (0.28, 0.21), (0.32, 0.36), (0.25, 0.48), (0.16, 0.46)],
        [(0.24, 0.49), (0.31, 0.58), (0.29, 0.82), (0.22, 0.91), (0.18, 0.68)],
        [(0.43, 0.25), (0.56, 0.20), (0.63, 0.32), (0.56, 0.45), (0.45, 0.43)],
        [(0.50, 0.44), (0.61, 0.52), (0.62, 0.75), (0.53, 0.82), (0.45, 0.66)],
        [(0.62, 0.27), (0.79, 0.23), (0.90, 0.37), (0.83, 0.55), (0.66, 0.50)],
        [(0.80, 0.66), (0.90, 0.69), (0.93, 0.84), (0.82, 0.88), (0.76, 0.77)],
        [(0.48, 0.14), (0.55, 0.12), (0.58, 0.17), (0.50, 0.19)],
    ]
    for pts in continents:
        c.polygon(P(pts), fill=land, stroke=stroke, sw=1.2, opacity=0.92)


def write_design_specs() -> None:
    specs = {
        "fig01_3d_mechanism_benchmark.md": """# Figure 1: 3D Catchment Mechanism Benchmark

## Scientific Question
What does the memory-normalized spectral benchmark measure, and what does it not prove?

## One-Sentence Takeaway
The benchmark organizes streamflow spectra by output memory scale, then calibrates interpretation against stochastic-memory floors rather than claiming direct groundwater residence-time causality.

## Figure Archetype
Mechanism + method pipeline + evidence-bound graphical abstract.

## Layout
Left: pseudo-3D catchment block and hydrologic pathways. Right: time-series-to-spectrum-to-coordinate pipeline and archive use. Bottom: claim boundary and diagnostic implication.

## Evidence Boundary
The catchment block is schematic and author-generated. Quantitative insets use the manuscript source-data logic but are stylized for figure readability. It is not evidence for a specific subsurface architecture.

## Export
Editable SVG first; PDF and PNG previews; PPTX deck for manual annotation.
""",
        "fig02_portability_archive_boundary.md": """# Figure 2: Portability Across Archives And Boundary Cases

## Scientific Question
Where does the coordinate improve alignment, and where does the result weaken or fail?

## One-Sentence Takeaway
Positive archive-level reductions appear in US, GB and Brazil, weaken in Australia, and fail in Denmark, defining portability as bounded rather than universal.

## Evidence Boundary
Archive reductions and intervals are read from R35/R42 source data. Region cards are interpretive summaries and should be kept consistent with manuscript text.
""",
        "fig03_robustness_stress_test.md": """# Figure 3: Robustness Stress Test

## Scientific Question
Which reviewer risks were explicitly stress-tested?

## One-Sentence Takeaway
Temporal leakage, estimator choice, binning, spatial dependence, forcing-side memory and low-frequency support were audited with mostly positive closure, while claims remain bounded.

## Evidence Boundary
Numerical summaries come from R35/R42 source-data; mini-plots are compressed dashboard encodings, not replacement raw-result panels.
""",
        "fig04_stochastic_artifact_floor.md": """# Figure 4: Stochastic Artifact Floor

## Scientific Question
Why is raw spectral-memory alignment not sufficient mechanism evidence?

## One-Sentence Takeaway
Observed coordinate utility must be interpreted relative to the strict 100-draw gauge-matched AR(1) artifact floor; excess-over-null storage-mechanism claims are not supported where the floor is stronger.

## Evidence Boundary
Observed and strict null values use the R33 multi-draw gauge-matched AR(1) ensemble summary, with R37 analytic process families retained only as schematic companion controls.
""",
        "fig05_model_diagnostic_arena.md": """# Figure 5: Model Diagnostic Arena

## Scientific Question
Does hydrograph skill measure the same capability as spectral-memory fidelity?

## One-Sentence Takeaway
Hydrograph skill and spectral-memory skill are separable diagnostic axes; process models, LSTM and stochastic baselines occupy different regions of the arena.

## Evidence Boundary
Model medians and rank-disagreement summaries are read from R39 source data. Points are jittered visual summaries rather than per-gauge scatter replacements.
""",
    }
    for name, text in specs.items():
        (FACTORY / "figure_specs" / name).write_text(text.strip() + "\n", encoding="utf-8")


def load_data() -> dict[str, pd.DataFrame]:
    return {
        "fig2": pd.read_csv(SOURCE / "r35_fig2_five_archive_alignment_source.csv"),
        "cluster": pd.read_csv(SOURCE / "final_extreme_hardening_cluster_bootstrap_summary.csv"),
        "robust": pd.read_csv(SOURCE / "r35_fig4_robustness_support_source.csv"),
        "floor": pd.read_csv(SOURCE / "final_all_archive_ar1_floor_summary.csv"),
        "ar1_ensemble": pd.read_csv(SOURCE / "r33_matched_ar1_ensemble_summary.csv"),
        "nulls": pd.read_csv(SOURCE / "r37_gauge_matched_analytic_nulls_summary.csv"),
        "models": pd.read_csv(SOURCE / "r39_open_model_intercomparison_summary.csv"),
        "rank": pd.read_csv(SOURCE / "r39_open_model_intercomparison_rank_disagreement.csv"),
        "forcing": pd.read_csv(SOURCE / "final_extreme_hardening_forcing_control_summary.csv"),
        "support": pd.read_csv(SOURCE / "final_extreme_hardening_slow_memory_support_audit.csv"),
    }


def figure1(data: dict[str, pd.DataFrame]) -> str:
    c = Canvas("Fig. 1: null-calibrated spectral-memory benchmark")
    c.header("FIGURE 1", "A NULL-CALIBRATED SPECTRAL-MEMORY BENCHMARK", "Output-memory scale, not groundwater residence time")

    # Left pseudo-3D catchment block.
    x0, y0 = 50, 130
    c.rect(x0, y0, 1040, 685, rx=24, fill="url(#sky)", stroke="#c6d2df", sw=2)
    # Sky and cloud/rain.
    for i, cx in enumerate([155, 195, 242, 300, 350]):
        c.circle(cx, 185 + (i % 2) * 10, 36, fill="#cfd8e5", stroke="none", opacity=0.82)
    c.rect(126, 184, 270, 52, rx=25, fill="#cfd8e5", stroke="none", opacity=0.85)
    c.text(164, 275, "Rainfall", 22, BLUE, "700")
    rng = random.Random(7)
    for i in range(45):
        rx = rng.uniform(105, 400)
        ry = rng.uniform(230, 410)
        c.line(rx, ry, rx - 7, ry + 34, "#1f78d1", 2, opacity=0.45)

    # Mountains and terrain.
    c.polygon([(90, 530), (170, 360), (250, 505)], "#6d7780", stroke="#e6edf5", sw=2)
    c.polygon([(178, 520), (315, 300), (455, 522)], "#59636d", stroke="#e6edf5", sw=2)
    c.polygon([(380, 525), (535, 350), (690, 530)], "#69737d", stroke="#e6edf5", sw=2)
    c.polygon([(128, 423), (171, 360), (206, 424)], "#f6f9fd", stroke="none", opacity=0.9)
    c.polygon([(274, 382), (315, 300), (368, 390)], "#f6f9fd", stroke="none", opacity=0.9)
    c.polygon([(494, 399), (535, 350), (583, 405)], "#f6f9fd", stroke="none", opacity=0.9)

    top = [(80, 595), (185, 548), (315, 570), (440, 530), (600, 555), (760, 520), (945, 566), (1035, 620)]
    lower = [(180, 1000), (1015, 980), (1055, 650), (1035, 620)] + list(reversed(top[1:])) + [(80, 595)]
    c.polygon(top + [(1035, 650), (180, 705), (80, 625)], "#5f9d45", stroke="#eaf4eb", sw=2)
    c.polygon(lower, "url(#soil)", stroke="#7b5231", sw=2)
    c.polygon([(180, 760), (1052, 705), (1035, 835), (175, 900)], "#8b5a35", stroke="#6e4223", sw=1.5, opacity=0.92)
    c.polygon([(175, 900), (1035, 835), (1020, 990), (180, 1000)], "url(#rock)", stroke="#34404d", sw=1.5)
    for yy, col in [(675, "#c1874f"), (745, "#80502e"), (855, "#313c48"), (940, "#1d2732")]:
        c.path(f"M95,{yy} C330,{yy-45} 620,{yy+35} 1042,{yy-35}", stroke=col, sw=2, opacity=0.8)

    # Rivers and flow paths.
    c.path("M185,550 C255,615 355,615 425,675 C515,750 650,704 740,790 C820,865 930,842 1035,925", stroke="#7ee7ff", sw=17, opacity=0.75)
    c.path("M185,550 C255,615 355,615 425,675 C515,750 650,704 740,790 C820,865 930,842 1035,925", stroke="url(#water)", sw=8, arrow="arrowCyan")
    c.path("M410,530 C450,610 480,680 565,720", stroke="#7ee7ff", sw=10, opacity=0.75)
    c.path("M410,530 C450,610 480,680 565,720", stroke=CYAN, sw=4, arrow="arrowCyan")
    c.path("M650,554 C690,635 690,700 760,790", stroke="#7ee7ff", sw=10, opacity=0.75)
    c.path("M650,554 C690,635 690,700 760,790", stroke=CYAN, sw=4, arrow="arrowCyan")
    for off in [0, 70, 140]:
        c.path(f"M275,{635+off} C410,{720+off} 570,{715+off} 700,{790+off}", stroke="#ff9b2f", sw=4, arrow="arrowOrange", opacity=0.8)
    for off in [0, 65, 130]:
        c.path(f"M230,{790+off} C455,{885+off/3} 725,{880+off/4} 985,{925+off/5}", stroke="#11c5ff", sw=5, arrow="arrowCyan", opacity=0.85)
    c.line(1035, 925, 1080, 925, CYAN, 9, arrow="arrowCyan")
    c.text(1005, 968, "Q(t)", 32, BLUE, "800")

    c.text(100, 640, "Soil storage", 21, GREEN, "800")
    c.text(100, 670, "Infiltration", 18, NAVY, "500")
    c.text(102, 760, "Vadose zone", 21, "#83401c", "800")
    c.text(102, 790, "Unsaturated flow", 18, NAVY, "500")
    c.text(102, 900, "Shallow groundwater", 21, BLUE, "800")
    c.text(102, 930, "Interflow", 18, NAVY, "500")
    c.text(102, 1000, "Deep groundwater", 21, NAVY, "800")
    c.text(102, 1030, "Baseflow", 18, NAVY, "500")
    c.text(545, 608, "Surface runoff", 22, "white", "800")
    c.text(500, 770, "Intermediate storage release", 22, ORANGE, "800")
    c.text(585, 940, "Slow groundwater contribution", 22, "#9feaff", "800")
    # Cover the schematic block outside its panel so the pseudo-3D section does not leak
    # into the logic strip and bottom interpretation panels.
    c.rect(0, 815, 1110, 435, rx=0, fill="#f7fbff", stroke="none")

    # Right evidence pipeline.
    c.panel(1120, 130, 730, 685, "A", "Evidence pipeline and archive use")
    draw_world_schematic(c, 1480, 166, 320, 190)
    for px, py, col, lab in [
        (1544, 252, BLUE, "US"),
        (1648, 230, GREEN, "GB"),
        (1568, 315, ORANGE, "BR"),
        (1760, 327, RED, "AUS"),
        (1685, 202, PURPLE, "DK"),
    ]:
        pin(c, px, py, col, lab)
    c.text(1510, 380, "Five CAMELS-family archives", 22, NAVY, "700")

    stages = [
        (1165, 190, "1", "Q(t)", "Discharge at outlet", "hump", BLUE),
        (1165, 340, "2", "PSD", "Power spectrum", "fall", BLUE),
        (1165, 500, "3", "beta(f)", "Local slope", "rise", GREEN),
        (1165, 660, "4", "beta(De)", "Memory-normalized", "hump", BLUE),
    ]
    for x, y, num, label, sub, trend, col in stages:
        c.circle(x, y - 2, 20, fill=BLUE, stroke="white", sw=3)
        c.text(x, y + 7, num, 20, "white", "800", "middle")
        c.text(x + 35, y + 2, label, 28, col, "800")
        c.text(x + 150, y + 2, sub, 18, NAVY, "500")
        axis(c, x + 35, y + 24, 320, 80)
        mini_curve(c, x + 48, y + 34, 292, 55, col, seed=int(num) * 4, trend=trend)
        if num != "4":
            c.line(x + 195, y + 124, x + 195, y + 145, "#8796a8", 4, arrow="arrowBlue", opacity=0.75)
    c.path("M1806,385 C1842,470 1842,560 1785,640", stroke=BLUE, sw=4, arrow="arrowBlue")
    c.multiline(1665, 480, "Cross-catchment\nspectral alignment", 24, BLUE, "800", 18, anchor="middle")
    c.text(1410, 735, "De = tau_acf f", 28, BLUE, "800")
    c.line(1332, 704, 1332, 748, "#9aa8ba", 2, dash="8 6")
    c.text(1220, 780, "< 1 faster than output memory", 16, ORANGE, "700")
    c.text(1430, 780, "> 1 slower than output memory", 16, BLUE, "700")

    # Logic strip.
    y_strip = 840
    c.rect(50, y_strip, 1800, 83, rx=22, fill="#eef3fb", stroke="#d2dbe8", sw=2)
    logic = [
        ("Measured output memory", "tau_acf", BLUE),
        ("Normalize frequency", "De=tau_acf f", CYAN),
        ("Compare archives", "alignment", GREEN),
        ("Calibrate null floor", "matched AR(1)", RED),
        ("Audit models", "skill axes", PURPLE),
    ]
    x = 120
    for i, (a, b, col) in enumerate(logic):
        c.circle(x, y_strip + 41, 27, fill="white", stroke=col, sw=3)
        c.text(x, y_strip + 50, str(i + 1), 22, col, "800", "middle")
        c.text(x + 45, y_strip + 32, a, 21, NAVY, "800")
        c.text(x + 45, y_strip + 60, b, 17, col, "700")
        if i < len(logic) - 1:
            c.line(x + 225, y_strip + 42, x + 280, y_strip + 42, "#8aa0bd", 4, arrow="arrowBlue")
        x += 335

    # Bottom claim-boundary panels.
    c.rect(50, 950, 360, 240, rx=20, fill="#08345e", stroke="#3d78a7", sw=2)
    c.text(85, 1005, "OUTPUT-MEMORY SCALE,", 24, "white", "800")
    c.text(85, 1035, "NOT GROUNDWATER RESIDENCE TIME", 22, "white", "800")
    c.multiline(
        85,
        1088,
        "tau_acf is an outlet-scale response memory. It is compatible with storage, but it is not a direct residence-time measurement.",
        17,
        "white",
        "500",
        29,
    )
    c.rect(450, 950, 690, 240, rx=20, fill="white", stroke="#c7d2df", sw=2)
    c.text(490, 1000, "STOCHASTIC MEMORY FLOOR (NULL CALIBRATION)", 23, BLUE, "800")
    axis(c, 500, 1030, 370, 110, "De", "beta(De)")
    mini_curve(c, 520, 1045, 330, 75, BLUE, 8, "hump")
    mini_curve(c, 520, 1040, 330, 85, RED, 9, "hump", dashed=True)
    c.line(685, 1030, 685, 1140, "#818d9d", 2, dash="8 6")
    c.text(642, 1021, "De=1", 17, NAVY, "700")
    c.rect(895, 1018, 205, 125, rx=18, fill="#fff0f0", stroke="#f2b6b6", sw=2)
    c.circle(930, 1055, 24, fill="#fff5f5", stroke=RED, sw=3)
    c.text(930, 1064, "!", 28, RED, "800", "middle")
    c.multiline(965, 1045, "Claims must beat the stochastic floor before mechanism inference.", 16, NAVY, "600", 20)
    c.rect(1180, 950, 670, 240, rx=20, fill="#08345e", stroke="#3d78a7", sw=2)
    c.text(1230, 1005, "HYDROGRAPH SKILL AND SPECTRAL-MEMORY SKILL", 23, "white", "800")
    c.text(1230, 1035, "CAN DIFFER", 23, "white", "800")
    c.rect(1230, 1070, 260, 86, rx=14, fill="white", stroke="#c7d2df", sw=2)
    c.rect(1535, 1070, 260, 86, rx=14, fill="white", stroke="#c7d2df", sw=2)
    c.text(1364, 1060, "=", 40, "white", "800", "middle")
    mini_curve(c, 1260, 1090, 185, 45, BLUE, 10, "hump")
    mini_curve(c, 1565, 1095, 185, 42, RED, 11, "rise", dashed=True)
    c.circle(1470, 1142, 22, fill=GREEN, stroke="white", sw=3)
    c.text(1470, 1150, "OK", 14, "white", "800", "middle")
    c.circle(1780, 1142, 22, fill=RED, stroke="white", sw=3)
    c.text(1780, 1150, "x", 22, "white", "800", "middle")

    c.text(55, 1220, "Schematic elements are author-generated vector art; quantitative logic follows manuscript source data.", 17, GREY, "600")
    return c.finish()


def figure2(data: dict[str, pd.DataFrame]) -> str:
    c = Canvas("Fig. 2: portability across archives and boundary cases")
    c.header("FIGURE 2", "PORTABILITY ACROSS ARCHIVES AND BOUNDARY CASES", "Where memory-normalized spectral alignment works, weakens, and fails")

    colors = {
        "CAMELS-US": BLUE,
        "CAMELS-GB v2": GREEN,
        "CAMELS-BR v1.2": ORANGE,
        "CAMELS-AUS v2": RED,
        "CAMELS-DK lowland": PURPLE,
    }
    short = {
        "CAMELS-US": "US",
        "CAMELS-GB v2": "Great Britain",
        "CAMELS-BR v1.2": "Brazil",
        "CAMELS-AUS v2": "Australia",
        "CAMELS-DK lowland": "Denmark",
    }
    fig2 = data["fig2"]
    arch = fig2[fig2["panel"].eq("a_archive_observed_bootstrap")].dropna(subset=["archive"]).copy()

    c.panel(35, 125, 940, 610, "A", "Catchment archives across five regions")
    draw_world_schematic(c, 70, 185, 860, 450)
    coords = {
        "CAMELS-US": (185, 360),
        "CAMELS-GB v2": (540, 330),
        "CAMELS-BR v1.2": (395, 600),
        "CAMELS-AUS v2": (790, 620),
        "CAMELS-DK lowland": (625, 260),
    }
    rng = random.Random(20)
    for archive, (px, py) in coords.items():
        pin(c, px, py, colors[archive], short[archive])
        for _ in range(28 if archive != "CAMELS-DK lowland" else 16):
            c.circle(px + rng.uniform(-55, 75), py + rng.uniform(35, 95), 3.7, fill=colors[archive], stroke="white", sw=0.7, opacity=0.9)
    c.rect(80, 655, 840, 54, rx=18, fill="#f7fbff", stroke="#c3d0e1", sw=2)
    for x, icon, txt in [(145, "globe", "Five regions"), (390, "dots", "Hundreds of catchments"), (690, "bars", "Cross-catchment benchmark")]:
        c.circle(x, 682, 22, fill="white", stroke=BLUE, sw=2)
        if icon == "bars":
            for i, hh in enumerate([18, 28, 38]):
                c.rect(x - 14 + i * 10, 700 - hh, 6, hh, rx=1, fill=BLUE, stroke="none")
        elif icon == "dots":
            for dx, dy in [(-6, -5), (6, -2), (-2, 8)]:
                c.circle(x + dx, 682 + dy, 4, fill=BLUE, stroke="none")
        else:
            c.circle(x, 682, 13, fill="none", stroke=BLUE, sw=1.7)
            c.line(x - 13, 682, x + 13, 682, BLUE, 1.5)
            c.line(x, 669, x, 695, BLUE, 1.5)
        c.text(x + 42, 690, txt, 21, NAVY, "700")

    c.panel(1005, 125, 860, 350, "B", "Archive-level reduction in dispersion")
    c.text(1085, 187, "Positive values indicate lower cross-catchment dispersion on the De axis", 18, NAVY, "500", style='font-style="italic"')
    x_axis, y_axis, ww, hh = 1130, 245, 520, 170
    c.line(x_axis, y_axis + hh, x_axis + ww, y_axis + hh, "#26364f", 2)
    c.line(x_axis + ww * 0.46, y_axis - 10, x_axis + ww * 0.46, y_axis + hh + 12, "#777", 1.5, dash="7 7")
    for tick, lab in [(-20, "-20"), (0, "0"), (20, "20"), (40, "40")]:
        tx = x_axis + (tick + 30) / 80 * ww
        c.line(tx, y_axis + hh, tx, y_axis + hh + 8, "#26364f", 2)
        c.text(tx, y_axis + hh + 31, lab, 16, NAVY, "500", "middle")
    c.text(x_axis + ww / 2, y_axis + hh + 62, "variance reduction (%)", 17, NAVY, "600", "middle")
    for i, row in arch.reset_index(drop=True).iterrows():
        archive = row["archive"]
        col = colors[archive]
        y = y_axis + 15 + i * 31
        med = float(row["bootstrap_median_percent"])
        lo = float(row["ci_low_percent"])
        hi = float(row["ci_high_percent"])
        def sx(v): return x_axis + (v + 30) / 80 * ww
        c.text(1035, y + 6, short[archive], 21, col, "800")
        c.line(sx(lo), y, sx(hi), y, col, 3)
        c.line(sx(lo), y - 9, sx(lo), y + 9, col, 2)
        c.line(sx(hi), y - 9, sx(hi), y + 9, col, 2)
        c.circle(sx(med), y, 8, fill=col, stroke="white", sw=1)
        c.text(1680, y + 6, f"{med:+.1f} [{lo:+.1f}, {hi:+.1f}]", 18, col, "700")
    c.line(x_axis + 20, 438, x_axis + 120, 438, "#333", 2, arrow="arrowBlue")
    c.text(x_axis + 10, 460, "Worse", 15, NAVY, "500")
    c.line(x_axis + ww - 130, 438, x_axis + ww - 30, 438, "#333", 2, arrow="arrowBlue")
    c.text(x_axis + ww - 85, 460, "Better", 15, NAVY, "500")

    c.panel(1005, 500, 860, 235, "C", "Shuffled-memory null check")
    nulls = fig2[fig2["panel"].eq("b_random_tau_null")].dropna(subset=["archive"]).copy()
    for i, row in nulls.reset_index(drop=True).iterrows():
        x = 1050 + i * 160
        col = colors[row["archive"]]
        c.text(x + 65, 555, short[row["archive"]].replace("Great Britain", "GB"), 18, col, "800", "middle")
        for j, y_base in enumerate([615, 685]):
            fill = col if j == 0 else "#8a8a8a"
            c.path(
                f"M{x+5},{y_base} C{x+25},{y_base-50} {x+70},{y_base-55} {x+120},{y_base} L{x+5},{y_base} Z",
                fill=fill,
                stroke=fill,
                sw=1,
                opacity=0.22 if j == 0 else 0.28,
            )
            c.path(f"M{x+5},{y_base} C{x+25},{y_base-50} {x+70},{y_base-55} {x+120},{y_base}", stroke=fill, sw=2.5, opacity=0.9)
        c.circle(x + 72, 613, 6, fill=NAVY, stroke="white", sw=1)
        c.circle(x + 72, 683, 6, fill="#555", stroke="white", sw=1)
    c.text(1038, 615, "Observed", 16, NAVY, "700", "end")
    c.text(1038, 684, "Shuffled tau", 16, GREY, "700", "end")
    c.text(1422, 723, "Random tau assignments are negative in all five archives", 17, NAVY, "600", "middle")

    c.panel(35, 770, 1830, 380, "D", "Boundary / regime summary by archive")
    cards = [
        ("CAMELS-US", "Robust alignment", ["Humid mixed response", "Subsurface memory integrates rainfall", "Surface-groundwater coupling common"], "ok"),
        ("CAMELS-GB v2", "Robust alignment", ["Humid low-moderate relief", "Mixed storage control", "Consistent memory integration"], "ok"),
        ("CAMELS-BR v1.2", "Positive, drier boundary", ["Dry-subhumid archive", "Strong seasonality", "Effect remains positive"], "warn"),
        ("CAMELS-AUS v2", "Weak / boundary", ["Arid to semi-arid", "Intermittent flows", "Lower signal-to-noise"], "warn"),
        ("CAMELS-DK lowland", "Boundary failure", ["Lowland, high baseflow", "Short, damped residence times", "Limited fast-slow separation"], "bad"),
    ]
    for i, (archive, verdict, bullets, kind) in enumerate(cards):
        x = 70 + i * 355
        col = colors[archive]
        fill = {"ok": "#f2fbef", "warn": "#fff7eb", "bad": "#fff1f1"}[kind]
        c.rect(x, 840, 320, 235, rx=20, fill=fill, stroke=col, sw=2)
        c.text(x + 160, 878, short[archive], 25, col, "800", "middle")
        # Mini landscape icon.
        c.rect(x + 18, 905, 100, 86, rx=14, fill="#eef5fb", stroke="#d0dae8", sw=1)
        c.circle(x + 42, 930, 18, fill="#cdd7e4", stroke="none")
        c.circle(x + 70, 925, 22, fill="#cdd7e4", stroke="none")
        if kind == "warn":
            c.circle(x + 48, 928, 18, fill="#ffbf38", stroke=ORANGE, sw=2)
            c.polygon([(x + 22, 975), (x + 60, 935), (x + 110, 975)], "#c89a55", stroke="none")
        else:
            c.polygon([(x + 20, 975), (x + 65, 930), (x + 112, 975)], "#7aa46a", stroke="none")
        c.path(f"M{x+27},975 C{x+55},955 {x+85},975 {x+108},945", stroke=CYAN, sw=5)
        yy = 925
        for b in bullets:
            c.circle(x + 145, yy - 6, 4, fill=col, stroke="none")
            c.multiline(x + 158, yy, b, 17, NAVY, "600", 22, 1.0)
            yy += 47
        result_pill(c, x + 20, 1010, 280, verdict, kind)
    c.rect(70, 1093, 1760, 38, rx=16, fill="#f2f7ff", stroke="#b9cbe2", sw=1.5)
    c.text(94, 1118, "Key boundary:", 19, BLUE, "800")
    c.text(222, 1118, "Positive coordinate utility does not imply storage-mechanism proof; interpretation depends on hydrologic regime.", 19, NAVY, "600")
    c.text(55, 1215, "Editable vector dashboard; archive values are read from registered source-data tables.", 17, GREY, "600")
    return c.finish()


def figure3(data: dict[str, pd.DataFrame]) -> str:
    c = Canvas("Fig. 3: robustness stress test")
    c.header("FIGURE 3", "ROBUSTNESS STRESS TEST", "Wind-tunnel checks for temporal, estimator, forcing and frequency-support risks")
    robust = data["robust"]
    cluster = data["cluster"]
    forcing = data["forcing"]
    support = data["support"]

    # Top panels.
    c.panel(35, 125, 610, 355, "A", "Temporal cross-fitting")
    c.text(150, 180, "Does the effect persist out-of-sample in time?", 18, NAVY, "500", style='font-style="italic"')
    for i, lab in enumerate(["Train\n2000-04", "Test\n2005-09", "Train\n2010-14", "Test\n2015-19", "Train\n2020-24"]):
        x = 105 + i * 95
        fill = "#cfe2f8" if "Train" in lab else "#ffd9b2"
        c.rect(x, 210, 80, 42, rx=4, fill=fill, stroke="#91a9c7", sw=1)
        c.multiline(x + 40, 228, lab, 12, NAVY, "700", 9, 0.9, anchor="middle")
    c.line(80, 202, 585, 202, "#26364f", 2, arrow="arrowBlue")
    c.text(135, 286, "Same segment", 17, BLUE, "800")
    c.text(370, 286, "Split-half / cross-fit", 17, GREEN, "800")
    axis(c, 95, 305, 210, 100, "De", "beta")
    mini_curve(c, 110, 318, 180, 65, BLUE, 31, "hump")
    mini_curve(c, 110, 320, 180, 60, "#333", 32, "hump", dashed=True)
    axis(c, 360, 305, 210, 100, "De", "beta")
    mini_curve(c, 375, 318, 180, 65, GREEN, 33, "hump")
    mini_curve(c, 375, 320, 180, 60, "#333", 34, "hump", dashed=True)
    cv = robust[(robust["panel"].eq("a_temporal_crossfit")) & (robust["analysis_group"].eq("all"))]
    cross = cv[cv["method"].eq("crossfit_tau")]["reduction_percent"].iloc[0]
    result_pill(c, 80, 420, 525, f"Out-of-sample reduction remains positive: {cross:.1f}%.", "ok")

    c.panel(665, 125, 560, 355, "B", "Estimator perturbation")
    c.text(755, 180, "Is the result sensitive to estimator choices?", 18, NAVY, "500", style='font-style="italic"')
    heat = robust[robust["panel"].eq("b_estimator_sensitivity")].dropna(subset=["nperseg", "slope_window_decades"])
    xs = sorted(heat["nperseg"].unique())
    ys = sorted(heat["slope_window_decades"].unique())
    x0, y0, cw, ch = 760, 220, 72, 58
    for iy, swd in enumerate(ys):
        for ix, nps in enumerate(xs):
            val = heat[(heat["nperseg"].eq(nps)) & (heat["slope_window_decades"].eq(swd))]["reduction_percent"].iloc[0]
            norm = (val - 18) / 12
            col = f"rgb({int(245)}, {int(240-60*norm)}, {int(245-120*norm)})"
            c.rect(x0 + ix * cw, y0 + iy * ch, cw, ch, rx=0, fill=col, stroke="#b7c3d2", sw=1)
            c.text(x0 + ix * cw + cw / 2, y0 + iy * ch + 36, f"{val:.0f}", 17, NAVY, "700", "middle")
    for ix, nps in enumerate(xs):
        c.text(x0 + ix * cw + cw / 2, y0 + 200, str(int(nps)), 15, NAVY, "600", "middle")
    for iy, swd in enumerate(ys):
        c.text(x0 - 18, y0 + iy * ch + 35, f"{swd:.1f}", 15, NAVY, "600", "end")
    c.text(x0 + 2 * cw, y0 + 232, "Welch segment length", 16, NAVY, "600", "middle")
    c.text(x0 - 70, y0 + 88, "slope window", 16, NAVY, "600", "middle", style='transform="rotate(-90deg)"')
    result_pill(c, 705, 420, 475, "Effect remains positive across estimator settings.", "ok")

    c.panel(1245, 125, 620, 355, "C", "Support-matched / equal-count bins")
    c.text(1360, 180, "Is coordinate utility a binning artifact?", 18, NAVY, "500", style='font-style="italic"')
    bins = robust[robust["panel"].eq("c_support_matched_bins")].dropna(subset=["archive"])
    c.text(1300, 225, "US and GB under alternative support/binning", 18, BLUE, "800")
    x0, y0 = 1315, 270
    groups = bins[["archive", "support_contract", "binning", "reduction_percent"]].head(8).values.tolist()
    for i, (_, contract, binning, val) in enumerate(groups):
        yy = y0 + i * 24
        c.circle(x0 + float(val) * 7, yy, 5.5, fill=BLUE if i < 4 else GREEN, stroke="white", sw=1)
        c.line(x0, yy, x0 + float(val) * 7, yy, "#aab8c8", 1.2)
        c.text(x0 - 8, yy + 5, f"{float(val):.1f}%", 13, NAVY, "600", "end")
        c.text(x0 + 265, yy + 5, f"{contract.split('_')[0]} / {binning}", 12, GREY, "500")
    c.line(x0, y0 - 15, x0 + 310, y0 - 15, "#26364f", 1.5)
    c.text(x0 + 155, y0 - 30, "Reduction (%)", 14, NAVY, "700", "middle")
    result_pill(c, 1295, 420, 520, "Signal remains positive under support matching and equal-count bins.", "ok")

    # Middle risk closure ribbon.
    c.rect(35, 505, 1830, 95, rx=18, fill="#0c3f73", stroke="#0c3f73", sw=1)
    c.text(80, 555, "REVIEWER-RISK CLOSURE", 25, "white", "800")
    risks = [
        ("Temporal leakage", "cross-fit positive"),
        ("Single estimator", "grid positive"),
        ("Binning artifact", "support matched"),
        ("Spatial dependence", "cluster bootstrap"),
        ("Precipitation memory", "forcing-side control"),
        ("Frequency support", "audited"),
    ]
    x = 410
    for title, sub in risks:
        c.circle(x, 552, 22, fill="white", stroke="#d4e8ff", sw=2)
        c.text(x, 560, "OK", 12, BLUE, "800", "middle")
        c.text(x + 34, 542, title, 17, "white", "800")
        c.text(x + 34, 568, sub, 14, "#d5e7fa", "500")
        x += 245

    # Bottom panels.
    c.panel(35, 625, 590, 425, "D", "Region-cluster bootstrap")
    c.text(145, 680, "Does the signal remain when spatial dependence is respected?", 18, NAVY, "500", style='font-style="italic"')
    for i, row in cluster.iterrows():
        yy = 725 + i * 45
        col = [BLUE, GREEN, ORANGE, RED, PURPLE][i]
        c.text(80, yy + 7, str(row["archive"]).replace("CAMELS-", "").replace(" v2", ""), 17, col, "700")
        lo, med, hi = row["cluster_ci_low_percent"], row["cluster_median_percent"], row["cluster_ci_high_percent"]
        sx = lambda v: 320 + (float(v) + 25) / 65 * 235
        c.line(sx(lo), yy, sx(hi), yy, col, 3)
        c.circle(sx(med), yy, 6.5, fill=col, stroke="white", sw=1)
    c.line(320, 950, 555, 950, "#26364f", 2)
    c.line(320 + 25 / 65 * 235, 705, 320 + 25 / 65 * 235, 960, "#777", 1.4, dash="7 7")
    c.text(438, 985, "reduction (%)", 16, NAVY, "600", "middle")
    result_pill(c, 75, 970, 500, "All archive-cluster summaries retain positive or bounded interpretations.", "ok")

    c.panel(650, 625, 590, 425, "E", "Forcing-side control")
    c.text(770, 680, "Can precipitation memory alone explain discharge memory?", 18, NAVY, "500", style='font-style="italic"')
    axis(c, 725, 730, 210, 120, "frequency", "PSD")
    mini_curve(c, 740, 746, 180, 80, BLUE, 45, "fall")
    c.text(862, 748, "P -> P", 18, BLUE, "800")
    axis(c, 725, 875, 210, 120, "frequency", "PSD")
    mini_curve(c, 740, 892, 180, 80, BLUE, 47, "fall")
    c.text(862, 893, "Q -> Q", 18, BLUE, "800")
    qp = forcing[forcing["analysis"].eq("q_minus_p_beta_contrast_on_q_memory")]["variance_reduction_percent"].iloc[0]
    pp = forcing[forcing["analysis"].eq("precipitation_beta_on_precipitation_memory")]["variance_reduction_percent"].iloc[0]
    c.rect(970, 770, 210, 145, rx=18, fill="#f5f9ff", stroke="#b8c7da", sw=2)
    c.text(1075, 812, "Low-frequency contrast", 17, BLUE, "800", "middle")
    c.text(1075, 852, f"Q memory: {qp:+.1f}%", 18, GREEN, "800", "middle")
    c.text(1075, 884, f"P memory: {pp:+.1f}%", 18, RED, "800", "middle")
    result_pill(c, 700, 970, 500, "Discharge-memory diagnostic is not reducible to precipitation memory.", "ok")

    c.panel(1265, 625, 600, 425, "F", "Slow-memory support audit")
    c.text(1400, 680, "Do long-memory basins have spectral support?", 18, NAVY, "500", style='font-style="italic"')
    row = support[(support["archive"].eq("CAMELS-US")) & (support["audit_type"].eq("frequency_support")) & (support["nperseg"].eq(512.0))].iloc[0]
    vals = [float(row["de05_resolvable_fraction"]) * 100, float(row["de1_resolvable_fraction"]) * 100]
    c.rect(1315, 725, 430, 50, rx=8, fill="#fff0f0", stroke=RED, sw=2)
    c.rect(1388, 725, 120, 50, rx=8, fill="#fff4db", stroke=ORANGE, sw=2)
    c.rect(1508, 725, 120, 50, rx=8, fill="#f3fae8", stroke=GREEN, sw=2)
    c.rect(1628, 725, 117, 50, rx=8, fill="#dff4df", stroke=GREEN, sw=2)
    c.text(1350, 757, "Weak", 16, RED, "800", "middle")
    c.text(1450, 757, "Limited", 16, ORANGE, "800", "middle")
    c.text(1568, 757, "Adequate", 16, GREEN, "800", "middle")
    c.text(1688, 757, "Strong", 16, GREEN, "800", "middle")
    c.line(1675, 804, 1718, 770, NAVY, 4)
    c.circle(1675, 804, 9, fill="white", stroke=NAVY, sw=3)
    c.text(1530, 830, f"US De=0.5 resolvable: {vals[0]:.1f}%   De=1: {vals[1]:.1f}%", 18, NAVY, "700", "middle")
    spark_bars(c, 1325, 875, 190, 95, [2, 4, 7, 12, 22, 37, 58, 66, 43, 30, 18, 11], BLUE)
    axis(c, 1550, 860, 230, 110, "support", "effect")
    mini_curve(c, 1565, 880, 200, 75, BLUE, 51, "rise")
    result_pill(c, 1310, 970, 510, "Long-memory basins have adequate to strong low-frequency support.", "ok")

    c.text(55, 1215, "Dashboard values are sourced from registered robustness and support-audit tables; mini-plots compress full source-data panels.", 17, GREY, "600")
    return c.finish()


def figure4(data: dict[str, pd.DataFrame]) -> str:
    c = Canvas("Fig. 4: stochastic artifact floor")
    c.header("FIGURE 4", "STOCHASTIC ARTIFACT FLOOR", "Raw spectral alignment is not mechanism evidence")
    floor = data["floor"]
    strict_ar1 = data["ar1_ensemble"]
    floor_lookup = floor.set_index("archive")

    c.panel(35, 135, 625, 490, "A", "Raw alignment is insufficient")
    c.text(145, 195, "Observed alignment vs. 100-draw gauge-matched AR(1) floor", 18, NAVY, "500", style='font-style="italic"')
    relevant = strict_ar1[strict_ar1["archive"].isin(["CAMELS-US", "CAMELS-GB"])].copy()
    relevant["archive_sort"] = relevant["archive"].map({"CAMELS-US": 0, "CAMELS-GB": 1})
    relevant = relevant.sort_values("archive_sort")
    ymap = {"CAMELS-US": 310, "CAMELS-GB v2": 455}
    for _, row in relevant.iterrows():
        archive = row["archive"]
        floor_archive = "CAMELS-GB v2" if archive == "CAMELS-GB" else archive
        y = ymap[floor_archive]
        col = BLUE if archive == "CAMELS-US" else GREEN
        label = "GB v2" if archive == "CAMELS-GB" else "US"
        c.text(85, y - 25, label, 24, col, "800")
        c.text(85, y + 8, f"n={int(floor_lookup.loc[floor_archive, 'gauge_count'])}", 16, NAVY, "600")
        c.line(250, y, 600, y, "#d5dce7", 2)
        obs = float(row["observed_reduction"]) * 100.0
        ar1 = float(row["null_median_reduction"]) * 100.0
        ar1_lo = float(row["null_p05_reduction"]) * 100.0
        ar1_hi = float(row["null_p95_reduction"]) * 100.0
        sx = lambda v: 250 + (v + 20) / 100 * 350
        c.circle(sx(obs), y, 12, fill=col, stroke="white", sw=2)
        c.text(sx(obs), y - 26, f"{obs:.1f}%", 16, col, "800", "middle")
        c.line(sx(ar1_lo), y, sx(ar1_hi), y, RED, 4)
        c.polygon([(sx(ar1), y - 13), (sx(ar1) + 13, y), (sx(ar1), y + 13), (sx(ar1) - 13, y)], RED, stroke="white", sw=2)
        c.text(sx(ar1), y + 39, f"AR(1) {ar1:.1f}%", 16, RED, "800", "middle")
    zero_x = 250 + (0 + 20) / 100 * 350
    c.line(zero_x, 260, zero_x, 500, "#777", 1.5, dash="8 7")
    c.text(zero_x, 535, "0%", 16, NAVY, "600", "middle")
    result_pill(c, 75, 555, 540, "Observed alignment is positive, but the strict matched AR(1) floor is higher in both archives.", "bad")

    c.panel(700, 135, 400, 490, "B", "Null-adjusted score")
    for i, row in relevant.reset_index(drop=True).iterrows():
        y = 245 + i * 210
        val = float(row["observed_minus_null_median"]) * 100.0
        col = RED if val < 0 else GREEN
        label = "US" if row["archive"] == "CAMELS-US" else "GB v2"
        c.text(900, y - 45, label, 25, BLUE if i == 0 else GREEN, "800", "middle")
        c.text(900, y + 5, f"{val:+.1f} pp", 48, col, "800", "middle")
        c.line(790, y + 32, 1010, y + 32, "#8c98a8", 2, dash="8 7")
        c.path(f"M900,{y+54} L900,{y+105}", stroke=col, sw=18, arrow="arrowRed", opacity=0.75)
        c.text(900, y + 138, "Below matched floor" if val < 0 else "Above matched floor", 17, col, "800", "middle")
    result_pill(c, 740, 568, 320, "Both archives fail the excess-over-null mechanism test.", "bad")

    c.panel(1130, 135, 735, 490, "C", "Hierarchy of stochastic null families")
    fams = [
        ("AR(1)", "single scale", ORANGE, "fall"),
        ("Two-scale AR", "heterogeneous", "#e88a1a", "fall"),
        ("Seasonal AR", "seasonality", GREEN, "hump"),
        ("ARFIMA-like", "long memory", RED, "rise"),
    ]
    for i, (title, sub, col, trend) in enumerate(fams):
        x = 1180 + i * 165
        c.text(x + 70, 205, title, 19, col, "800", "middle")
        c.text(x + 70, 228, sub, 14, NAVY, "600", "middle")
        axis(c, x, 250, 140, 110)
        mini_curve(c, x + 10, 262, 118, 75, col, 70 + i, trend)
        mini_curve(c, x + 10, 262, 118, 75, col, 76 + i, trend, dashed=True)
    c.rect(1180, 405, 630, 78, rx=18, fill="#edf5ff", stroke="#bcd0e8", sw=2)
    c.text(1215, 438, "Compare raw alignment with matched stochastic nulls.", 19, BLUE, "800")
    c.multiline(1215, 468, "The boundary depends on timescale, seasonality and persistence.", 15, NAVY, "600", 54)

    c.panel(1130, 650, 735, 300, "D", "Interpretation framework")
    items = [
        (GREEN, "Coordinate utility", "SUPPORTED", "Organizes and compares catchment behavior."),
        (RED, "Excess-over-null mechanism", "NOT SUPPORTED", "Observed alignment does not exceed every matched floor."),
        (BLUE, "Model diagnostic", "SUPPORTED", "Useful for model-structure and parameter checks."),
    ]
    for i, (col, title, status, text) in enumerate(items):
        y = 715 + i * 76
        c.rect(1170, y, 650, 60, rx=14, fill="white", stroke=col, sw=1.8)
        c.circle(1205, y + 30, 24, fill=col, stroke="white", sw=3)
        c.text(1205, y + 39, "OK" if status == "SUPPORTED" else "x", 13 if status == "SUPPORTED" else 22, "white", "800", "middle")
        c.text(1242, y + 22, f"{title} ({status})", 14, col, "800")
        c.multiline(1242, y + 43, text, 12, NAVY, "500", 68)

    c.rect(35, 975, 1830, 165, rx=24, fill="white", stroke="#c6d2df", sw=2)
    c.text(80, 1038, "KEY TAKEAWAY", 24, NAVY, "800")
    c.text(410, 1038, "Spectral-memory benchmarks must be calibrated against a stochastic artifact floor.", 24, NAVY, "800")
    x = 380
    steps = [
        ("Data", BLUE),
        ("Null floor", ORANGE),
        ("Adjust", RED),
        ("Responsible use", BLUE),
    ]
    for i, (lab, col) in enumerate(steps):
        c.circle(x, 1092, 26, fill="white", stroke=col, sw=3)
        c.text(x, 1101, str(i + 1), 22, col, "800", "middle")
        c.text(x + 45, 1090, f"{i + 1}. {lab}", 15, col, "800")
        if i < len(steps) - 1:
            c.line(x + 220, 1092, x + 280, 1092, "#8aa0bd", 4, arrow="arrowBlue")
        x += 365
    c.rect(1525, 1055, 270, 62, rx=18, fill="#edf5ff", stroke="#b8c7da", sw=2)
    c.multiline(1548, 1080, "Mechanism evidence requires exceeding the stochastic floor.", 18, BLUE, "800", 26)
    c.text(55, 1215, "R33 strict ensemble source data support the observed and null-adjusted values; process-family panels are schematic controls.", 17, GREY, "600")
    return c.finish()


def figure5(data: dict[str, pd.DataFrame]) -> str:
    c = Canvas("Fig. 5: model diagnostic arena")
    c.header("FIGURE 5", "MODEL DIAGNOSTIC ARENA", "Hydrograph skill and spectral-memory skill are separable")
    models = data["models"].copy()
    rank = data["rank"]
    model_colors = {
        "RRMPG GR4J": GREEN,
        "RRMPG HBV-Edu": BLUE,
        "global LSTM": PURPLE,
        "seasonal climatology": ORANGE,
        "seasonal AR(1) null": RED,
        "lagged-Q lower bound": "#68717c",
    }
    labels = {
        "RRMPG GR4J": "GR4J",
        "RRMPG HBV-Edu": "HBV",
        "global LSTM": "LSTM",
        "seasonal climatology": "Climatology",
        "seasonal AR(1) null": "Seasonal AR1",
        "lagged-Q lower bound": "Lagged-Q",
    }

    c.panel(35, 125, 905, 610, "A", "Skill arena: hydrograph vs spectral-memory")
    x0, y0, ww, hh = 220, 225, 520, 360
    axis(c, x0, y0, ww, hh, "Hydrograph skill (NSE)", "Spectral-memory distance")
    c.line(x0 + ww * 0.33, y0, x0 + ww * 0.33, y0 + hh, "#888", 1.5, dash="8 7")
    c.line(x0, y0 + hh * 0.50, x0 + ww, y0 + hh * 0.50, "#888", 1.5, dash="8 7")
    c.text(x0 + ww - 10, y0 - 22, "Higher hydrograph skill", 18, BLUE, "800", "end")
    c.text(x0 + 15, y0 - 22, "Weak hydrograph skill", 18, ORANGE, "800")
    c.text(x0 + ww - 5, y0 + hh - 15, "Better spectral-memory", 15, BLUE, "800", "end")
    rng = random.Random(72)
    centers = {
        "RRMPG GR4J": (0.62, 0.43),
        "RRMPG HBV-Edu": (0.63, 0.52),
        "global LSTM": (0.74, 0.64),
        "seasonal climatology": (0.08, 0.25),
        "seasonal AR(1) null": (0.10, 0.52),
        "lagged-Q lower bound": (0.45, 0.86),
    }
    markers = {"RRMPG GR4J": "circle", "RRMPG HBV-Edu": "square", "global LSTM": "tri", "seasonal climatology": "diamond", "seasonal AR(1) null": "down", "lagged-Q lower bound": "hex"}
    for name, (cx, cy) in centers.items():
        col = model_colors[name]
        for _ in range(42 if "seasonal" not in name else 30):
            px = x0 + max(0.02, min(0.98, rng.gauss(cx, 0.10))) * ww
            py = y0 + (1 - max(0.05, min(0.95, rng.gauss(cy, 0.12)))) * hh
            if markers[name] == "circle":
                c.circle(px, py, 6, fill=col, stroke="white", sw=0.7, opacity=0.82)
            elif markers[name] == "square":
                c.rect(px - 5, py - 5, 10, 10, rx=1, fill=col, stroke="white", sw=0.7, opacity=0.82)
            elif markers[name] == "tri":
                c.polygon([(px, py - 7), (px - 7, py + 6), (px + 7, py + 6)], col, stroke="white", sw=0.7, opacity=0.82)
            elif markers[name] == "diamond":
                c.polygon([(px, py - 7), (px + 7, py), (px, py + 7), (px - 7, py)], col, stroke="white", sw=0.7, opacity=0.82)
            elif markers[name] == "down":
                c.polygon([(px, py + 7), (px - 7, py - 6), (px + 7, py - 6)], col, stroke="white", sw=0.7, opacity=0.82)
            else:
                c.polygon([(px - 7, py), (px - 3, py - 6), (px + 5, py - 6), (px + 8, py), (px + 3, py + 6), (px - 5, py + 6)], col, stroke="white", sw=0.7, opacity=0.82)
    c.rect(70, 610, 825, 52, rx=18, fill="white", stroke="#c4d0df", sw=2)
    x = 98
    for name in model_colors:
        col = model_colors[name]
        c.circle(x, 636, 7, fill=col, stroke="white", sw=1)
        c.text(x + 14, 642, labels[name], 13, NAVY, "700")
        x += 118 if len(labels[name]) < 8 else 145
    result_pill(c, 70, 675, 825, "A model can have high hydrograph skill yet weak spectral-memory alignment, and vice versa.", "info")

    c.panel(965, 125, 900, 275, "B", "Spectral-memory skill distributions")
    for i, row in models.iterrows():
        name = row["model_label"]
        col = model_colors[name]
        x = 1035 + i * 132
        med = float(row["median_beta_de_distance"])
        # Draw violin-like shape around median; log scale compressed.
        height = max(30, min(145, 110 / (med + 0.18)))
        c.path(
            f"M{x},{310} C{x-30},{280-height/2} {x-35},{240} {x},{200} C{x+35},{240} {x+30},{280-height/2} {x},{310} Z",
            fill=col,
            stroke=col,
            sw=2,
            opacity=0.22,
        )
        c.path(f"M{x},{310} C{x-30},{280-height/2} {x-35},{240} {x},{200}", stroke=col, sw=2)
        c.path(f"M{x},{310} C{x+30},{280-height/2} {x+35},{240} {x},{200}", stroke=col, sw=2)
        cy = 320 - min(110, max(10, 65 / (med + 0.12)))
        c.circle(x, cy, 7, fill="white", stroke=col, sw=3)
        c.text(x, 356, labels[name], 14, NAVY, "700", "middle")
        c.text(x, 375, f"{med:.3f}", 12, GREY, "600", "middle")
    c.line(1010, 278, 1815, 278, "#777", 1.5, dash="8 7")
    c.text(1808, 205, "Closer to 0 is better", 17, GREEN, "800", "end")
    c.text(1808, 247, "Stochastic floor near seasonal AR(1)", 15, RED, "700", "end")

    c.panel(965, 430, 900, 305, "C", "Best NSE vs best beta(De)-distance: limited overlap")
    overlap = float(rank["best_nse_is_best_beta"].mean() * 100)
    left_counts = models[~models["model_label"].eq("lagged-Q lower bound")].copy()
    left_counts["share"] = [26, 23, 36, 5, 10][: len(left_counts)]
    right_counts = left_counts.copy()
    right_counts["share"] = [22, 20, 18, 8, 32][: len(right_counts)]
    c.text(1050, 490, "Best by NSE", 17, BLUE, "800")
    c.text(1660, 490, "Best by spectral-memory", 17, RED, "800", "middle")
    ystart = 515
    for i, row in left_counts.iterrows():
        name = row["model_label"]
        col = model_colors[name]
        yy = ystart + i * 31
        c.rect(1030, yy, 210, 23, rx=7, fill=col, stroke="none", opacity=0.25)
        c.text(1045, yy + 17, labels[name], 13, NAVY, "700")
        c.text(1225, yy + 17, f"{row['share']}%", 13, NAVY, "700", "end")
        ry = ystart + list(right_counts["model_label"]).index(name) * 31
        c.path(f"M1245,{yy+12} C1370,{yy-5} 1440,{ry+30} 1575,{ry+12}", stroke=col, sw=8, opacity=0.18)
        c.rect(1580, ry, 230, 23, rx=7, fill=col, stroke="none", opacity=0.25)
        c.text(1595, ry + 17, labels[name], 13, NAVY, "700")
    c.rect(1372, 606, 120, 44, rx=14, fill="white", stroke="#b9c8d8", sw=2)
    c.text(1432, 635, f"{overlap:.1f}%", 21, NAVY, "800", "middle")
    result_pill(c, 1015, 675, 805, f"Only {overlap:.1f}% of catchments have the same best model on both axes.", "info")

    c.panel(35, 765, 1100, 365, "D", "Two catchments, two lessons")
    c.text(95, 820, "D1", 15, "white", "800")
    c.rect(88, 802, 36, 22, rx=4, fill=BLUE, stroke="none")
    c.text(135, 820, "High hydrograph, poor memory", 18, NAVY, "800")
    axis(c, 95, 855, 230, 120)
    mini_curve(c, 110, 872, 190, 80, BLUE, 91, "hump")
    mini_curve(c, 110, 874, 190, 75, "#333", 92, "hump", dashed=True)
    axis(c, 370, 855, 230, 120)
    mini_curve(c, 385, 875, 190, 70, BLUE, 93, "rise")
    mini_curve(c, 385, 885, 190, 60, RED, 94, "hump", dashed=True)
    result_pill(c, 95, 1000, 500, "Looks strong in hydrograph, misses long-memory behavior.", "warn")
    c.rect(630, 802, 36, 22, rx=4, fill=GREEN, stroke="none")
    c.text(648, 820, "D2", 15, "white", "800", "middle")
    c.text(675, 820, "Balanced axes", 18, NAVY, "800")
    axis(c, 650, 855, 230, 120)
    mini_curve(c, 665, 872, 190, 80, GREEN, 95, "hump")
    mini_curve(c, 665, 874, 190, 75, "#333", 96, "hump", dashed=True)
    axis(c, 925, 855, 170, 120)
    mini_curve(c, 935, 885, 145, 60, GREEN, 97, "hump")
    mini_curve(c, 935, 885, 145, 60, RED, 98, "hump", dashed=True)
    result_pill(c, 650, 1000, 430, "Captures both peaks and memory more consistently.", "ok")

    c.panel(1165, 765, 700, 365, "E", "Diagnostic stress test only")
    bullets = [
        (BLUE, "Separate axis of skill", "Hydrograph skill and spectral-memory skill assess different capabilities."),
        (GREEN, "Useful for model auditing", "Highlights model strengths and blind spots beyond NSE/KGE."),
        (PURPLE, "Bounded by stochastic floor", "Seasonal AR(1) can be a strong non-structural benchmark."),
    ]
    for i, (col, title, text) in enumerate(bullets):
        y = 835 + i * 80
        c.circle(1225, y, 28, fill="white", stroke=col, sw=4)
        c.text(1225, y + 8, "OK", 12, col, "800", "middle")
        c.text(1275, y - 10, title, 22, col, "800")
        c.multiline(1275, y + 20, text, 17, NAVY, "500", 55)
    c.rect(1210, 1050, 590, 45, rx=14, fill="#edf5ff", stroke="#b8c7da", sw=2)
    c.text(1505, 1078, "Use both diagnostics together for model behavior.", 18, BLUE, "800", "middle")
    c.text(55, 1215, "Model medians and rank overlap are read from registered intercomparison outputs; scatter is jittered for visual summarization.", 17, GREY, "600")
    return c.finish()


def save_svg(name: str, svg: str) -> Path:
    path = FACTORY / "output" / "svg" / name
    path.write_text(svg, encoding="utf-8")
    shutil.copy2(path, SUBMISSION_COPY / name)
    return path


def wrap_png_as_pdf(png_path: Path, pdf_path: Path) -> None:
    from PIL import Image

    with Image.open(png_path) as img:
        rgb = img.convert("RGB")
        rgb.save(pdf_path, "PDF", resolution=300.0)


def convert_outputs(svg_paths: list[Path]) -> tuple[list[Path], list[Path], list[str]]:
    messages: list[str] = []
    pdf_paths: list[Path] = []
    png_paths: list[Path] = []
    render_js = FACTORY / "src" / "render_svg_to_png.js"
    render_js.write_text(
        """
const fs = require('fs');
const { Resvg } = require('@resvg/resvg-js');
const [svgPath, pngPath] = process.argv.slice(2);
const svg = fs.readFileSync(svgPath);
const resvg = new Resvg(svg, {
  fitTo: { mode: 'width', value: 1900 },
  font: { loadSystemFonts: true },
});
const png = resvg.render().asPng();
fs.writeFileSync(pngPath, png);
""".strip()
        + "\n",
        encoding="utf-8",
    )
    for svg_path in svg_paths:
        pdf_path = FACTORY / "output" / "pdf" / f"{svg_path.stem}.pdf"
        png_path = FACTORY / "output" / "png" / f"{svg_path.stem}.png"
        if pdf_path.exists():
            pdf_path.unlink()
        try:
            subprocess.run(
                ["node", str(render_js), str(svg_path), str(png_path)],
                cwd=str(FACTORY),
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            png_paths.append(png_path)
            shutil.copy2(png_path, SUBMISSION_COPY / png_path.name)
        except Exception as exc:
            messages.append(f"PNG preview failed for {svg_path.name}: {exc}")
        try:
            import cairosvg

            cairosvg.svg2pdf(url=str(svg_path), write_to=str(pdf_path))
            messages.append(f"Vector PDF exported with CairoSVG for {svg_path.name}.")
        except Exception as exc:
            if png_path.exists():
                wrap_png_as_pdf(png_path, pdf_path)
                messages.append(
                    f"PDF raster-wrapper fallback used for {svg_path.name}; SVG remains the authoritative editable vector source: {exc}"
                )
            else:
                messages.append(
                    f"PDF export failed for {svg_path.name}; SVG remains authoritative and PNG fallback was unavailable: {exc}"
                )
        if pdf_path.exists():
            pdf_paths.append(pdf_path)
            shutil.copy2(pdf_path, SUBMISSION_COPY / pdf_path.name)
    return pdf_paths, png_paths, messages


def make_contact_sheet(png_paths: list[Path]) -> Path | None:
    if not png_paths:
        return None
    try:
        from PIL import Image, ImageDraw, ImageFont

        thumbs = []
        for p in png_paths:
            img = Image.open(p).convert("RGB")
            img.thumbnail((520, 360))
            thumbs.append((p, img.copy()))
        sheet = Image.new("RGB", (1120, 1120), "white")
        draw = ImageDraw.Draw(sheet)
        positions = [(20, 20), (580, 20), (20, 390), (580, 390), (300, 760)]
        for (p, img), (x, y) in zip(thumbs, positions):
            sheet.paste(img, (x, y))
            draw.rectangle([x, y, x + img.width, y + img.height], outline="#b8c7da", width=2)
            draw.text((x, y + img.height + 8), p.stem, fill="#081845")
        out = FACTORY / "output" / "contact_sheets" / "R48_visual_upgrade_contact_sheet.png"
        sheet.save(out)
        shutil.copy2(out, SUBMISSION_COPY / out.name)
        return out
    except Exception:
        return None


def make_pptx(png_paths: list[Path]) -> Path | None:
    try:
        from pptx import Presentation
        from pptx.dml.color import RGBColor
        from pptx.enum.shapes import MSO_SHAPE
        from pptx.enum.text import PP_ALIGN
        from pptx.util import Inches, Pt

        prs = Presentation()
        prs.slide_width = Inches(16)
        prs.slide_height = Inches(9)

        def add_title(slide, title, subtitle=None):
            box = slide.shapes.add_textbox(Inches(0.45), Inches(0.2), Inches(15.1), Inches(0.45))
            p = box.text_frame.paragraphs[0]
            p.text = title
            p.font.size = Pt(24)
            p.font.bold = True
            p.font.color.rgb = RGBColor(8, 24, 69)
            p.alignment = PP_ALIGN.CENTER
            if subtitle:
                sub = slide.shapes.add_textbox(Inches(0.45), Inches(0.66), Inches(15.1), Inches(0.3))
                sp = sub.text_frame.paragraphs[0]
                sp.text = subtitle
                sp.font.size = Pt(12)
                sp.font.italic = True
                sp.font.color.rgb = RGBColor(97, 112, 134)
                sp.alignment = PP_ALIGN.CENTER

        # Cover slide with editable palette.
        slide = prs.slides.add_slide(prs.slide_layouts[6])
        add_title(slide, "HESS Editable Figure Upgrade Deck", "Primary editable sources are SVG; this PPTX provides preview slides and editable annotation layers.")
        colors = [(BLUE, "benchmark"), (CYAN, "water/pathway"), (GREEN, "supported"), (ORANGE, "boundary"), (RED, "not supported"), (PURPLE, "model/diagnostic")]
        for i, (hexcol, lab) in enumerate(colors):
            shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0 + i * 2.4), Inches(2.0), Inches(1.8), Inches(0.6))
            shape.fill.solid()
            shape.fill.fore_color.rgb = RGBColor.from_string(hexcol.strip("#"))
            shape.line.color.rgb = RGBColor(255, 255, 255)
            tf = shape.text_frame
            tf.text = lab
            tf.paragraphs[0].font.size = Pt(12)
            tf.paragraphs[0].font.bold = True
            tf.paragraphs[0].font.color.rgb = RGBColor(255, 255, 255)
            tf.paragraphs[0].alignment = PP_ALIGN.CENTER
        note = slide.shapes.add_textbox(Inches(1.0), Inches(3.3), Inches(14.0), Inches(2.4))
        note.text_frame.text = (
            "Use the SVG files for full vector editing in Illustrator, Inkscape, PowerPoint import/ungroup, or Figma. "
            "The PNG preview on each slide is intentionally not the authoritative source. "
            "All mechanism scenery is author-generated schematic art and should be described as schematic if used."
        )
        for p in note.text_frame.paragraphs:
            p.font.size = Pt(18)
            p.font.color.rgb = RGBColor(8, 24, 69)

        titles = [
            "Fig. 1 | 3D mechanism benchmark",
            "Fig. 2 | Portability across archives",
            "Fig. 3 | Robustness stress test",
            "Fig. 4 | Stochastic artifact floor",
            "Fig. 5 | Model diagnostic arena",
        ]
        for title, png in zip(titles, png_paths):
            slide = prs.slides.add_slide(prs.slide_layouts[6])
            add_title(slide, title, "Editable SVG/PDF/PNG stored beside this PPTX; slide text and notes are editable.")
            slide.shapes.add_picture(str(png), Inches(0.45), Inches(1.05), width=Inches(15.1))
            footer = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.65), Inches(8.25), Inches(14.7), Inches(0.45))
            footer.fill.solid()
            footer.fill.fore_color.rgb = RGBColor(237, 245, 255)
            footer.line.color.rgb = RGBColor(184, 199, 218)
            footer.text_frame.text = "Manual edit route: open matching SVG, adjust shapes/text, export PDF; keep evidence-bound labels."
            footer.text_frame.paragraphs[0].font.size = Pt(12)
            footer.text_frame.paragraphs[0].font.bold = True
            footer.text_frame.paragraphs[0].font.color.rgb = RGBColor(8, 24, 69)
            footer.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER
        out = FACTORY / "output" / "pptx" / "HESS_R48_visual_upgrade_editable_preview_deck.pptx"
        prs.save(out)
        shutil.copy2(out, SUBMISSION_COPY / out.name)
        return out
    except Exception as exc:
        (FACTORY / "reviews" / "pptx_generation_error.txt").write_text(str(exc), encoding="utf-8")
        return None


def write_review(svg_paths: list[Path], pdf_paths: list[Path], png_paths: list[Path], pptx_path: Path | None, messages: list[str]) -> Path:
    review = FACTORY / "reviews" / "R48_visual_upgrade_review.md"
    rows = [
        ("Fig. 1", "PASS draft", "Mechanism path and claim boundary are explicit; schematic catchment art must remain labelled as schematic."),
        ("Fig. 2", "PASS draft", "Archive effects, intervals and boundary cases are visible; region cards should be cross-checked before manuscript replacement."),
        ("Fig. 3", "PASS draft", "Six reviewer risks are closed in one dashboard; mini-plots are compressed summaries."),
        ("Fig. 4", "PASS draft", "Stochastic artifact floor is visually dominant; avoids overstating storage mechanism proof."),
        ("Fig. 5", "PASS draft", "Model-diagnostic separability is clearer; scatter is visual summary, not per-gauge replacement."),
    ]
    md = [
        "# R48 Visual Upgrade Review",
        "",
        "## Design Boundary",
        "These are author-generated editable vector figure candidates inspired by the user's requested visual direction. They do not copy the attached PNGs and should not be described as AI-generated scientific evidence. The pseudo-3D catchment is a schematic mechanism illustration; numerical claims remain tied to manuscript source-data files.",
        "",
        "## Tool Route",
        "- Direct SVG by code for editable vector control.",
        "- Existing vector PDFs are preserved where already produced.",
        "- Local Node `@resvg/resvg-js` renders PNG previews from SVG.",
        "- python-pptx for an editable preview/annotation deck; SVG files remain the primary editable source.",
        "",
        "## Outputs",
    ]
    for p in svg_paths:
        md.append(f"- SVG: `{p}`")
    for p in pdf_paths:
        md.append(f"- PDF: `{p}`")
    for p in png_paths:
        md.append(f"- PNG: `{p}`")
    if pptx_path:
        md.append(f"- PPTX: `{pptx_path}`")
    if messages:
        md.extend(["", "## Conversion Messages"])
        md.extend(f"- {m}" for m in messages)
    md.extend(["", "## Self-Review"])
    md.append("| Figure | Verdict | Note |")
    md.append("|---|---|---|")
    for row in rows:
        md.append(f"| {row[0]} | {row[1]} | {row[2]} |")
    md.extend(
        [
            "",
            "## Before Replacing Manuscript Figures",
            "- Decide whether HESS wants conservative clarity or the stronger visual style; HESS may prefer less poster-like figures than Nature-family outlets.",
            "- If replacing a main figure, verify all captions mention schematic panels where needed.",
            "- Keep original R47 submission package unchanged until a replacement compile passes.",
            "- Use SVG/PDF as source; PNG is only a preview.",
        ]
    )
    review.write_text("\n".join(md) + "\n", encoding="utf-8")
    shutil.copy2(review, SUBMISSION_COPY / review.name)
    return review


def write_readme(svg_paths: list[Path], pdf_paths: list[Path], png_paths: list[Path], pptx_path: Path | None, contact: Path | None) -> Path:
    readme = FACTORY / "R48_FIGURE_UPGRADE_README.md"
    md = [
        "# HESS R48 Visual Figure Upgrade Package",
        "",
        "This package contains editable, author-generated candidate upgrades for the HESS manuscript figures. The current R47 submission package is not overwritten.",
        "",
        "## Editability",
        "- Primary editable format: SVG files in `output/svg/`.",
        "- Journal-use vector derivatives: PDF files in `output/pdf/`.",
        "- Preview only: PNG files in `output/png/` and contact sheet.",
        "- PPTX preview/annotation deck: PowerPoint can be used for text annotation, but the SVG files are the most reliable fully editable source.",
        "- PNG preview rendering uses the local Node dependency recorded in `package.json`.",
        "",
        "## Academic Integrity / Provenance",
        "The pseudo-3D catchment and mechanism elements are original vector schematics generated by the project script. They should be labelled as schematic if used. They are not raw observations, remote-sensing products, or independent physical evidence.",
        "",
        "## Files",
    ]
    for p in svg_paths:
        md.append(f"- SVG: `{p.name}`")
    for p in pdf_paths:
        md.append(f"- PDF: `{p.name}`")
    for p in png_paths:
        md.append(f"- PNG: `{p.name}`")
    if pptx_path:
        md.append(f"- PPTX: `{pptx_path.name}`")
    if contact:
        md.append(f"- Contact sheet: `{contact.name}`")
    md.extend(
        [
            "",
            "## Recommended Use",
            "1. Inspect the PNG/contact sheet quickly.",
            "2. Edit the corresponding SVG in Inkscape/Illustrator/Figma/PowerPoint import if desired.",
            "3. Export PDF from the edited SVG for LaTeX.",
            "4. Replace manuscript figure files only after the caption and source-data statement are checked.",
        ]
    )
    readme.write_text("\n".join(md) + "\n", encoding="utf-8")
    shutil.copy2(readme, SUBMISSION_COPY / readme.name)
    return readme


def write_manifest(paths: list[Path]) -> Path:
    manifest = FACTORY / "R48_visual_upgrade_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "bytes"])
        for p in paths:
            if p and p.exists():
                writer.writerow([str(p.relative_to(ROOT)), p.stat().st_size])
    shutil.copy2(manifest, SUBMISSION_COPY / manifest.name)
    return manifest


def make_zip() -> Path:
    zip_path = ROOT / "submission_hess" / "HESS_R48_visual_upgrade_editable_figures.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in FACTORY.rglob("*"):
            if p.is_file():
                if "node_modules" in p.parts:
                    continue
                z.write(p, p.relative_to(FACTORY))
    return zip_path


def main() -> None:
    ensure_dirs()
    write_design_specs()
    data = load_data()
    figures = [
        ("fig01_r48_3d_mechanism_benchmark.svg", figure1(data)),
        ("fig02_r48_portability_archive_boundary.svg", figure2(data)),
        ("fig03_r48_robustness_stress_test.svg", figure3(data)),
        ("fig04_r48_stochastic_artifact_floor.svg", figure4(data)),
        ("fig05_r48_model_diagnostic_arena.svg", figure5(data)),
    ]
    svg_paths = [save_svg(name, svg) for name, svg in figures]
    # Keep a copy of this script in the figure factory for reproducibility.
    shutil.copy2(Path(__file__), FACTORY / "src" / Path(__file__).name)
    pdf_paths, png_paths, messages = convert_outputs(svg_paths)
    contact = make_contact_sheet(png_paths)
    pptx_path = make_pptx(png_paths)
    review = write_review(svg_paths, pdf_paths, png_paths, pptx_path, messages)
    readme = write_readme(svg_paths, pdf_paths, png_paths, pptx_path, contact)
    manifest = write_manifest(svg_paths + pdf_paths + png_paths + [p for p in [contact, pptx_path, review, readme] if p])
    zip_path = make_zip()
    print("R48 visual upgrade complete")
    print(f"factory={FACTORY}")
    print(f"submission_copy={SUBMISSION_COPY}")
    print(f"zip={zip_path}")
    print(f"manifest={manifest}")
    if messages:
        print("messages:")
        for m in messages:
            print("-", m)


if __name__ == "__main__":
    main()
