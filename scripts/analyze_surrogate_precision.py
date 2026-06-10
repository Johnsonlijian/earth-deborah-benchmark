"""Compare CAMELS null-model rejection counts across surrogate ensemble sizes."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports" / "tables"
FIGURES = ROOT / "reports" / "figures"
NOTES = ROOT / "reports" / "manuscript_notes"


NULLS = [
    ("seasonal", "Seasonal"),
    ("ar1", "AR(1)"),
    ("wy_shuffle", "Water-year"),
    ("phase_rand", "Phase"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        default=str(TABLES / "camels_674_n15_summary.csv"),
        help="Baseline CAMELS summary table, usually the N=15 run.",
    )
    parser.add_argument(
        "--candidate",
        default=str(TABLES / "camels_673_n30_surrogates_summary.csv"),
        help="Higher-N CAMELS summary table to compare against baseline.",
    )
    parser.add_argument(
        "--candidate-label",
        default="N30",
        help="Short label for the higher-N run.",
    )
    parser.add_argument(
        "--output-prefix",
        default="camels_673_surrogate_precision_n15_vs_n30",
    )
    return parser.parse_args()


def summarize_table(path: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"gauge_id": str})
    rows = []
    total = len(df)
    for prefix, null_label in NULLS:
        n_col = f"{prefix}_n"
        p_col = f"{prefix}_p_ge"
        if n_col not in df.columns or p_col not in df.columns:
            continue
        valid = df[n_col].fillna(0) > 0
        n_valid = int(valid.sum())
        n = df.loc[valid, n_col].astype(float)
        p = df.loc[valid, p_col].astype(float)
        rows.append(
            {
                "run": label,
                "null_prefix": prefix,
                "null_model": null_label,
                "n_total_rows": total,
                "n_valid": n_valid,
                "median_n_surrogates": float(n.median()) if n_valid else np.nan,
                "min_n_surrogates": float(n.min()) if n_valid else np.nan,
                "max_n_surrogates": float(n.max()) if n_valid else np.nan,
                "p_zero_count": int((p == 0).sum()),
                "p_zero_fraction_total": float((p == 0).sum() / total),
                "p_le_1_over_n_count": int((p <= 1.0 / n).sum()),
                "p_le_1_over_n_fraction_total": float((p <= 1.0 / n).sum() / total),
            }
        )
    return pd.DataFrame(rows)


def write_figure(summary: pd.DataFrame, output_prefix: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    runs = list(dict.fromkeys(summary["run"].tolist()))
    nulls = [label for _, label in NULLS if label in set(summary["null_model"])]
    x = np.arange(len(nulls))
    width = 0.34 if len(runs) == 2 else 0.24
    colors = {"N15": "#477A9C", "N30": "#C75D4D", "N50": "#5B8F62"}

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2), constrained_layout=True)
    ax = axes[0]
    for i, run in enumerate(runs):
        vals = []
        for null in nulls:
            row = summary[(summary["run"] == run) & (summary["null_model"] == null)]
            vals.append(float(row["p_le_1_over_n_count"].iloc[0]) if len(row) else np.nan)
        offset = (i - (len(runs) - 1) / 2) * width
        ax.bar(x + offset, vals, width=width, label=run, color=colors.get(run, "#777777"))
    ax.set_xticks(x)
    ax.set_xticklabels(nulls, rotation=25, ha="right")
    ax.set_ylabel("catchments at empirical threshold")
    ax.set_title("Resolution-limited rejections", loc="left")
    ax.legend(frameon=False)
    ax.set_ylim(0, 700)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.8)

    ax = axes[1]
    if len(runs) >= 2:
        base, cand = runs[0], runs[-1]
        deltas = []
        for null in nulls:
            b = summary[(summary["run"] == base) & (summary["null_model"] == null)][
                "p_le_1_over_n_count"
            ].iloc[0]
            c = summary[(summary["run"] == cand) & (summary["null_model"] == null)][
                "p_le_1_over_n_count"
            ].iloc[0]
            deltas.append(float(c - b))
        delta_colors = ["#5B8F62" if d >= 0 else "#C75D4D" for d in deltas]
        ax.axhline(0, color="#222222", linewidth=0.8)
        ax.bar(x, deltas, color=delta_colors)
        ax.set_xticks(x)
        ax.set_xticklabels(nulls, rotation=25, ha="right")
        ax.set_ylabel(f"change vs {base} (catchments)")
        ax.set_title(f"{cand} stability", loc="left")
        ax.grid(axis="y", color="#E6E6E6", linewidth=0.8)
    else:
        ax.axis("off")

    for ext in ["png", "pdf", "svg"]:
        fig.savefig(FIGURES / f"nature_extended_surrogate_precision_{output_prefix}.{ext}", dpi=300)
    plt.close(fig)


def write_note(summary: pd.DataFrame, baseline: Path, candidate: Path, output_prefix: str) -> None:
    NOTES.mkdir(parents=True, exist_ok=True)
    runs = list(dict.fromkeys(summary["run"].tolist()))
    lines = [
        "# CAMELS surrogate precision comparison",
        "",
        "Date: 2026-05-31",
        "",
        "## Inputs",
        "",
        f"- Baseline: `{baseline}`.",
        f"- Candidate: `{candidate}`.",
        "",
        "## Surrogate-count audit",
        "",
    ]
    for run in runs:
        sub = summary[summary["run"] == run]
        med = [int(x) for x in sorted(sub["median_n_surrogates"].dropna().unique())]
        lines.append(f"- {run}: median surrogate counts per null = {med}.")
    lines.extend(["", "## Rejection counts", ""])
    for _, row in summary.iterrows():
        lines.append(
            f"- {row['run']} {row['null_model']}: "
            f"{int(row['p_le_1_over_n_count'])}/{int(row['n_total_rows'])} at p <= 1/N; "
            f"{int(row['p_zero_count'])}/{int(row['n_total_rows'])} strict no-exceedance; "
            f"median N={row['median_n_surrogates']:.0f}."
        )
    lines.extend(
        [
            "",
            "## Manuscript implication",
            "",
            "The higher-N run can replace the N=15 counts only if every reported null model has the expected median surrogate count and passes consistency checks. Resolution-limited counts use a stricter threshold when N increases, so direct percentage changes should be interpreted as precision hardening rather than a change in the observed statistic.",
            "",
            "## Outputs",
            "",
            f"- `reports/tables/{output_prefix}.csv`",
            f"- `reports/figures/nature_extended_surrogate_precision_{output_prefix}.*`",
        ]
    )
    (NOTES / f"{output_prefix}.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    baseline = Path(args.baseline)
    candidate = Path(args.candidate)
    if not baseline.exists():
        raise FileNotFoundError(baseline)
    if not candidate.exists():
        raise FileNotFoundError(candidate)

    base_summary = summarize_table(baseline, "N15")
    cand_summary = summarize_table(candidate, args.candidate_label)
    summary = pd.concat([base_summary, cand_summary], ignore_index=True)

    TABLES.mkdir(parents=True, exist_ok=True)
    out_csv = TABLES / f"{args.output_prefix}.csv"
    summary.to_csv(out_csv, index=False)
    write_figure(summary, args.output_prefix)
    write_note(summary, baseline, candidate, args.output_prefix)
    print(f"Saved {out_csv.relative_to(ROOT)}")
    print(f"Saved {(NOTES / (args.output_prefix + '.md')).relative_to(ROOT)}")


if __name__ == "__main__":
    main()
