"""Run the current full first-pass recomputation suite."""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd


def run_step(name: str, command: list[str], cwd: Path) -> dict[str, object]:
    """Run one recomputation step and return manifest metadata."""

    started = datetime.now(UTC)
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    ended = datetime.now(UTC)
    log_dir = cwd / "reports" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / f"{name}_stdout.log"
    stderr_path = log_dir / f"{name}_stderr.log"
    stdout_path.write_text(result.stdout, encoding="utf-8")
    stderr_path.write_text(result.stderr, encoding="utf-8")
    return {
        "step": name,
        "command": " ".join(command),
        "returncode": result.returncode,
        "started_at_utc": started.isoformat(),
        "ended_at_utc": ended.isoformat(),
        "duration_seconds": (ended - started).total_seconds(),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endtime", default=datetime.now(UTC).date().isoformat())
    parser.add_argument("--n-null", type=int, default=100)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-heavy-null", action="store_true")
    parser.add_argument("--skip-nwis", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cwd = Path(__file__).resolve().parents[1]
    py = sys.executable
    steps: list[tuple[str, list[str]]] = []
    if not args.skip_download:
        steps.append(
            (
                "download_first_pass_raw",
                [
                    py,
                    "scripts/download_first_pass_raw.py",
                    "--endtime",
                    args.endtime,
                    "--itslive-limit",
                    "100",
                ],
            )
        )
    steps.extend(
        [
            ("synthetic_day1", [py, "scripts/run_synthetic_day1.py"]),
            (
                "earthquake_sensitivity",
                [
                    py,
                    "scripts/run_usgs_earthquake_sensitivity.py",
                    "--starttime",
                    "2000-01-01",
                    "--endtime",
                    args.endtime,
                    "--minmagnitudes",
                    "5.0,5.5,6.0",
                    "--tau-days",
                    "7,30,365",
                    "--rules",
                    "D,MS",
                    "--output-prefix",
                    f"earthquake_sensitivity_2000_{args.endtime.replace('-', '')}",
                ],
            ),
            (
                "earthquake_regions",
                [
                    py,
                    "scripts/run_usgs_earthquake_regions.py",
                    "--input",
                    f"data/raw/usgs/comcat_global_m5_2000-01-01_{args.endtime}.geojson",
                    "--starttime",
                    "2000-01-01",
                    "--endtime",
                    args.endtime,
                    "--minmagnitudes",
                    "5.0,5.5",
                    "--tau-days",
                    "7,30,90,365",
                    "--output-prefix",
                    f"earthquake_regions_2000_{args.endtime.replace('-', '')}",
                ],
            ),
            ("landslide_glc_mvp", [py, "scripts/run_nasa_glc_landslide_mvp.py", "--output-prefix", "landslide_glc_mvp"]),
        ]
    )
    if not args.skip_heavy_null:
        steps.append(
            (
                "landslide_trigger_nulls",
                [
                    py,
                    "scripts/run_nasa_glc_trigger_nulls.py",
                    "--n-null",
                    str(args.n_null),
                    "--tau-days",
                    "90,180,365",
                    "--top-triggers",
                    "7",
                    "--min-trigger-events",
                    "100",
                    "--output-prefix",
                    "landslide_glc_trigger_nulls",
                ],
            )
        )
    steps.append(("river_demo_mvp", [py, "scripts/run_river_mvp.py", "--output-prefix", "river_demo_mvp"]))
    if not args.skip_nwis:
        steps.append(
            (
                "usgs_nwis_river_mvp",
                [
                    py,
                    "scripts/run_usgs_nwis_river_mvp.py",
                    "--end-date",
                    args.endtime,
                    "--output-prefix",
                    f"usgs_nwis_river_mvp_1980_{args.endtime.replace('-', '')}",
                ],
            )
        )
        steps.append(
            (
                "nwis_memory_scaling",
                [
                    py,
                    "scripts/run_nwis_memory_scaling.py",
                    "--n-null",
                    "50",
                    "--output-prefix",
                    f"nwis_memory_scaling_1980_{args.endtime.replace('-', '')}",
                ],
            )
        )
    steps.append(("glacier_demo_mvp", [py, "scripts/run_itslive_glacier_mvp.py", "--output-prefix", "glacier_demo_mvp"]))
    steps.append(("falsification_table", [py, "scripts/build_falsification_table.py"]))

    rows: list[dict[str, object]] = []
    for name, command in steps:
        print(f"Running {name}...")
        row = run_step(name, command, cwd)
        rows.append(row)
        if row["returncode"] != 0:
            manifest = pd.DataFrame(rows)
            out = cwd / "reports" / "tables" / "recompute_manifest.csv"
            out.parent.mkdir(parents=True, exist_ok=True)
            manifest.to_csv(out, index=False)
            raise SystemExit(f"Step failed: {name}; see {row['stderr_log']}")

    manifest = pd.DataFrame(rows)
    out = cwd / "reports" / "tables" / "recompute_manifest.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out, index=False)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
