"""Build a clean public reproducibility release bundle.

The public bundle is separate from the submission package. It includes code,
configuration, runbooks, dataset registries, selected derived source data and
generated figures, while excluding raw third-party data, active manuscripts,
submission cover letters, internal rounds, logs and private review material.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "project_completion" / "public_reproducibility_release"

TOP_FILES = [
    ".gitattributes",
    ".gitignore",
    "README.md",
    "REPRODUCIBLE_RUNBOOK.md",
    "LICENSE",
    "CITATION.cff",
    ".zenodo.json",
    "DATASETS_AND_LINKS.csv",
    "pyproject.toml",
    "environment.yml",
]
TOP_DIRS = ["configs", "src", "scripts", "tests"]
SOURCE_DATA = ROOT / "source_data"
if not SOURCE_DATA.exists():
    SOURCE_DATA = ROOT / "source_data"
MAIN_FIGURES = ROOT / "figures"
if not MAIN_FIGURES.exists():
    MAIN_FIGURES = ROOT / "figures"
FIGURE_KEEP = [
    "fig01_mechanism_evidence",
    "fig02_five_archive_alignment_boundary",
    "fig03_robustness_support_checks",
    "fig04_artifact_floor_boundary",
    "fig05_multimodel_benchmark",
    "r33_matched_ar1_ensemble",
    "r36_multiscale_process_family_nulls",
    "r37_artifact_floor_boundary",
    "r37_gauge_matched_analytic_nulls",
    "final_all_archive_ar1_floor",
]
FIGURE_RENAME: dict[str, str] = {}

FORBIDDEN_PARTS = {
    ".git",
    "raw",
    "external",
    "logs",
    "rounds",
    "submission_clean",
    "paper_attack",
    "multi_model_review",
    "writing_audit",
    "__pycache__",
}
FORBIDDEN_SUFFIXES = {".zip", ".7z", ".rar", ".tar", ".gz", ".shp", ".shx", ".dbf", ".prj", ".sbn", ".sbx"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def reset_out() -> None:
    resolved = OUT_ROOT.resolve()
    parent = (ROOT / "project_completion").resolve()
    if parent not in resolved.parents and resolved != parent:
        raise RuntimeError(f"Refusing to reset outside project_completion: {resolved}")
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    OUT_ROOT.mkdir(parents=True)


def copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def ignore_for_copy(_dir: str, names: list[str]) -> set[str]:
    ignored: set[str] = set()
    for name in names:
        lower = name.lower()
        if lower in {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}:
            ignored.add(name)
        if lower.endswith((".pyc", ".pyo", ".log", ".zip", ".7z", ".tar", ".gz")):
            ignored.add(name)
        if "submission" in lower or "cover_letter" in lower:
            ignored.add(name)
    return ignored


def copy_public_tree(src_name: str) -> None:
    src = ROOT / src_name
    if not src.exists():
        return
    shutil.copytree(src, OUT_ROOT / src_name, ignore=ignore_for_copy)
    scripts = OUT_ROOT / "scripts"
    if scripts.exists():
        rename_pairs: dict[str, str] = {}
        for old, new in rename_pairs.items():
            old_path = scripts / old
            if old_path.exists():
                new_path = scripts / new
                shutil.copy2(old_path, new_path)
                old_path.unlink()


def copy_source_data() -> None:
    dst = OUT_ROOT / "source_data"
    shutil.copytree(SOURCE_DATA, dst, ignore=ignore_for_copy)
    rename_prefixes: dict[str, str] = {}
    for path in list(dst.iterdir()):
        if not path.is_file():
            continue
        for old, new in rename_prefixes.items():
            if path.name.startswith(old):
                path.rename(dst / path.name.replace(old, new, 1))
                break
    text_replacements = {
        "r42_hess_extreme_hardening": "final_extreme_hardening",
        "r42_model_ensemble_hardening": "final_model_ensemble_hardening",
        "HESS hardening:": "Robustness hardening:",
        "Methods: HESS hardening controls": "Methods: spatial-dependence, forcing-side and slow-memory controls",
        "HESS model ensemble": "Model ensemble",
    }
    for path in [dst / "README.md", dst / "source_data_index.csv"]:
        if path.exists():
            text = path.read_text(encoding="utf-8")
            for old, new in text_replacements.items():
                text = text.replace(old, new)
            text = re.sub(r"Supplementary Figs\. (\d+)-(\d+)", r"Figs. S\1-S\2", text)
            text = re.sub(r"Supplementary Fig\. (\d+)", r"Fig. S\1", text)
            text = re.sub(r"Supplementary Table (\d+)", r"Table S\1", text)
            path.write_text(text, encoding="utf-8", newline="\n")


def copy_figures() -> None:
    dst = OUT_ROOT / "figures"
    dst.mkdir(parents=True, exist_ok=True)
    for stem in FIGURE_KEEP:
        out_stem = FIGURE_RENAME.get(stem, stem)
        copied = False
        for ext in (".pdf", ".svg", ".png"):
            src = MAIN_FIGURES / f"{stem}{ext}"
            if src.exists():
                copy_file(src, dst / f"{out_stem}{ext}")
                copied = True
        if copied:
            continue
        for ext in (".pdf", ".svg", ".png"):
            src = ROOT / "reports" / "figures" / f"{stem}{ext}"
            if src.exists():
                copy_file(src, dst / f"{out_stem}{ext}")


def scan_out() -> list[str]:
    bad: list[str] = []
    for path in OUT_ROOT.rglob("*"):
        rel = path.relative_to(OUT_ROOT)
        parts = {p.lower() for p in rel.parts}
        if rel.parts and rel.parts[0].lower() == "data":
            bad.append(str(rel))
        if FORBIDDEN_PARTS & parts:
            bad.append(str(rel))
        if path.is_file() and path.suffix.lower() in FORBIDDEN_SUFFIXES:
            bad.append(str(rel))
        if path.is_file() and "submission" in path.name.lower():
            bad.append(str(rel))
    return sorted(set(bad))


def write_checksum_manifest() -> None:
    rows = ["path,bytes,sha256"]
    for path in sorted(OUT_ROOT.rglob("*")):
        if path.is_file() and ".git" not in path.parts:
            rel = path.relative_to(OUT_ROOT).as_posix()
            rows.append(f"{rel},{path.stat().st_size},{sha256(path)}")
    (OUT_ROOT / "CHECKSUM_MANIFEST.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")


def write_release_metadata(tag: str) -> None:
    seed_rows = [
        "analysis,script,seed_or_determinism,notes",
        "matched_ar1_ensemble,scripts/run_r33_matched_ar1_ensemble.py,20260604,US/GB 100-draw stochastic artifact floor",
        "open_model_intercomparison,scripts/run_r39_open_model_intercomparison.py,20260608,CAMELS-GB GR4J/HBV-Edu/global-LSTM diagnostic split and initialization",
        "model_ensemble_sensitivity,scripts/run_final_model_ensemble_hardening.py,20260608 plus analysis-level deterministic summaries,Seasonal AR(1) draw uncertainty, five-seed LSTM sensitivity and RRMPG budget audit",
        "all_archive_analytic_ar1,scripts/run_final_all_archive_ar1_floor.py,deterministic,Analytic companion check; not a stochastic floor",
        "figure_generation,various scripts,deterministic unless noted,Generated from derived source-data CSVs",
    ]
    (OUT_ROOT / "random_seed_registry.csv").write_text("\n".join(seed_rows) + "\n", encoding="utf-8")
    (OUT_ROOT / "GITHUB_RELEASE_CHECKLIST.md").write_text(
        "\n".join(
            [
                "# GitHub release checklist",
                "",
                "Intended repository: https://github.com/Johnsonlijian/earth-deborah-benchmark",
                f"Recommended release tag: {tag}",
                "",
                "1. Create the public GitHub repository under Johnsonlijian.",
                "2. Push this release-ready directory only; do not push raw data, submission manuscripts, cover letters, rounds or logs.",
                "3. Create a GitHub release using the same tag.",
                "4. Connect the release to Zenodo and mint a DOI.",
                "5. Insert the final GitHub release URL and Zenodo DOI into manuscript Data/Code availability.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (OUT_ROOT / "ZENODO_DEPOSITION_METADATA.md").write_text(
        "\n".join(
            [
                "# Zenodo deposition metadata",
                "",
                "Title: A null-calibrated spectral-memory benchmark for hydrological model timescale diagnostics: reproducibility package",
                "Creators: Lijian Ren (ORCID: 0000-0003-1629-4368)",
                "Description: Code, configuration, derived non-sensitive source data, generated figures and runbook for the HESS-targeted streamflow spectral-memory benchmark study. Raw third-party CAMELS-family, groundwater, chemistry and tracer datasets are not redistributed.",
                "License: see LICENSE",
                "Related identifiers: GitHub repository URL is https://github.com/Johnsonlijian/earth-deborah-benchmark. Add the manuscript DOI after journal publication or preprint deposition.",
                "",
                "Do not upload raw third-party datasets or active submission manuscripts.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (OUT_ROOT / f"RELEASE_NOTES_{tag}.md").write_text(
        "\n".join(
            [
                f"# Release notes {tag}",
                "",
                "Release-ready reproducibility package for the streamflow spectral-memory benchmark.",
                "",
                "Included hardening increments:",
                "",
                "- Derived source-data tables and model consequence diagnostic.",
                "- All-archive analytic AR(1) companion source data.",
                "- Random-seed registry.",
                "- Checksum manifest.",
                "- GitHub/Zenodo release checklist.",
                "",
                "This package is release-ready but not yet DOI-minted.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def make_zip(tag: str) -> Path:
    zip_path = ROOT / "project_completion" / f"earth-deborah-benchmark_public_reproducibility_{tag}.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in sorted(OUT_ROOT.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(OUT_ROOT).as_posix())
    return zip_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="R58_HESS")
    args = parser.parse_args()

    reset_out()
    for name in TOP_FILES:
        copy_file(ROOT / name, OUT_ROOT / name)
    for name in TOP_DIRS:
        copy_public_tree(name)
    copy_source_data()
    copy_figures()
    write_release_metadata(args.tag)

    readme = OUT_ROOT / "PUBLIC_RELEASE_CONTENTS.md"
    readme.write_text(
        "\n".join(
            [
                "# Public Reproducibility Release Contents",
                "",
                "This bundle is prepared for a public GitHub/Zenodo-style reproducibility release.",
                "It intentionally excludes active submission manuscripts, cover letters, internal review rounds, logs, raw third-party datasets and downloaded archives.",
                "",
                "Included:",
                "",
                "- code, tests and configuration;",
                "- dataset registry and reproducible runbook;",
                "- derived non-sensitive source-data CSVs;",
                "- generated publication figures;",
                "- license and citation metadata.",
                "",
                "Human-only before DOI finalization: confirm Zenodo account authorization or GitHub-Zenodo integration, then record the minted archival DOI.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    bad = scan_out()
    if bad:
        raise RuntimeError("Forbidden public-release entries:\n" + "\n".join(bad[:50]))
    write_checksum_manifest()

    zip_path = make_zip(args.tag)
    file_count = sum(1 for p in OUT_ROOT.rglob("*") if p.is_file())
    print(f"Public release dir: {OUT_ROOT}")
    print(f"Files: {file_count}")
    print(f"Zip: {zip_path}")
    print(f"Bytes: {zip_path.stat().st_size}")
    print(f"SHA256: {sha256(zip_path)}")


if __name__ == "__main__":
    main()
