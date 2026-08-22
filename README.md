# Earth Deborah Benchmark

Public code, derived source data, and deterministic figures for **A
null-calibrated spectral-memory benchmark for hydrological model timescale
diagnostics**.

The benchmark uses the dimensionless frequency coordinate

```text
De = tau_acf * f
```

to compare local streamflow spectral-slope curves across catchments and to
describe hydrological-model timescale errors alongside conventional hydrograph
scores. This repository is the software and reproducibility companion for an
Earth System Science Data (ESSD) manuscript in preparation; it is not the
manuscript or the dataset DOI record.

## Evidence boundary

The released analyses support a bounded, falsification-first benchmark:

- observed De-axis dispersion changes are compared with support-matched,
  shuffled-memory, analytic, and gauge-matched stochastic-memory controls;
- the archive-level effect is positive for several CAMELS archives but is not
  universal, with CAMELS-DK retained as a lowland boundary case;
- observed reductions for CAMELS-US and CAMELS-GB remain below the strict
  gauge-matched AR(1) artifact floor;
- the CAMELS-GB model layer compares GR4J, HBV-Edu, a global LSTM, seasonal
  climatology, and a seasonal AR(1) reference on a common split;
- the time-stamped decision test did **not** show that beta(De)-based model
  selection improved held-out timescale error relative to NSE-based selection.

These results do not establish storage causality, a universal spectral
collapse, or a validated operational model-selection rule.

## Public-release boundary

Included:

- analysis and figure-generation code;
- derived, non-sensitive CSV tables;
- generated SVG/PDF/PNG figures;
- official dataset links, source-data indexes, runbooks, and citation metadata.

Excluded:

- raw third-party datasets and downloaded archives;
- active or rejected manuscripts, cover letters, reviewer correspondence, and
  submission-system files;
- private workflow rounds, logs, credentials, author/funding records, and
  redistribution-uncertain material.

Obtain raw public data from `DATASETS_AND_LINKS.csv` and
`source_data/dataset_registry.csv`. Keep local raw files under ignored folders
such as `data/raw/` and `data/external/`.

## Quick verification from released tables

Windows PowerShell:

```powershell
python -m pip install -e ".[dev,model]"
python -m pytest
python scripts\run_r65_essd_figures.py
python paper_figures\src\render_essd_fig05_decision_test.py
```

Linux/macOS:

```bash
python -m pip install -e ".[dev,model]"
python -m pytest
python scripts/run_r65_essd_figures.py
python paper_figures/src/render_essd_fig05_decision_test.py
```

The figure commands read only `source_data/` and write to `figures/essd/`.
Expected ESSD-facing outputs are:

- `fig01_dataset_overview.{png,svg}`;
- `fig02_dispersion_intervals.{png,svg}`;
- `fig03_null_calibration_ladder.{png,svg}`;
- `fig04_model_layer.{png,svg}`;
- `fig05_decision_test.{png,pdf,svg}` and `fig05_provenance.md`;
- `supp_figS1_decision_delta_ecdf.{png,svg}`.

## Main reproducibility modules

| Module | Purpose | Representative scripts |
| --- | --- | --- |
| CAMELS-US/GB benchmark | De-axis variance and curve-distance tests | `run_camels_full.py`, `run_camels_gb_replication.py`, `analyze_convergent_curve_alignment_metric.py` |
| Artifact-floor gates | Support matching, stochastic-memory controls, null-adjusted score | `analyze_r20_support_matched_sensitivity.py`, `run_r33_matched_ar1_ensemble.py`, `run_r33_null_calibrated_ensemble_summary.py`, `run_r37_gauge_matched_analytic_nulls.py` |
| Cross-archive boundary | CAMELS-BR, CAMELS-AUS, CAMELS-DK, bootstrap and forcing/support audits | `run_r23_camels_br_third_archive.py`, `run_r25_camels_aus_fourth_archive.py`, `run_r27_camels_dk_groundwater_storage_validation.py`, `run_final_extreme_hardening.py` |
| Model layer | Same-split GR4J/HBV-Edu/LSTM and reference outputs | `run_r39_open_model_intercomparison.py`, `run_final_model_ensemble_hardening.py` |
| Decision test | Timescale-error comparison and coordinate ablations | `run_r62_decision_poc_gb.py` |
| ESSD figures | Released-table-only rendering | `run_r65_essd_figures.py`, `paper_figures/src/render_essd_fig05_decision_test.py` |

The dense R39 curve table is intentionally omitted from Git tracking because
of its size. `source_data/LARGE_DERIVED_OUTPUTS.csv` identifies large omitted
outputs and their generating scripts. A full R62 rerun requires that dense
table plus an authorized local CAMELS-GB v2 extraction; the released R62 tables
are sufficient to verify the reported decision-test result and regenerate its
figures.

## Decision-test provenance

`DECISION_TEST_PROTOCOL_PUBLIC_RECORD.md` is a public transcription of the
scientific design that was frozen before the full decision-test run. It records
the original protocol hash and amendment times, while clearly stating that the
transcription itself is not independent proof of the earlier timestamp. The
source protocol is excluded because it also contains private workflow
administration unrelated to scientific reproduction.

## Citation and archival status

Use `CITATION.cff` for the software repository. Dataset citations must use the
official DOIs or URLs in the dataset registry. No new dataset DOI is claimed in
this repository; insert a manuscript-side DOI only after an archival record
has actually been minted.

See `REPRODUCIBLE_RUNBOOK.md` for the detailed execution route and
`RELEASE_NOTES_ESSD_R68.md` for the 2026-08-22 public update.
