# Earth Deborah Benchmark

Reproducible code and derived outputs for a null-calibrated benchmark of
streamflow timescale structure. The manuscript tests whether a
memory-normalized frequency coordinate,

```text
De = tau_acf * f
```

can compare local streamflow spectral-slope curves across catchments and expose
hydrological model timescale-structure errors beyond NSE/KGE-style hydrograph
skill.

## Scientific Scope

The current paper is **not** a proof that discharge spectra identify storage
mechanisms. It is a falsification-first benchmark:

- observed De-axis reductions are tested against support-matched bins,
  shuffled-memory and proxy-coordinate nulls;
- strict gauge-matched AR(1) gates are used as an artifact floor;
- region-cluster, forcing-side and slow-memory support audits are used as HESS
  hardening checks;
- independent groundwater, tracer and storage-state data are treated as
  compatibility and boundary tests, not causal closure;
- a same-split open GR4J/HBV-Edu/LSTM model benchmark asks whether
  beta(De)-curve skill ranks outputs differently from hydrograph skill, with
  seasonal AR(1) draw uncertainty and RRMPG calibration-budget sensitivity.

The strongest current boundary is explicit: CAMELS-US and CAMELS-GB v2 have
positive observed De reductions, but negative null-adjusted excess relative to
the gauge-matched AR(1) artifact floor.

## Repository Boundary

This repository is intended to be the clean public reproducibility route. It may
contain code, configuration, derived non-sensitive tables, generated figures,
source-data indexes, dataset links, licenses and runbooks.

It must not contain raw third-party data archives, credentials, active
submission manuscripts, cover letters, reviewer-response drafts, internal
rounds, private author/funding files or unclear-redistribution material.

Raw public datasets should be obtained from the official sources listed in
`DATASETS_AND_LINKS.csv` and `source_data/dataset_registry.csv` in the public
release. In the full author submission workspace, the same registry is mirrored
in the private submission source-data folder and the accompanying HESS
reproducibility support package.

## Quick Start

Windows PowerShell:

```powershell
cd earth-deborah-benchmark
python -m pip install -e ".[dev,model]"
python -m pytest
python scripts\run_r33_matched_ar1_ensemble.py --n-surrogates 100 --workers 16 --output-prefix r33_matched_ar1_ensemble
python scripts\run_r33_null_calibrated_ensemble_summary.py --ensemble-prefix r33_matched_ar1_ensemble
python scripts\run_r33_artifact_floor_main_figure.py
python scripts\make_r35_five_archive_fig2.py
python scripts\make_r35_robustness_fig4.py
python scripts\run_r39_open_model_intercomparison.py --workers 4 --rrmpg-samples 128 --lstm-epochs 5 --lstm-batches 260 --lstm-batch-size 64 --lstm-hidden 32 --lstm-seq-len 180
python scripts\run_final_extreme_hardening.py --cluster-draws 1000
python scripts\run_final_model_ensemble_hardening.py --ar1-draws 50 --rrmpg-samples 512 --workers 4
```

Linux/macOS shell:

```bash
cd earth-deborah-benchmark
python -m pip install -e ".[dev,model]"
python -m pytest
python scripts/run_r33_matched_ar1_ensemble.py --n-surrogates 100 --workers 16 --output-prefix r33_matched_ar1_ensemble
python scripts/run_r33_null_calibrated_ensemble_summary.py --ensemble-prefix r33_matched_ar1_ensemble
python scripts/run_r33_artifact_floor_main_figure.py
python scripts/make_r35_five_archive_fig2.py
python scripts/make_r35_robustness_fig4.py
python scripts/run_r39_open_model_intercomparison.py --workers 4 --rrmpg-samples 128 --lstm-epochs 5 --lstm-batches 260 --lstm-batch-size 64 --lstm-hidden 32 --lstm-seq-len 180
python scripts/run_final_extreme_hardening.py --cluster-draws 1000
python scripts/run_final_model_ensemble_hardening.py --ar1-draws 50 --rrmpg-samples 512 --workers 4
```

Expected minimal outputs:

- `reports/tables/r33_null_calibrated_ensemble_summary.csv`
- `figures/fig01_mechanism_evidence.{pdf,png,svg}`
- `figures/fig02_five_archive_alignment_boundary.{pdf,png,svg}`
- `figures/fig03_robustness_support_checks.{pdf,png,svg}`
- `figures/fig04_artifact_floor_boundary.{pdf,png,svg}`
- `figures/fig05_multimodel_benchmark.{pdf,png,svg}`
- `reports/tables/r39_open_model_intercomparison_summary.csv`
- `reports/figures/r39_open_model_intercomparison.{pdf,png,svg}`
- `reports/tables/final_extreme_hardening_cluster_bootstrap_summary.csv`
- `reports/tables/final_extreme_hardening_forcing_control_summary.csv`
- `reports/tables/final_model_ensemble_hardening_summary.csv`
- `reports/figures/final_extreme_hardening_windtunnel.{pdf,png,svg}`
- `reports/figures/final_model_ensemble_hardening.{pdf,png,svg}`
- `source_data/source_data_index.csv` in the public release, or
  the full author workspace when a local submission-source mirror is prepared
- `source_data/dataset_registry.csv`

The R39 model benchmark requires the optional model dependencies and local
CAMELS-GB v2 extraction. The expected author-workspace layout is:

```text
data/external/camels_gb_v2/
  hydromet_daily/
    camels_gb_v2_hydromet_daily_timeseries_*_19701001-20220930.csv
  attributes/
    camels_gb_v2_hydrologic_attributes.csv
    camels_gb_v2_topographic_attributes.csv
    camels_gb_v2_climatic_attributes.csv
```

The formal R39 run used 671 hydromet daily CSV files, 128 RRMPG parameter
samples per gauge/model, seed `20260608`, and a global LSTM with 32 hidden
units, 180-day windows, five epochs and 260 batches per epoch. Expected valid
summary counts are RRMPG GR4J = 579, RRMPG HBV-Edu = 602, global LSTM = 658 and
seasonal AR(1) null = 666.

The final HESS hardening run used 1000 region-cluster bootstrap draws, a GB
forcing-side precipitation spectral control, slow-memory support audits, a
50-draw seasonal AR(1) model-null ensemble, a five-seed LSTM sensitivity check
and a 512-vs-128 RRMPG calibration-budget audit. The RRMPG budget audit
produced 1136 paired gauge/model deltas.

## Main Evidence Modules

| Module | Purpose | Representative scripts |
| --- | --- | --- |
| CAMELS-US/GB benchmark | Main De-axis variance and curve-distance tests | `run_camels_full.py`, `run_camels_gb_replication.py`, `analyze_convergent_curve_alignment_metric.py` |
| Artifact-floor gates | Support-matched bins, multi-draw gauge-matched AR(1), null-adjusted score | `analyze_r20_support_matched_sensitivity.py`, `run_r33_matched_ar1_ensemble.py`, `run_r31_gauge_level_ar1_boundary.py`, `run_r33_null_calibrated_ensemble_summary.py` |
| HESS hardening controls | Region-cluster uncertainty, forcing-side precipitation control, slow-memory support audit | `run_final_extreme_hardening.py` |
| Model benchmark | Same-split open GR4J/HBV-Edu/LSTM, conceptual-model history, memory-null comparison, AR(1) ensemble and RRMPG budget sensitivity | `run_r39_open_model_intercomparison.py`, `run_final_model_ensemble_hardening.py`, `run_r38_established_model_suite.py`, `run_r23_multimodel_hydrology_benchmark.py`, `run_r22_model_null_context.py` |
| Portability and boundaries | CAMELS-BR, CAMELS-AUS, CAMELS-DK, groundwater and tracer-facing checks | `run_r23_camels_br_third_archive.py`, `run_r25_camels_aus_fourth_archive.py`, `run_r27_camels_dk_groundwater_storage_validation.py`, `run_r28_tracer_compatibility_validation.py` |
| Main figure synthesis | Five-archive Fig. 2 and robustness Fig. 4 from traceable source tables | `make_r35_five_archive_fig2.py`, `make_r35_robustness_fig4.py` |
| Submission source-data map | Reviewer-facing file-to-claim/source/script index | `source_data/source_data_index.csv` |

## Data And Code Availability

The code is prepared for public release under the remote
`https://github.com/Johnsonlijian/earth-deborah-benchmark.git`. Repository
archival and DOI minting are handled through a GitHub release and Zenodo
integration.

No raw third-party archives are redistributed. Derived source-data files are
included only when they are non-sensitive and do not violate dataset terms.
Two dense derived curve tables are omitted from Git tracking because they exceed
GitHub file-size limits; see `source_data/LARGE_DERIVED_OUTPUTS.csv` for the
regeneration scripts.

## Current Prepared Artifacts

The current package round is R58-HESS-figure-5c-balance, dated 2026-06-11.

| Artifact | Path | Notes |
| --- | --- | --- |
| HESS review package | not stored in this public repository | The upload-ready manuscript/SI/source package is kept in the private author workspace. |
| Public reproducibility release | this repository | Code, tests, runbook, dataset registry, derived source data and generated figures. |

Final journal upload still requires author-only portal metadata: APC/billing
route, funding, competing interests, acknowledgements and any suggested or
excluded reviewers.

## Citation

Use `CITATION.cff` after the final title, author list and repository DOI are
confirmed. Dataset citations must use the official source DOIs/URLs listed in
`DATASETS_AND_LINKS.csv`.
