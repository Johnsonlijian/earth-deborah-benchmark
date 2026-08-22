# Reproducible Runbook

This runbook separates checks that use only released derived tables from full
recomputations that require official third-party data. The repository excludes
raw datasets and active submission documents.

## 1. Environment and tests

Python 3.10 or newer is recommended.

```powershell
python -m pip install -e ".[dev,model]"
python -m pytest
```

The optional `model` dependencies are needed for the GR4J/HBV-Edu/LSTM model
layer. Figure-only verification requires pandas, NumPy, SciPy, and Matplotlib.

## 2. Released-table figure reproduction

The following commands do not require raw CAMELS data:

```powershell
python scripts\run_r65_essd_figures.py
python paper_figures\src\render_essd_fig05_decision_test.py
```

They read `source_data/` and write `figures/essd/`. The Figure 5 renderer also
writes SHA-256 input/output provenance to
`figures/essd/fig05_provenance.md`.

Verify that these files exist after execution:

```text
figures/essd/
  fig01_dataset_overview.png
  fig01_dataset_overview.svg
  fig02_dispersion_intervals.png
  fig02_dispersion_intervals.svg
  fig03_null_calibration_ladder.png
  fig03_null_calibration_ladder.svg
  fig04_model_layer.png
  fig04_model_layer.svg
  fig05_decision_test.png
  fig05_decision_test.pdf
  fig05_decision_test.svg
  fig05_provenance.md
  supp_figS1_decision_delta_ecdf.png
  supp_figS1_decision_delta_ecdf.svg
```

## 3. Data acquisition boundary

Official data sources, versions, links, and redistribution boundaries are in:

- `DATASETS_AND_LINKS.csv`;
- `source_data/dataset_registry.csv`.

Do not place downloaded data in Git. Use ignored local paths:

```text
data/raw/
data/external/
data/interim/
```

For CAMELS-GB v2 model-layer analyses, the expected local layout is:

```text
data/external/camels_gb_v2/
  hydromet_daily/
    camels_gb_v2_hydromet_daily_timeseries_*_19701001-20220930.csv
  attributes/
    camels_gb_v2_hydrologic_attributes.csv
    camels_gb_v2_topographic_attributes.csv
    camels_gb_v2_climatic_attributes.csv
```

Confirm the provider record and DOI before downloading. Do not redistribute
the downloaded archive from this repository.

## 4. Artifact-floor reproduction

After the required official archive files are available locally, the main
stochastic-memory controls can be recomputed with:

```powershell
python scripts\run_r33_matched_ar1_ensemble.py --n-surrogates 100 --workers 16 --output-prefix r33_matched_ar1_ensemble
python scripts\run_r33_null_calibrated_ensemble_summary.py --ensemble-prefix r33_matched_ar1_ensemble
python scripts\run_r33_artifact_floor_main_figure.py
python scripts\run_r37_gauge_matched_analytic_nulls.py
```

Reference checks from the released R33 summary:

| Archive | Observed De reduction | Matched AR(1) median | Null-adjusted excess |
| --- | ---: | ---: | ---: |
| CAMELS-US | 23.69% | 69.03% | -45.34 percentage points |
| CAMELS-GB v2 | 14.69% | 65.67% | -50.97 percentage points |

The correct interpretation is that observed coordinate compression is
positive but the strict matched-AR(1) mechanism-excess gate fails.

## 5. Open model-layer reproduction

With CAMELS-GB v2 and optional model dependencies installed:

```powershell
python scripts\run_r39_open_model_intercomparison.py --workers 4 --rrmpg-samples 128 --lstm-epochs 5 --lstm-batches 260 --lstm-batch-size 64 --lstm-hidden 32 --lstm-seq-len 180
```

The formal run used seed `20260608`. Expected valid summary counts are:

- RRMPG GR4J: 579;
- RRMPG HBV-Edu: 602;
- global LSTM: 658;
- seasonal AR(1) reference: 666.

The generating script writes dense curves under `reports/tables/`. These are
not tracked in Git. Keep the exact run manifest and dependency versions with
any recomputation.

## 6. Decision-test reproduction

The released result can be inspected in:

- `source_data/r62_decision_poc_gb_variant_summary.csv`;
- `source_data/r62_decision_poc_gb_ind3_gauge_variants.csv`;
- `source_data/r62_decision_poc_gb_threshold_grid.csv`;
- `source_data/r62_decision_poc_gb_rank_sanity.csv`.

To recompute it from the R39 products:

```powershell
python scripts\run_r62_decision_poc_gb.py --reuse-obs-cache --seed 62 --n-boot 5000
```

The script first looks for R39 inputs in `source_data/`, then in
`reports/tables/`. A full rerun therefore requires the untracked dense
`r39_open_model_intercomparison_curves.csv` output. The released observed-curve
cache can be reused; regenerating it also requires the official CAMELS-GB v2
daily files.

The public protocol transcription is
`DECISION_TEST_PROTOCOL_PUBLIC_RECORD.md`. Preserve the reported null result:
the beta(De)-selected model did not systematically reduce held-out timescale
error relative to the NSE-selected model on the registered disagreement
subset.

## 7. Cross-archive and hardening modules

Representative commands are:

```powershell
python scripts\make_r35_five_archive_fig2.py
python scripts\make_r35_robustness_fig4.py
python scripts\run_final_extreme_hardening.py --cluster-draws 1000
python scripts\run_final_model_ensemble_hardening.py --ar1-draws 50 --rrmpg-samples 512 --workers 4
```

These modules have different raw-data requirements. Consult the dataset
registry before execution and do not interpret synthetic or stochastic
controls as observed hydrological mechanisms.

## 8. Release verification

Before a public commit or archival release:

1. run `python -m pytest`;
2. rerender the ESSD figures and inspect the PNGs;
3. run a clean scan for unresolved fields, credentials, local absolute paths, and
   manuscript/reviewer residue;
4. confirm that no tracked file exceeds GitHub's file-size limit;
5. regenerate `CHECKSUM_MANIFEST.csv` after the final file set is fixed;
6. verify `CITATION.cff` and `.zenodo.json` without inventing a DOI;
7. keep the repository commit and any future archival dataset record as
   distinct, accurately typed objects.

The author manuscript, cover letter, reviewer materials, and submission portal
files are deliberately outside this public repository.
