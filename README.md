# Null-calibrated streamflow spectral-memory benchmark

Companion code and reproducibility materials for:

**A benchmark dataset of null-calibrated streamflow spectral-memory diagnostics for five CAMELS-family archives, with open rainfall-runoff model outputs for CAMELS-GB v2**

The Dataset covers 3,010 gauges in five CAMELS-family archives and includes
open rainfall-runoff model-comparison outputs for CAMELS-GB v2. It provides
memory-time estimates, local spectral-slope curves, uncertainty summaries and
matched stochastic references.

## Data and code

- Dataset version 1.0.0: [https://doi.org/10.5281/zenodo.22055294](https://doi.org/10.5281/zenodo.22055294).
- Exact code accompanying that Dataset:
  [dataset_release/v1.0.0/code](dataset_release/v1.0.0/code).
- File identity: [CODE_SHA256.csv](dataset_release/v1.0.0/CODE_SHA256.csv).
- Dataset reconstruction guide:
  [DATASET_RUNBOOK.md](dataset_release/v1.0.0/DATASET_RUNBOOK.md).
- Dataset input sources:
  [DATASETS_AND_LINKS.csv](dataset_release/v1.0.0/DATASETS_AND_LINKS.csv).

The Dataset archive supplies the full derived-data product and source mapping.
The versioned code directory matches its archived code exactly. Repository-root
scripts and source_data/ retain the earlier public implementation; the
historical commit 6d5804208b46a39cb26aa8650a3eac8cd442e8e1 remains accessible.
Use the versioned directory for the Dataset release, or the historical commit
for an earlier pinned analysis. Their identities are not interchangeable.

## Reuse

Read [REPRODUCIBLE_RUNBOOK.md](REPRODUCIBLE_RUNBOOK.md) before running scripts.
Historical analysis paths differ from the deposit layout. Raw CAMELS files are
not redistributed; obtain permitted inputs from the original providers when
rerunning the raw-data pipeline. Released derived tables can be inspected
without downloading those raw archives.

The product supports frequency-specific characterization and model comparison.
Archive applicability limits and the preregistered negative model-selection
result are retained. It does not establish a causal storage mechanism or a
universal spectral model-ranking rule.

## Citation, licences and article status

Author: Lijian Ren, Inner Mongolia University of Technology.
Use [CITATION.cff](CITATION.cff) for this software repository and cite the Dataset
separately:

Ren, L.: A benchmark dataset of null-calibrated streamflow spectral-memory diagnostics for five CAMELS-family archives, with open rainfall-runoff model outputs for CAMELS-GB v2, Zenodo [data set], version 1.0.0, https://doi.org/10.5281/zenodo.22055294, 2026.

Code is licensed under MIT. Dataset-derived data and documentation use CC BY
4.0, with source-specific attribution notices in the Zenodo archive.
No third-party raw archives, active manuscripts or editorial correspondence
are added to this repository.

The associated ESSD article has been approved for preprint posting and public
discussion; final journal acceptance and article publication are not claimed.
