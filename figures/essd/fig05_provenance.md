# Provenance — ESSD Figure 5

This figure is a deterministic matplotlib rendering of release-eligible
derived tables. It contains no generative or illustrative visual component.

## Build command

```text
python paper_figures/src/render_essd_fig05_decision_test.py
```

## Input checksums

- `source_data/r39_open_model_intercomparison_metrics.csv`: `7ba16b7088a3e2e7105eefd6b461f1b26d756940c27812c2b36b90d220c4f15e`
- `source_data/r62_decision_poc_gb_variant_summary.csv`: `673d508f7e81788f166ee7faaaa8ca51a70b1c3be5dd9b305080200d6e26965b`

## Output checksums

- `figures/essd/fig05_decision_test.png`: `1197c15f2f537ccc72ea460f895db58613f182e168abf3b29cd32cefe8c8f1a9`
- `figures/essd/fig05_decision_test.pdf`: `063b13146600761a8669228a768debf15fea3d64aa5618488499ed1438948ba0`
- `figures/essd/fig05_decision_test.svg`: `1bf94b7808585101e7952e3e722a65205e18a6b7e92d5eb4ab326362dbf94b1c`

## Scope boundary

Panel (a) is descriptive. Panels (b) and (c) report the
time-stamped disagreement-subset comparison and preserve its null
result; they do not validate a general model-selection rule.
