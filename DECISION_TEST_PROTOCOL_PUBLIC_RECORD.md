# Decision-Test Protocol: Public Scientific Record

This document is a public transcription of the scientific design frozen in the
author archive before the full CAMELS-GB decision-test run. It removes only
workflow-administration material that is irrelevant to scientific
reproduction. It is not, by itself, independent proof of the earlier
timestamp.

Source protocol metadata:

- initial timestamp: 2026-08-21 18:05 +08:00 (Asia/Shanghai);
- scientific amendments: v1.1 at 18:20, v1.2 at 18:45, and v1.3 at 19:00,
  all on 2026-08-21 and before the full run;
- SHA-256 of the retained source protocol:
  `2243b37a178361f38a5da4a30794116556b467fbf74d8d67c65ff91d5ba45172`;
- public transcription date: 2026-08-22.

## Primary question

For CAMELS-GB v2 gauges on the common R39 train/evaluation split, compare the
model selected by held-out NSE with the model selected by the beta(De) curve
distance. On gauges where those selections disagree, test whether the
beta(De)-selected model has a lower held-out timescale error,
`abs_log10_tau_error`, than the NSE-selected model.

The signed decision difference is:

```text
delta_tau_error = tau_error(distance-selected) - tau_error(NSE-selected)
```

Negative values favour the distance-selected model.

## Primary model set and sensitivity set

- primary modelling decision set: RRMPG GR4J, RRMPG HBV-Edu, and global LSTM;
- contextual five-output set: the three models above plus seasonal climatology
  and the seasonal AR(1) reference.

The three-model set is the main decision test because the seasonal AR(1)
reference inherits a timescale construction that can make its timescale error
artificially competitive. The five-output set remains visible as context.

## Registered summaries and thresholds

For each model set and each distance variant, report:

- number of valid gauges;
- paired median and mean decision difference;
- two-sided Wilcoxon signed-rank p-value;
- non-parametric bootstrap 95% interval for the paired median, using 5000
  bootstrap draws;
- fraction of disagreement gauges with a negative decision difference.

The directional reference threshold is a negative median whose 95% interval
excludes zero and a lower-error fraction of at least 0.55. Fractions at 0.50,
0.55, and 0.60 are all reported; no post-result threshold is selected.

## Distance variants

1. `beta_de`: released local beta(De) curve distance.
2. `beta_raw`: released local curve distance on raw frequency.
3. `scalar`: absolute difference between scalar spectral-slope summaries over
   De in [0.5, 2].
4. `de_const`: one common tau applied to observed and simulated curves. This
   was identified before the full run as mathematically equivalent to the raw
   frequency distance under the shared quantile-binning implementation and is
   retained as an equivalence check, not an independent ablation.
5. `de_shuff`: observed and simulated curves receive independently shuffled
   measured tau values from other gauges, with fixed seed 62.

## Inputs and outputs

Inputs:

- `r39_open_model_intercomparison_metrics.csv`;
- `r39_open_model_intercomparison_curves.csv` (dense regenerated output,
  omitted from Git because of size);
- `r39_open_model_intercomparison_rank_disagreement.csv`;
- official CAMELS-GB v2 daily series when the observed-curve cache is rebuilt.

Public derived outputs:

- `source_data/r62_decision_poc_gb_gauge_variants.csv`;
- `source_data/r62_decision_poc_gb_ind3_gauge_variants.csv`;
- `source_data/r62_decision_poc_gb_variant_summary.csv`;
- `source_data/r62_decision_poc_gb_threshold_grid.csv`;
- `source_data/r62_decision_poc_gb_rank_sanity.csv`;
- `source_data/r62_decision_poc_gb_obs_curves.csv`.

## Interpretation boundary

The test evaluates decision increment for held-out timescale error. It does not
test storage causality and does not establish a general model-selection rule.
The released result is null: the beta(De)-based selection did not meet the
registered improvement criterion. That result must remain visible in any reuse
of the released model-layer data product.
