# Figure 5 — bounded decision-test diagnostics

## Submission role

This is the final main-text technical-validation figure for the ESSD data
description. It makes the released model-layer products inspectable without
claiming that the tested spectral-distance rule is a validated model-selection
method.

## One-sentence takeaway

On the time-stamped disagreement subset, the released `\(\beta(De)\)`-based
selection rule did **not** improve held-out timescale error relative to
held-out NSE selection; the figure preserves that null result as a bounded
usage diagnostic.

## Evidence boundary

- Inputs are the released, derived R39 and R62 tables only; no raw CAMELS data
  are read or redistributed.
- Panel (a) is descriptive association, not evidence of causal storage
  control.
- Panels (b)–(c) report the time-stamped comparison on the disagreement
  subset. They do not establish a general model-selection rule.
- The figure must not be cited as evidence that `\(\beta(De)\)` improves
  hydrological-model performance.

## Panel contract

| Panel | Content | Source table | Visual encoding |
|---|---|---|---|
| a | Held-out NSE versus median absolute `\(\beta(De)\)` distance | `source_data/r39_open_model_intercomparison_metrics.csv` | Model-specific scatter, dashed descriptive trend |
| b | Difference in held-out timescale error: distance selection minus NSE selection | `source_data/r62_decision_poc_gb_variant_summary.csv` | Point estimate and reported 95% bootstrap interval; zero reference line |
| c | Share of gauges for which the distance-selected model has lower timescale error | `source_data/r62_decision_poc_gb_variant_summary.csv` | Point estimate with reference thresholds 0.50, 0.55, and 0.60 |

## Rendering contract

- Generate from `paper_figures/src/render_essd_fig05_decision_test.py`.
- Export SVG and PDF master files plus a 300 dpi PNG review copy.
- Use a colour-blind-safe categorical palette and deterministic programmatic
  rendering; do not add illustrative elements.
- Embed a provenance file with source and output SHA-256 checksums.
