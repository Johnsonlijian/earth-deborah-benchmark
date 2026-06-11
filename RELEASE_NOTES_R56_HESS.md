# Release notes R56_HESS

R56 is the final HESS pre-submission reproducibility release after the deep
figure/source-package hardcheck.

Increment relative to the prior public release:

- Public main-figure generator now runs inside the public repository and reads
  `source_data/` directly.
- Final main figures exported as editable PDF/SVG/PNG under `figures/`.
- Obsolete release notes and stale package-round labels removed from the public
  release surface.
- Seed registry now points to the active model-ensemble sensitivity audit rather
  than the superseded model-consequence companion analysis.
- Public code comments and test names cleaned of early sprint and unfinished
  scaffold wording.

The release contains derived non-sensitive source data and reproducible code
only. Raw third-party datasets, active submission manuscripts, cover letters,
internal review rounds and private files are not included.
