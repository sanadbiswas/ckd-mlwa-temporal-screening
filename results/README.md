# Final v5 result summary

These files correspond to the final manuscript analysis after the NHANES August 2021–August 2023 survey-weight revision.

## Weighting specification

- Development (NHANES 2017–2018): `WTMEC2YR`
- Temporal validation (NHANES August 2021–August 2023): `WTPH2YR`

Survey weights are used only for secondary population-weighted evaluation and do not enter model fitting, calibration fitting, or threshold selection.

## Files

- `final_metrics.csv` — principal calibrated development OOF and temporal-validation metrics.
- `survey_weighted_performance_v5.csv` — revised weighted performance used in the final manuscript.
- `final_thresholds.csv` — frozen global threshold and exploratory age-/diabetes-stratified thresholds.

## Key revised temporal weighted values

- weighted prevalence: 0.1406
- ROC-AUC: 0.7938
- PR-AUC: 0.4378
- Brier score: 0.0989
- log loss: 0.3309
- accuracy: 0.7201
- precision: 0.3019
- sensitivity: 0.7549
- specificity: 0.7144
- F1: 0.4313
- ECE: 0.0128

The primary unweighted temporal results remain ROC-AUC 0.7892, PR-AUC 0.4648, Brier 0.1160, sensitivity 0.8118, and specificity 0.6202 at the globally frozen threshold 0.131182.
