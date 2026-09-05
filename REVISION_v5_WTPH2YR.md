# v5 WTPH2YR revision

This revision aligns the public repository with the final manuscript analysis.

## Main change

The August 2021–August 2023 NHANES survey-weighted temporal analysis uses `WTPH2YR` rather than `WTMEC2YR`, because the CKD screening phenotype depends on serum creatinine. The 2017–2018 development weighted analysis continues to use `WTMEC2YR`.

This change affects only the secondary survey-weighted evaluation. It does **not** alter model fitting, nested cross-validation, final model selection, probability calibration, frozen global threshold selection, or the primary unweighted temporal-validation results.

## Revised weighted temporal results

- Prevalence: 14.06%
- ROC-AUC: 0.7938
- PR-AUC: 0.4378
- Brier score: 0.0989
- Log loss: 0.3309
- Accuracy: 0.7201
- Precision: 0.3019
- Sensitivity: 0.7549
- Specificity: 0.7144
- F1: 0.4313
- ECE: 0.0128

## Primary results retained

- Development calibrated OOF ROC-AUC: 0.7890
- Temporal ROC-AUC: 0.7892
- Temporal PR-AUC: 0.4648
- Temporal Brier score: 0.1160
- Frozen global threshold: 0.131182
- Temporal sensitivity: 0.8118
- Temporal specificity: 0.6202
- Temporal calibration intercept: 0.0238
- Temporal calibration slope: 1.0756
- Temporal ECE: 0.0128

## Manuscript interpretation updates

The final manuscript treats age- and diabetes-stratified threshold analyses as exploratory sensitivity analyses, describes threshold changes as changes in operating characteristics rather than discrimination, and keeps the single global threshold as the primary analysis.
