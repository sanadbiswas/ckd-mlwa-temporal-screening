# Temporal Validation of a Non-Laboratory ML Framework for CKD Screening

Reproducibility materials for the manuscript:

> Dutta S, Lamsal C, Deb P, Sikder S, Biswas S. *Temporal Validation of a Non-Laboratory Machine Learning Framework for Chronic Kidney Disease Screening in U.S. Adults.* Manuscript prepared for submission to *Machine Learning with Applications*.

## Overview

This repository contains the analysis code, cleaned development data, manuscript source, and reproducibility materials used to develop and temporally validate a non-laboratory machine-learning framework for prioritizing adults for confirmatory chronic kidney disease (CKD) testing using National Health and Nutrition Examination Survey (NHANES) data.

The primary analysis uses NHANES 2017–2018 for model development and NHANES August 2021–August 2023 for temporal validation. The outcome is a single-visit CKD-compatible screening phenotype defined by eGFR < 60 mL/min/1.73 m² or urine albumin-to-creatinine ratio (ACR) ≥ 30 mg/g. Kidney-specific laboratory variables used to define the phenotype are excluded from the predictor matrix.

## Final analysis version

The current reproducibility version is **v5 (WTPH2YR revision)**.

The principal change from the earlier research notebook is the survey-weighted temporal analysis:

- **2017–2018 development:** `WTMEC2YR`
- **2021–2023 temporal validation:** `WTPH2YR`

`WTPH2YR` is used for the 2021–2023 weighted sensitivity analysis because the CKD screening phenotype includes serum creatinine, a blood analyte. Survey weights are used only for secondary population-weighted evaluation; they are not used for model fitting, calibration fitting, or threshold selection.

## Repository contents

- `CKD_MLWA_Research_v5_WTPH2YR_REVISED.ipynb` — clean final notebook containing the complete v5 analysis pipeline.
- `CKD_MLWA_Research.ipynb` — earlier executed research notebook retained for provenance.
- `nhanes_ckd_risk_data.csv` — cleaned NHANES 2017–2018 development dataset.
- `results/final_metrics.csv` — principal development and temporal-validation metrics.
- `results/survey_weighted_performance_v5.csv` — revised survey-weighted performance using `WTPH2YR` for 2021–2023.
- `results/final_thresholds.csv` — frozen global and exploratory subgroup-specific operating thresholds.
- `manuscript/main.tex` — final MLWA manuscript source.
- `manuscript/supplementary_material.tex` — supplementary-material source.
- `manuscript/references.bib` — bibliography used by the final manuscript.
- `mlwa_results.zip` and `mlwa_research_grade_results.zip` — earlier result snapshots retained for provenance.
- `requirements.txt` — Python dependencies.
- `LICENSE` — MIT License for repository code.

## Analysis pipeline

The final notebook includes:

- leakage-safe preprocessing;
- nested 5×5 stratified cross-validation;
- randomized hyperparameter search with 20 configurations per candidate model;
- comparison of logistic regression, random forest, HistGradientBoosting, XGBoost, and LightGBM;
- final HistGradientBoosting model selection using discrimination plus probability accuracy;
- sigmoid probability calibration;
- frozen sensitivity-oriented global threshold selection;
- temporal validation on NHANES August 2021–August 2023;
- 2,000-replicate bootstrap confidence intervals;
- revised survey-weighted sensitivity analyses;
- decision-curve analysis;
- temporal permutation feature importance;
- subgroup discrimination and calibration;
- age-removal and outcome-component sensitivity analyses;
- race/activity and class-imbalance sensitivity analyses; and
- exploratory age- and diabetes-stratified threshold analyses.

## Data

This study uses publicly available, deidentified NHANES data from the CDC National Center for Health Statistics.

- **Development cohort:** NHANES 2017–2018, n = 5,533 adults
- **Temporal validation cohort:** NHANES August 2021–August 2023, n = 6,337 adults
- **Development unweighted phenotype prevalence:** 18.43%
- **Development survey-weighted prevalence:** 13.61%
- **Temporal unweighted phenotype prevalence:** 17.11%
- **Temporal survey-weighted prevalence (WTPH2YR):** 14.06%

The cleaned development dataset used for analysis is included as `nhanes_ckd_risk_data.csv`. The temporal cohort is reconstructed in the notebook from public NHANES source files.

## Reproducing the analysis

1. Clone or download this repository.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Open `CKD_MLWA_Research_v5_WTPH2YR_REVISED.ipynb` in Jupyter or Google Colab.
4. Run the notebook cells in order.

The notebook can download/read the cleaned development data and reconstruct the temporal cohort from NHANES source files. A fixed random seed of 42 is used wherever supported. Exact bit-for-bit reproducibility may vary across package versions, operating systems, and hardware, particularly for gradient-boosting libraries.

## Principal results

The final sigmoid-calibrated HistGradientBoosting model achieved:

| Analysis | ROC-AUC | PR-AUC | Brier | Sensitivity | Specificity |
|---|---:|---:|---:|---:|---:|
| Development OOF | 0.7890 | 0.4650 | 0.1223 | 0.8000 | 0.6497 |
| Temporal validation | 0.7892 | 0.4648 | 0.1160 | 0.8118 | 0.6202 |
| Weighted development | 0.7852 | 0.4017 | 0.0978 | 0.7342 | 0.7252 |
| Weighted temporal (`WTPH2YR`) | 0.7938 | 0.4378 | 0.0989 | 0.7549 | 0.7144 |

The globally frozen screening threshold is **0.131182**.

Temporal calibration remained stable (intercept 0.0238, slope 1.0756, ECE 0.0128). The manuscript additionally reports decision-curve analysis, permutation importance, subgroup heterogeneity, and exploratory age- and diabetes-stratified operating thresholds.

## Important interpretation

The target is a **single-visit CKD-compatible screening phenotype**, not confirmed chronic kidney disease. NHANES does not establish persistence for at least three months. The model is intended as a pre-laboratory prioritization tool for confirmatory eGFR and albuminuria testing, not as a diagnostic replacement.

Both cohorts originate from NHANES. The validation is therefore described as temporal validation/temporal transport evaluation rather than fully independent external validation.

## License

Code in this repository is released under the MIT License. NHANES data are public-use data provided by the U.S. National Center for Health Statistics; users remain responsible for complying with applicable NHANES documentation and terms.

## Citation

A finalized citation and DOI will be added after publication.

## Corresponding author

Sanad Biswas  
Department of Mathematics and Statistics  
Sam Houston State University  
Huntsville, Texas, USA  
Email: sxb218@shsu.edu
