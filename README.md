# Temporal Validation of a Non-Laboratory ML Framework for CKD Screening

Reproducibility materials for the manuscript:

> Dutta S, Lamsal C, Deb P, Sikder S, Biswas S. *Temporal Validation of a Non-Laboratory Machine Learning Framework for Chronic Kidney Disease Screening in U.S. Adults.* Manuscript prepared for submission.

## Overview

This repository contains the analysis code, cleaned development data, and result archives used to develop and temporally validate a non-laboratory machine-learning framework for prioritizing adults for confirmatory chronic kidney disease (CKD) testing using National Health and Nutrition Examination Survey (NHANES) data.

The primary analysis uses NHANES 2017–2018 for model development and NHANES August 2021–August 2023 for temporal validation. The outcome is a single-visit CKD-compatible screening phenotype defined by eGFR < 60 mL/min/1.73 m² or urine albumin-to-creatinine ratio (ACR) ≥ 30 mg/g. Kidney-specific laboratory variables used to define the outcome are excluded from the predictor matrix.

## Repository contents

- `CKD_MLWA_Research.ipynb` — complete research notebook used for the manuscript analyses.
- `nhanes_ckd_risk_data.csv` — cleaned development dataset used by the notebook.
- `mlwa_results.zip` — primary analysis result outputs.
- `mlwa_research_grade_results.zip` — research-grade result outputs and supporting artifacts.
- `requirements.txt` — Python dependencies.
- `LICENSE` — MIT License for repository code.

The notebook includes:

- leakage-safe preprocessing;
- nested 5×5 stratified cross-validation;
- comparison of logistic regression, random forest, HistGradientBoosting, XGBoost, and LightGBM;
- final HistGradientBoosting model selection;
- sigmoid probability calibration;
- frozen sensitivity-oriented threshold selection;
- temporal validation on NHANES 2021–2023;
- bootstrap confidence intervals;
- survey-weighted sensitivity analyses;
- decision-curve analysis;
- permutation feature importance;
- subgroup discrimination and calibration;
- age- and diabetes-stratified threshold sensitivity analyses; and
- additional sensitivity analyses reported in the manuscript.

## Data

This study uses publicly available, deidentified NHANES data from the CDC National Center for Health Statistics.

- **Development cohort:** NHANES 2017–2018 (n = 5,533 adults after cohort construction)
- **Temporal validation cohort:** NHANES August 2021–August 2023 (n = 6,337 adults)

The cleaned development dataset used for analysis is included as `nhanes_ckd_risk_data.csv`. The temporal cohort is reconstructed in the notebook from publicly available NHANES source files.

## Reproducing the analysis

1. Clone or download this repository.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Open `CKD_MLWA_Research.ipynb` in Jupyter or Google Colab.
4. Set the notebook data path to `nhanes_ckd_risk_data.csv` if needed.
5. Set `RESULTS_DIR` to a writable output directory.
6. Run the notebook cells in order.

A fixed random seed of 42 is used wherever supported. Exact bit-for-bit reproducibility may vary across package versions and hardware, particularly for gradient-boosting libraries.

## Primary manuscript results

The final sigmoid-calibrated HistGradientBoosting model achieved an internal ROC-AUC of 0.7890 and temporal ROC-AUC of 0.7892. The frozen global screening threshold was 0.131182, yielding temporal sensitivity of 0.8118 and specificity of 0.6202.

The study additionally evaluates survey-weighted performance, calibration transport, decision-curve net benefit, temporal permutation importance, subgroup performance, age removal, outcome components, predictor-set sensitivities, class-imbalance strategies, and frozen age- and diabetes-stratified thresholds.

## Reproducibility notes

The ZIP archives are retained as complete snapshots of analysis outputs produced during manuscript development. Individual result files may also be extracted locally for inspection or downstream use.

## License

Code in this repository is released under the MIT License. NHANES data are public-use data provided by the U.S. National Center for Health Statistics; users remain responsible for complying with the applicable NHANES terms and documentation.

## Citation

A finalized citation and DOI will be added after publication.

## Corresponding author

Sanad Biswas  
Department of Mathematics and Statistics  
Sam Houston State University  
Huntsville, Texas, USA  
Email: sxb218@shsu.edu
