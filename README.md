# Temporal Validation of a Non-Laboratory ML Framework for CKD Screening

Reproducibility materials for the manuscript:

> Dutta S, Lamsal C, Deb P, Sikder S, Biswas S. *Temporal Validation of a Non-Laboratory Machine Learning Framework for Chronic Kidney Disease Screening in U.S. Adults.* Manuscript prepared for submission.

## Overview

This repository contains the analysis code used to develop and temporally validate a non-laboratory machine-learning framework for prioritizing adults for confirmatory chronic kidney disease (CKD) testing using National Health and Nutrition Examination Survey (NHANES) data.

The primary analysis uses NHANES 2017–2018 for model development and NHANES August 2021–August 2023 for temporal validation. The outcome is a single-visit CKD-compatible screening phenotype defined by eGFR < 60 mL/min/1.73 m² or urine albumin-to-creatinine ratio (ACR) ≥ 30 mg/g. Kidney-specific laboratory variables used to define the outcome are excluded from the predictor matrix.

## Repository contents

- `CKD_MLWA_Research_Grade_Colab_v4_WITH_STRATIFIED_THRESHOLDS.ipynb` — complete analysis notebook.
- `requirements.txt` — Python dependencies needed to run the notebook.

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
- additional prespecified sensitivity analyses.

## Data

This study uses publicly available, deidentified NHANES data from the CDC National Center for Health Statistics.

- **Development cohort:** NHANES 2017–2018 (n = 5,533 adults after cohort construction)
- **Temporal validation cohort:** NHANES August 2021–August 2023 (n = 6,337 adults)

Raw NHANES files are not redistributed in this repository. The notebook reconstructs the temporal cohort from publicly available NHANES files. The cleaned development dataset is not included; users should update the `DATA_PATH` variable in the notebook to point to their local development file.

## Reproducing the analysis

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Open the notebook in Jupyter or Google Colab.
3. Update `DATA_PATH` and `RESULTS_DIR` in the setup cell.
4. Run the notebook cells in order.
5. Output tables and figures will be written to `RESULTS_DIR`.

A fixed random seed of 42 is used wherever supported. Exact bit-for-bit reproducibility may vary across package versions and hardware, particularly for gradient-boosting libraries.

## Primary manuscript results

The final calibrated HistGradientBoosting model achieved an internal ROC-AUC of 0.7890 and temporal ROC-AUC of 0.7892. The frozen global screening threshold was 0.131182, yielding temporal sensitivity of 0.8118 and specificity of 0.6202.

## License

Code in this repository is released under the MIT License. NHANES data are U.S. government public-use data and are governed by the terms of the original data source.

## Citation

A finalized citation will be added after publication.

## Corresponding author

Sanad Biswas  
Department of Mathematics and Statistics  
Sam Houston State University  
Huntsville, Texas, USA
