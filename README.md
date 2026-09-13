# Temporal Validation of a Non-Laboratory Machine Learning Framework for CKD Screening

This repository contains the analysis code and reproducibility materials for:

**Temporal Validation of a Non-Laboratory Machine Learning Framework for Chronic Kidney Disease Screening in U.S. Adults**

The study develops a non-laboratory CKD screening model using NHANES 2017–2018 and evaluates the frozen pipeline on NHANES August 2021–August 2023.

## Study design

- Development cohort: NHANES 2017–2018
- Temporal validation cohort: NHANES August 2021–August 2023
- Primary model: sigmoid-calibrated histogram gradient boosting
- Predictors: 12 routinely available non-laboratory variables
- Primary outcome: single-visit CKD-compatible screening phenotype defined as eGFR < 60 mL/min/1.73 m² or urinary ACR ≥ 30 mg/g
- Final analytic sample sizes: 5,063 development participants and 5,552 temporal-validation participants

Participants with missing serum creatinine or urinary ACR are excluded before outcome construction in both cohorts.

## Repository structure

```text
.
├── ckd-runner.ipynb
├── pipeline_runner.py
├── requirements.txt
├── modules/
│   ├── 00_setup.py
│   ├── 01_data_preparation.py
│   ├── ...
│   └── 25_reproducibility_and_master_table.py
└── ckd_results/
```

The cleaned 2017–2018 development dataset is rebuilt directly from the public CDC/NCHS NHANES XPT files by `modules/01_data_preparation.py`. The temporal cohort is reconstructed independently from the public NHANES 2021–2023 files during the analysis.

## Reproducing the analysis

Install the required packages:

```bash
pip install -r requirements.txt
```

Run `ckd-runner.ipynb` from top to bottom. The runner rebuilds the development dataset, executes all modular analysis stages in sequence, and writes generated tables and figures to `ckd_results/`.

## Reproducibility safeguards

Outcome-defining laboratory values are handled explicitly. Participants are excluded before phenotype construction when serum creatinine (`LBXSCR`) or urinary ACR (`URDACT`) is missing. Laboratory variables used to construct the outcome are removed before modeling.

The final screening threshold is derived from development out-of-fold predictions and applied unchanged to the temporal cohort.

## Results

The `ckd_results/` directory contains the manuscript-aligned performance tables, calibration summaries, subgroup analyses, sensitivity analyses, survey-weighted results, decision-curve outputs, permutation importance, reproducibility metadata, and figures.

The main manuscript figures are stored in the native formats produced by the analysis code, including PNG and EPS files.

- `internal_calibration_oof`
- `calibration_transport`
- `decision_curve_temporal`
- `temporal_permutation_importance`

## Data source

NHANES data are publicly available from the U.S. Centers for Disease Control and Prevention, National Center for Health Statistics:

https://www.cdc.gov/nchs/nhanes/

## Corresponding author

**Sanad Biswas, Ph.D.**  
Assistant Professor of Data Science  
Department of Mathematics and Statistics  
Sam Houston State University  
Huntsville, Texas, USA  
Email: sxb218@shsu.edu  
Alternate email: biswas.sanad@gmail.com
