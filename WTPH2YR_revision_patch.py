"""Final v5 survey-weight revision for the CKD MLWA analysis.

This patch is intended for the existing research notebook/runtime after the frozen
model, temporal cohort, and temporal probabilities have been created.

Required existing objects:
    ext                  temporal analytic DataFrame with SEQN
    y_ext                temporal outcome vector
    p_ext                frozen calibrated temporal probabilities
    SCREENING_THRESHOLD  globally frozen screening threshold
    RESULTS_DIR          output Path

The 2021–2023 weighted analysis uses WTPH2YR. The development weighted analysis
continues to use WTMEC2YR. Survey weights are evaluation-only.
"""

import io
import numpy as np
import pandas as pd
import requests

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

BIOPRO_L_URL = (
    "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/BIOPRO_L.XPT"
)


def read_xpt_url(url, timeout=120):
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return pd.read_sas(io.BytesIO(r.content), format="xport")


def safe_specificity(y_true, y_pred, sample_weight=None):
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
        sample_weight=sample_weight,
    )
    tn, fp, fn, tp = cm.ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else np.nan


def metrics_from_prob(y_true, prob, threshold, sample_weight=None):
    y_true = np.asarray(y_true)
    prob = np.asarray(prob)
    pred = (prob >= threshold).astype(int)

    return {
        "roc_auc": roc_auc_score(y_true, prob, sample_weight=sample_weight),
        "pr_auc": average_precision_score(
            y_true, prob, sample_weight=sample_weight
        ),
        "brier": brier_score_loss(
            y_true, prob, sample_weight=sample_weight
        ),
        "log_loss": log_loss(
            y_true,
            np.clip(prob, 1e-15, 1 - 1e-15),
            sample_weight=sample_weight,
            labels=[0, 1],
        ),
        "accuracy": accuracy_score(
            y_true, pred, sample_weight=sample_weight
        ),
        "precision": precision_score(
            y_true,
            pred,
            sample_weight=sample_weight,
            zero_division=0,
        ),
        "sensitivity": recall_score(
            y_true,
            pred,
            sample_weight=sample_weight,
            zero_division=0,
        ),
        "specificity": safe_specificity(
            y_true, pred, sample_weight=sample_weight
        ),
        "f1": f1_score(
            y_true,
            pred,
            sample_weight=sample_weight,
            zero_division=0,
        ),
    }


def weighted_ece(y, p, n_bins=10, sample_weight=None):
    y = np.asarray(y)
    p = np.asarray(p)
    w = (
        np.ones_like(p, dtype=float)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=float)
    )

    quantiles = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    if len(quantiles) < 3:
        return np.nan

    ids = np.digitize(p, quantiles[1:-1], right=True)
    total_w = w.sum()
    ece = 0.0

    for b in np.unique(ids):
        m = ids == b
        wb = w[m]
        obs = np.average(y[m], weights=wb)
        pred = np.average(p[m], weights=wb)
        ece += wb.sum() / total_w * abs(obs - pred)

    return float(ece)


# -------------------------------------------------------------------------
# Attach WTPH2YR if the temporal analytic DataFrame does not already have it
# -------------------------------------------------------------------------
if "WTPH2YR" not in ext.columns:
    biopro_l = read_xpt_url(BIOPRO_L_URL)[["SEQN", "WTPH2YR"]].copy()
    biopro_l = biopro_l.drop_duplicates(subset="SEQN")
    ext = ext.merge(biopro_l, on="SEQN", how="left", validate="one_to_one")

w_ext = pd.to_numeric(ext["WTPH2YR"], errors="coerce").to_numpy()

valid = np.isfinite(w_ext) & (w_ext > 0)
if not np.all(valid):
    print(f"Positive WTPH2YR observations: {valid.sum()} / {len(valid)}")

# The final analytic cohort used in the manuscript has positive WTPH2YR values.
# If a future reconstruction produces nonpositive/missing weights, restrict the
# weighted calculation to valid weights without changing the primary unweighted
# analysis.
y_w = np.asarray(y_ext)[valid]
p_w = np.asarray(p_ext)[valid]
w_w = w_ext[valid]

weighted_temporal = metrics_from_prob(
    y_w,
    p_w,
    threshold=SCREENING_THRESHOLD,
    sample_weight=w_w,
)

weighted_prevalence = np.average(y_w, weights=w_w)
weighted_temporal_ece = weighted_ece(
    y_w,
    p_w,
    n_bins=10,
    sample_weight=w_w,
)

print("\n=== FINAL 2021–2023 WTPH2YR RESULTS ===")
print("Weighted prevalence:", round(weighted_prevalence, 4))
for k, v in weighted_temporal.items():
    print(f"{k}: {v:.4f}")
print("ece:", round(weighted_temporal_ece, 4))

weighted_output = pd.DataFrame([
    {
        "cohort": "2021-2023 temporal",
        "survey_weight": "WTPH2YR",
        "weighted_prevalence": weighted_prevalence,
        **weighted_temporal,
        "ece": weighted_temporal_ece,
    }
])

weighted_output.to_csv(
    RESULTS_DIR / "survey_weighted_temporal_WTPH2YR_v5.csv",
    index=False,
)

print(
    "\nExpected manuscript values: prevalence 0.1406, ROC-AUC 0.7938, "
    "PR-AUC 0.4378, Brier 0.0989, log loss 0.3309, sensitivity 0.7549, "
    "specificity 0.7144, F1 0.4313, ECE 0.0128."
)
