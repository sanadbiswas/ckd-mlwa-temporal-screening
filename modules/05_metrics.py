def safe_specificity(y_true, y_pred, sample_weight=None):
    cm = confusion_matrix(
        y_true, y_pred, labels=[0, 1], sample_weight=sample_weight
    )
    tn, fp, fn, tp = cm.ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else np.nan


def metrics_from_prob(y_true, prob, threshold=None, sample_weight=None):
    out = {
        "roc_auc": roc_auc_score(y_true, prob, sample_weight=sample_weight),
        "pr_auc": average_precision_score(y_true, prob, sample_weight=sample_weight),
        "brier": brier_score_loss(y_true, prob, sample_weight=sample_weight),
        "log_loss": log_loss(y_true, prob, sample_weight=sample_weight, labels=[0, 1])
    }

    if threshold is not None:
        pred = (np.asarray(prob) >= threshold).astype(int)
        out.update({
            "accuracy": accuracy_score(y_true, pred, sample_weight=sample_weight),
            "precision": precision_score(y_true, pred, sample_weight=sample_weight, zero_division=0),
            "sensitivity": recall_score(y_true, pred, sample_weight=sample_weight, zero_division=0),
            "specificity": safe_specificity(y_true, pred, sample_weight=sample_weight),
            "f1": f1_score(y_true, pred, sample_weight=sample_weight, zero_division=0)
        })
    return out


def threshold_for_target_sensitivity(y_true, prob, target=0.80):
    fpr, tpr, thresholds = roc_curve(y_true, prob)
    eligible = np.where(tpr >= target)[0]
    if len(eligible) == 0:
        return 0.0
    i = eligible[np.argmin(fpr[eligible])]
    return float(thresholds[i])
