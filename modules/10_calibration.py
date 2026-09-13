fixed_selected = clone(final_raw_model)

calibrated_sigmoid = CalibratedClassifierCV(
    estimator=clone(fixed_selected),
    method="sigmoid",
    cv=5,
    n_jobs=-1
)

calibrated_isotonic = CalibratedClassifierCV(
    estimator=clone(fixed_selected),
    method="isotonic",
    cv=5,
    n_jobs=-1
)

cv_for_calibration = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE + 200
)

raw_oof_fixed = cross_val_predict(
    clone(fixed_selected), X_dev, y_dev,
    cv=cv_for_calibration, method="predict_proba", n_jobs=-1
)[:, 1]

sigmoid_oof = cross_val_predict(
    clone(calibrated_sigmoid), X_dev, y_dev,
    cv=cv_for_calibration, method="predict_proba", n_jobs=-1
)[:, 1]

isotonic_oof = cross_val_predict(
    clone(calibrated_isotonic), X_dev, y_dev,
    cv=cv_for_calibration, method="predict_proba", n_jobs=-1
)[:, 1]


def calibration_intercept_slope(y, p):
    eps = 1e-6
    p = np.clip(np.asarray(p), eps, 1 - eps)
    logit_p = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=3000)
    lr.fit(logit_p, np.asarray(y))
    return float(lr.intercept_[0]), float(lr.coef_[0][0])


def weighted_ece(y, p, n_bins=10, sample_weight=None):
    y = np.asarray(y)
    p = np.asarray(p)
    w = np.ones_like(p, dtype=float) if sample_weight is None else np.asarray(sample_weight, dtype=float)
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


calibration_candidates = {
    "Raw": raw_oof_fixed,
    "Sigmoid": sigmoid_oof,
    "Isotonic": isotonic_oof
}

cal_rows = []
for name, p in calibration_candidates.items():
    intercept, slope = calibration_intercept_slope(y_dev, p)
    m = metrics_from_prob(y_dev, p)
    cal_rows.append({
        "calibration": name,
        **m,
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "ece": weighted_ece(y_dev, p, 10)
    })

calibration_internal = pd.DataFrame(cal_rows).sort_values("brier")
display(calibration_internal.round(4))
calibration_internal.to_csv(
    RESULTS_DIR / "internal_calibration_comparison.csv",
    index=False
)

BEST_CALIBRATION = calibration_internal.iloc[0]["calibration"]
print("Selected calibration:", BEST_CALIBRATION)

fig, ax = plt.subplots(figsize=(8, 6.5))
for name, p in calibration_candidates.items():
    bins = pd.qcut(pd.Series(p), q=10, duplicates="drop")
    temp = pd.DataFrame({"p": p, "y": y_dev.to_numpy(), "bin": bins})
    grouped = temp.groupby("bin", observed=True).agg(
        mean_pred=("p", "mean"), observed=("y", "mean")
    )
    ax.plot(
        grouped["mean_pred"], grouped["observed"],
        marker="o", linewidth=2.0, markersize=6.5, label=name
    )

ax.plot([0, 1], [0, 1], linestyle="--", linewidth=2.0, color="red", label="Ideal")
ax.set_xlabel("Predicted probability", fontsize=13)
ax.set_ylabel("Observed outcome frequency", fontsize=13)
ax.set_title("Internal Calibration — Out-of-Fold Predictions", fontsize=15, pad=12)
for spine in ax.spines.values():
    spine.set_linewidth(1.8)
    spine.set_color("black")
ax.tick_params(axis="both", which="major", labelsize=11, width=1.4, length=5)
ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.45)
ax.set_axisbelow(True)
legend = ax.legend(fontsize=10.5, frameon=True, loc="best")
legend.get_frame().set_linewidth(1.0)
legend.get_frame().set_edgecolor("black")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "internal_calibration_oof.png", dpi=600, bbox_inches="tight")
plt.savefig(RESULTS_DIR / "internal_calibration_oof.eps", format="eps", bbox_inches="tight")
plt.show()
