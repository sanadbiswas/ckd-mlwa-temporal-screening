p_ext = FINAL_MODEL.predict_proba(X_ext)[:, 1]

temporal_unweighted = metrics_from_prob(
    y_ext, p_ext, threshold=SCREENING_THRESHOLD
)
temporal_weighted = metrics_from_prob(
    y_ext, p_ext, threshold=SCREENING_THRESHOLD, sample_weight=w_ext
)

print("Unweighted temporal performance")
display(pd.Series(temporal_unweighted).to_frame("estimate").round(4))
print("Survey-weighted temporal performance")
display(pd.Series(temporal_weighted).to_frame("estimate").round(4))


def bootstrap_ci(y, prob, threshold=None, n_boot=2000, seed=42):
    y = np.asarray(y)
    prob = np.asarray(prob)
    rng = np.random.default_rng(seed)
    n = len(y)
    point = metrics_from_prob(y, prob, threshold=threshold)
    boot_rows = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y_boot = y[idx]
        p_boot = prob[idx]
        if np.unique(y_boot).size < 2:
            continue
        boot_rows.append(metrics_from_prob(y_boot, p_boot, threshold=threshold))

    boot = pd.DataFrame(boot_rows)
    ci = pd.DataFrame([
        {
            "metric": metric,
            "estimate": estimate,
            "ci_low_95": boot[metric].quantile(0.025),
            "ci_high_95": boot[metric].quantile(0.975)
        }
        for metric, estimate in point.items()
    ])
    return ci, boot


temporal_ci, temporal_boot = bootstrap_ci(
    y_ext, p_ext,
    threshold=SCREENING_THRESHOLD,
    n_boot=N_BOOT,
    seed=RANDOM_STATE + 500
)

display(temporal_ci.round(4))
temporal_ci.to_csv(RESULTS_DIR / "temporal_validation_bootstrap_CI.csv", index=False)

pd.DataFrame([
    {"analysis": "temporal_unweighted", **temporal_unweighted},
    {"analysis": "temporal_survey_weighted", **temporal_weighted}
]).to_csv(RESULTS_DIR / "temporal_validation_metrics.csv", index=False)
