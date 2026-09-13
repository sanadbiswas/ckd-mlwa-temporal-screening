from scipy.stats import t

fold_summary_rows = []

for model_name, result in nested_results.items():
    fm = result["fold_metrics"].copy()
    n_folds = len(fm)
    tcrit = t.ppf(0.975, df=n_folds - 1)

    for metric in ["roc_auc", "pr_auc", "brier", "log_loss"]:
        vals = fm[metric].to_numpy(dtype=float)
        mean = vals.mean()
        sd = vals.std(ddof=1)
        se = sd / np.sqrt(n_folds)

        fold_summary_rows.append({
            "model": model_name,
            "metric": metric,
            "mean": mean,
            "sd": sd,
            "ci_low_95": mean - tcrit * se,
            "ci_high_95": mean + tcrit * se
        })

nested_cv_fold_CI = pd.DataFrame(fold_summary_rows)

display(
    nested_cv_fold_CI[
        nested_cv_fold_CI["metric"].isin(["roc_auc", "pr_auc", "brier"])
    ].round(4)
)

nested_cv_fold_CI.to_csv(
    RESULTS_DIR / "nested_cv_outer_fold_summary_CI.csv",
    index=False
)
