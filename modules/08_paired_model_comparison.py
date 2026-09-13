from scipy.stats import ttest_rel, wilcoxon

selected_fold_metrics = nested_results[SELECTED_MODEL_NAME]["fold_metrics"]
paired_rows = []

for other_name, result in nested_results.items():
    if other_name == SELECTED_MODEL_NAME:
        continue

    other_fm = result["fold_metrics"]
    row = {"comparison": f"{SELECTED_MODEL_NAME} vs {other_name}"}

    for metric in ["roc_auc", "pr_auc", "brier", "log_loss"]:
        a = selected_fold_metrics[metric].to_numpy(dtype=float)
        b = other_fm[metric].to_numpy(dtype=float)
        diff = a - b
        _, t_p = ttest_rel(a, b)

        try:
            _, w_p = wilcoxon(diff)
        except ValueError:
            w_p = np.nan

        row[f"{metric}_mean_diff"] = diff.mean()
        row[f"{metric}_sd_diff"] = diff.std(ddof=1)
        row[f"{metric}_paired_t_p"] = t_p
        row[f"{metric}_wilcoxon_p"] = w_p

    paired_rows.append(row)

paired_fold_results = pd.DataFrame(paired_rows)
display(paired_fold_results.round(4))
paired_fold_results.to_csv(
    RESULTS_DIR / "paired_outer_fold_model_comparison.csv",
    index=False
)
