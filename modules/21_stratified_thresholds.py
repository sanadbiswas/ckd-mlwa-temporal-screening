dev_age_group = pd.cut(
    df["RIDAGEYR"].to_numpy(),
    bins=[17, 39, 59, np.inf],
    labels=["18-39", "40-59", "60+"]
)

age_thresholds = {}
for level in ["18-39", "40-59", "60+"]:
    mask = dev_age_group == level
    y_g = y_dev.to_numpy()[mask]
    p_g = np.asarray(selected_internal_prob)[mask]
    t_g = threshold_for_target_sensitivity(y_g, p_g, target=0.80)
    age_thresholds[level] = t_g
    dev_op = metrics_from_prob(y_g, p_g, threshold=t_g)
    print(
        f"[{level}] development n={mask.sum()}, threshold={t_g:.6f}, "
        f"sensitivity={dev_op['sensitivity']:.4f}, specificity={dev_op['specificity']:.4f}"
    )

rows = []
for level in ["18-39", "40-59", "60+"]:
    mask = ext_eval["age_group"] == level
    y_g = ext_eval.loc[mask, "ckd"].to_numpy(dtype=int)
    p_g = ext_eval.loc[mask, "prob"].to_numpy(dtype=float)
    t_g = age_thresholds[level]
    global_metrics = metrics_from_prob(y_g, p_g, threshold=SCREENING_THRESHOLD)
    strat_metrics = metrics_from_prob(y_g, p_g, threshold=t_g)
    rows.append({
        "age_group": level,
        "n": int(mask.sum()),
        "cases": int(y_g.sum()),
        "global_threshold": SCREENING_THRESHOLD,
        "global_sensitivity": global_metrics["sensitivity"],
        "global_specificity": global_metrics["specificity"],
        "stratified_threshold": t_g,
        "stratified_sensitivity": strat_metrics["sensitivity"],
        "stratified_specificity": strat_metrics["specificity"]
    })

age_strat_comparison = pd.DataFrame(rows)
display(age_strat_comparison.round(4))
age_strat_comparison.to_csv(
    RESULTS_DIR / "temporal_age_stratified_threshold_comparison.csv", index=False
)

pooled_pred = np.zeros(len(ext_eval), dtype=int)
for level in ["18-39", "40-59", "60+"]:
    mask = (ext_eval["age_group"] == level).to_numpy()
    pooled_pred[mask] = (
        ext_eval.loc[mask, "prob"].to_numpy() >= age_thresholds[level]
    ).astype(int)

y_all = ext_eval["ckd"].to_numpy(dtype=int)
print(
    "Age-stratified pooled temporal performance:",
    f"sensitivity={recall_score(y_all, pooled_pred, zero_division=0):.4f},",
    f"specificity={safe_specificity(y_all, pooled_pred):.4f}"
)


diabetes_thresholds = {}
dev_diabetes = df["has_diabetes_dx"].to_numpy()
for level in ["no", "yes"]:
    mask = dev_diabetes == level
    y_g = y_dev.to_numpy()[mask]
    p_g = np.asarray(selected_internal_prob)[mask]
    t_g = threshold_for_target_sensitivity(y_g, p_g, target=0.80)
    diabetes_thresholds[level] = t_g
    dev_op = metrics_from_prob(y_g, p_g, threshold=t_g)
    print(
        f"[diabetes={level}] development n={mask.sum()}, threshold={t_g:.6f}, "
        f"sensitivity={dev_op['sensitivity']:.4f}, specificity={dev_op['specificity']:.4f}"
    )

rows = []
for level in ["no", "yes"]:
    mask = ext_eval["has_diabetes_dx"] == level
    y_g = ext_eval.loc[mask, "ckd"].to_numpy(dtype=int)
    p_g = ext_eval.loc[mask, "prob"].to_numpy(dtype=float)
    t_g = diabetes_thresholds[level]
    global_metrics = metrics_from_prob(y_g, p_g, threshold=SCREENING_THRESHOLD)
    strat_metrics = metrics_from_prob(y_g, p_g, threshold=t_g)
    rows.append({
        "diabetes_dx": level,
        "n": int(mask.sum()),
        "cases": int(y_g.sum()),
        "global_threshold": SCREENING_THRESHOLD,
        "global_sensitivity": global_metrics["sensitivity"],
        "global_specificity": global_metrics["specificity"],
        "stratified_threshold": t_g,
        "stratified_sensitivity": strat_metrics["sensitivity"],
        "stratified_specificity": strat_metrics["specificity"]
    })

diabetes_strat_comparison = pd.DataFrame(rows)
display(diabetes_strat_comparison.round(4))
diabetes_strat_comparison.to_csv(
    RESULTS_DIR / "temporal_diabetes_stratified_threshold_comparison.csv", index=False
)
