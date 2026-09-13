component_rows = []

for outcome_name in ["low_egfr", "albuminuria", "ckd"]:
    yy = ext[outcome_name].astype(int)
    component_rows.append({
        "outcome": outcome_name,
        "prevalence": yy.mean(),
        "roc_auc": roc_auc_score(yy, p_ext),
        "pr_auc": average_precision_score(yy, p_ext),
        "brier": brier_score_loss(yy, p_ext)
    })

outcome_component_results = pd.DataFrame(component_rows)
display(outcome_component_results.round(4))
outcome_component_results.to_csv(
    RESULTS_DIR / "outcome_component_sensitivity.csv", index=False
)
