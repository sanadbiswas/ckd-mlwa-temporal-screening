import sklearn
import lightgbm
import xgboost
import imblearn
import statsmodels

environment = {
    "python": platform.python_version(),
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "sklearn": sklearn.__version__,
    "lightgbm": lightgbm.__version__,
    "xgboost": xgboost.__version__,
    "imbalanced_learn": imblearn.__version__,
    "statsmodels": statsmodels.__version__,
    "random_state": RANDOM_STATE,
    "outer_folds": OUTER_FOLDS,
    "inner_folds": INNER_FOLDS,
    "n_iter": N_ITER,
    "selected_model": SELECTED_MODEL_NAME,
    "best_calibration": BEST_CALIBRATION,
    "screening_threshold_target_sensitivity": 0.80,
    "screening_threshold": SCREENING_THRESHOLD,
    "primary_features": PRIMARY_FEATURES,
    "final_params": final_search.best_params_
}

with open(RESULTS_DIR / "reproducibility_environment.json", "w") as f:
    json.dump(environment, f, indent=2, default=str)

master = pd.DataFrame([
    {
        "cohort": "2017-2018 internal OOF",
        **metrics_from_prob(y_dev, selected_internal_prob, SCREENING_THRESHOLD)
    },
    {
        "cohort": "2021-2023 temporal validation",
        **metrics_from_prob(y_ext, p_ext, SCREENING_THRESHOLD)
    }
])

display(master.round(4))
master.to_csv(RESULTS_DIR / "MASTER_PERFORMANCE_TABLE.csv", index=False)

print("\nAll outputs saved to:", RESULTS_DIR)
print("\n=== FINAL SURVEY-WEIGHTED RESULTS ===")
display(survey_weighted_table.round(4))
print("Development weighted prevalence (WTMEC2YR):", round(np.average(y_dev, weights=w_dev), 4))
print("Temporal weighted prevalence (WTPH2YR):", round(np.average(y_ext, weights=w_ext), 4))
print("Temporal weighted calibration:")
print({k: round(v, 4) for k, v in temporal_calibration_weighted.items()})
