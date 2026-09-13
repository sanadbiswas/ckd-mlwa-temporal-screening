outer_cv = StratifiedKFold(
    n_splits=OUTER_FOLDS,
    shuffle=True,
    random_state=RANDOM_STATE
)

nested_results = {}
best_params_by_model = defaultdict(list)

for model_name, spec in MODEL_SPECS.items():
    print(f"\n=== {model_name} ===")
    oof_prob = np.full(len(X_dev), np.nan, dtype=float)
    fold_rows = []

    for fold, (train_idx, test_idx) in enumerate(
        outer_cv.split(X_dev, y_dev), start=1
    ):
        X_tr = X_dev.iloc[train_idx]
        y_tr = y_dev.iloc[train_idx]
        X_te = X_dev.iloc[test_idx]
        y_te = y_dev.iloc[test_idx]

        inner_cv = StratifiedKFold(
            n_splits=INNER_FOLDS,
            shuffle=True,
            random_state=RANDOM_STATE + fold
        )

        search = RandomizedSearchCV(
            estimator=clone(spec["pipeline"]),
            param_distributions=spec["params"],
            n_iter=N_ITER,
            scoring="roc_auc",
            cv=inner_cv,
            random_state=RANDOM_STATE + fold,
            n_jobs=-1,
            refit=True,
            verbose=0
        )

        search.fit(X_tr, y_tr)
        p = search.best_estimator_.predict_proba(X_te)[:, 1]
        oof_prob[test_idx] = p

        fold_metric = metrics_from_prob(y_te, p)
        fold_metric["fold"] = fold
        fold_rows.append(fold_metric)
        best_params_by_model[model_name].append(search.best_params_)

        print(
            f"fold {fold}: AUC={fold_metric['roc_auc']:.3f}, "
            f"AP={fold_metric['pr_auc']:.3f}, "
            f"Brier={fold_metric['brier']:.3f}"
        )

    if np.isnan(oof_prob).any():
        raise RuntimeError(f"Missing OOF predictions for {model_name}")

    nested_results[model_name] = {
        "oof_prob": oof_prob,
        "fold_metrics": pd.DataFrame(fold_rows)
    }

print("\nNested CV complete.")

# Model selection uses outer-fold test metrics rather than a pooled AUC from
# separately tuned outer-fold models, whose probability scales can differ.
model_comparison_rows = []

for model_name, result in nested_results.items():
    fm = result["fold_metrics"].copy()

    model_comparison_rows.append({
        "model": model_name,
        "mean_roc_auc": fm["roc_auc"].mean(),
        "sd_roc_auc": fm["roc_auc"].std(ddof=1),
        "se_roc_auc": fm["roc_auc"].std(ddof=1) / np.sqrt(len(fm)),
        "mean_pr_auc": fm["pr_auc"].mean(),
        "sd_pr_auc": fm["pr_auc"].std(ddof=1),
        "mean_brier": fm["brier"].mean(),
        "sd_brier": fm["brier"].std(ddof=1),
        "mean_log_loss": fm["log_loss"].mean(),
        "sd_log_loss": fm["log_loss"].std(ddof=1)
    })

model_comparison = (
    pd.DataFrame(model_comparison_rows)
    .sort_values("mean_roc_auc", ascending=False)
    .reset_index(drop=True)
)

display(model_comparison.round(4))
model_comparison.to_csv(
    RESULTS_DIR / "nested_cv_model_comparison_CORRECTED.csv",
    index=False
)

# One-standard-error rule followed by probability-accuracy tiebreaking.
best_mean_auc = model_comparison.loc[0, "mean_roc_auc"]
best_auc_se = model_comparison.loc[0, "se_roc_auc"]
auc_cutoff = best_mean_auc - best_auc_se

competitive_models = model_comparison[
    model_comparison["mean_roc_auc"] >= auc_cutoff
].copy()

competitive_models = competitive_models.sort_values(
    ["mean_brier", "mean_roc_auc"],
    ascending=[True, False]
)

SELECTED_MODEL_NAME = competitive_models.iloc[0]["model"]

print("Best mean outer-fold AUC:", round(best_mean_auc, 4))
print("One-SE AUC cutoff:", round(auc_cutoff, 4))
print("\nCompetitive models:")
display(
    competitive_models[
        ["model", "mean_roc_auc", "se_roc_auc", "mean_pr_auc",
         "mean_brier", "mean_log_loss"]
    ].round(4)
)
print("Selected model family:", SELECTED_MODEL_NAME)
