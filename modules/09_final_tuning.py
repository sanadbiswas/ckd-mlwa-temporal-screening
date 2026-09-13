selected_spec = MODEL_SPECS[SELECTED_MODEL_NAME]

final_inner_cv = StratifiedKFold(
    n_splits=INNER_FOLDS,
    shuffle=True,
    random_state=RANDOM_STATE + 100
)

final_search = RandomizedSearchCV(
    estimator=clone(selected_spec["pipeline"]),
    param_distributions=selected_spec["params"],
    n_iter=max(N_ITER, 30),
    scoring="roc_auc",
    cv=final_inner_cv,
    random_state=RANDOM_STATE + 100,
    n_jobs=-1,
    refit=True
)

final_search.fit(X_dev, y_dev)
final_raw_model = final_search.best_estimator_

print("Selected model:", SELECTED_MODEL_NAME)
print("Final hyperparameters:")
print(final_search.best_params_)
