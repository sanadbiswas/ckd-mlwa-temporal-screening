AGE_FREE_FEATURES = [c for c in PRIMARY_FEATURES if c != "RIDAGEYR"]
X_dev_agefree = df[AGE_FREE_FEATURES].copy()
X_ext_agefree = ext[AGE_FREE_FEATURES].copy()

agefree_model_template = clone(selected_spec["pipeline"].named_steps["model"])
agefree_pipeline = make_pipeline(agefree_model_template, X_dev_agefree)

agefree_search = RandomizedSearchCV(
    agefree_pipeline,
    selected_spec["params"],
    n_iter=N_ITER,
    scoring="roc_auc",
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE + 700),
    random_state=RANDOM_STATE + 700,
    n_jobs=-1,
    refit=True
)

agefree_oof = cross_val_predict(
    agefree_search,
    X_dev_agefree,
    y_dev,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE + 701),
    method="predict_proba",
    n_jobs=-1
)[:, 1]

agefree_internal = metrics_from_prob(y_dev, agefree_oof)
agefree_search.fit(X_dev_agefree, y_dev)
agefree_final = agefree_search.best_estimator_
p_ext_agefree = agefree_final.predict_proba(X_ext_agefree)[:, 1]
agefree_temporal = metrics_from_prob(y_ext, p_ext_agefree)

age_sensitivity = pd.DataFrame([
    {
        "analysis": "Primary model",
        "internal_auc": roc_auc_score(y_dev, selected_internal_prob),
        "temporal_auc": roc_auc_score(y_ext, p_ext)
    },
    {
        "analysis": "Age removed",
        "internal_auc": agefree_internal["roc_auc"],
        "temporal_auc": agefree_temporal["roc_auc"]
    }
])

display(age_sensitivity.round(4))
age_sensitivity.to_csv(RESULTS_DIR / "age_coupling_sensitivity.csv", index=False)
