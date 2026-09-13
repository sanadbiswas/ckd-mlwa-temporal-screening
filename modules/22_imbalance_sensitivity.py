def selected_base_estimator():
    return clone(selected_spec["pipeline"].named_steps["model"])


imbalance_pipelines = {
    "none": Pipeline([
        ("prep", make_preprocessor(X_dev)),
        ("model", selected_base_estimator())
    ]),
    "smote": ImbPipeline([
        ("prep", make_preprocessor(X_dev)),
        ("resample", SMOTE(random_state=RANDOM_STATE)),
        ("model", selected_base_estimator())
    ]),
    "smoteenn": ImbPipeline([
        ("prep", make_preprocessor(X_dev)),
        ("resample", SMOTEENN(random_state=RANDOM_STATE)),
        ("model", selected_base_estimator())
    ])
}

cw_model = selected_base_estimator()
if "class_weight" in cw_model.get_params():
    cw_model.set_params(class_weight="balanced")
    imbalance_pipelines["class_weight"] = Pipeline([
        ("prep", make_preprocessor(X_dev)),
        ("model", cw_model)
    ])

imbalance_rows = []
cv_imb = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE + 1000)

for name, pipe in imbalance_pipelines.items():
    p = cross_val_predict(
        pipe, X_dev, y_dev, cv=cv_imb, method="predict_proba", n_jobs=-1
    )[:, 1]
    imbalance_rows.append({"strategy": name, **metrics_from_prob(y_dev, p)})

imbalance_results = pd.DataFrame(imbalance_rows)
display(imbalance_results.round(4))
imbalance_results.to_csv(RESULTS_DIR / "imbalance_sensitivity.csv", index=False)
