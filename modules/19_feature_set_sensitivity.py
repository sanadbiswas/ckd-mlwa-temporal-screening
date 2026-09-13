SENSITIVITY_FEATURE_SETS = {
    "Primary": PRIMARY_FEATURES,
    "Primary_plus_race": PRIMARY_FEATURES + ["RIDRETH1"],
}

pa_cols = ["PAQ605", "PAQ620", "PAQ635", "PAQ650", "PAQ665"]
if all(c in df.columns for c in pa_cols):
    df["activity_any"] = ((df[pa_cols] == "yes").any(axis=1)).astype(float)
    df.loc[df[pa_cols].isna().all(axis=1), "activity_any"] = np.nan
    SENSITIVITY_FEATURE_SETS["Primary_plus_activity"] = PRIMARY_FEATURES + ["activity_any"]

feature_sensitivity_rows = []
for label, cols in SENSITIVITY_FEATURE_SETS.items():
    XX = df[cols].copy()
    template = clone(selected_spec["pipeline"].named_steps["model"])
    pipe = make_pipeline(template, XX)

    search = RandomizedSearchCV(
        pipe,
        selected_spec["params"],
        n_iter=max(10, N_ITER // 2),
        scoring="roc_auc",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE + 800),
        random_state=RANDOM_STATE + 800,
        n_jobs=-1
    )

    oof = cross_val_predict(
        search,
        XX,
        y_dev,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE + 801),
        method="predict_proba",
        n_jobs=-1
    )[:, 1]

    feature_sensitivity_rows.append({
        "feature_set": label,
        **metrics_from_prob(y_dev, oof)
    })

feature_sensitivity = pd.DataFrame(feature_sensitivity_rows)
display(feature_sensitivity.round(4))
feature_sensitivity.to_csv(RESULTS_DIR / "race_activity_feature_sensitivity.csv", index=False)
