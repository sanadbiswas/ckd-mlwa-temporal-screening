pos = int(y_dev.sum())
neg = int((1 - y_dev).sum())
imbalance_ratio = neg / pos

MODEL_SPECS = {
    "Logistic Regression": {
        "pipeline": make_pipeline(
            LogisticRegression(max_iter=5000, random_state=RANDOM_STATE), X_dev
        ),
        "params": {
            "model__C": loguniform(1e-3, 1e2),
            "model__class_weight": [None, "balanced"]
        }
    },
    "Random Forest": {
        "pipeline": make_pipeline(
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1), X_dev
        ),
        "params": {
            "model__n_estimators": randint(300, 900),
            "model__max_depth": [None, 4, 6, 8, 12],
            "model__min_samples_leaf": randint(1, 12),
            "model__max_features": ["sqrt", "log2", 0.5, 0.8],
            "model__class_weight": [None, "balanced", "balanced_subsample"]
        }
    },
    "HistGradientBoosting": {
        "pipeline": make_pipeline(
            HistGradientBoostingClassifier(random_state=RANDOM_STATE), X_dev
        ),
        "params": {
            "model__learning_rate": loguniform(0.01, 0.2),
            "model__max_iter": randint(100, 500),
            "model__max_leaf_nodes": randint(7, 40),
            "model__min_samples_leaf": randint(10, 60),
            "model__l2_regularization": loguniform(1e-4, 10)
        }
    },
    "XGBoost": {
        "pipeline": make_pipeline(
            XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                tree_method="hist"
            ),
            X_dev
        ),
        "params": {
            "model__n_estimators": randint(100, 600),
            "model__max_depth": randint(2, 7),
            "model__learning_rate": loguniform(0.01, 0.2),
            "model__subsample": uniform(0.65, 0.35),
            "model__colsample_bytree": uniform(0.65, 0.35),
            "model__min_child_weight": randint(1, 10),
            "model__reg_lambda": loguniform(1e-3, 20),
            "model__scale_pos_weight": [1.0, imbalance_ratio]
        }
    },
    "LightGBM": {
        "pipeline": make_pipeline(
            LGBMClassifier(
                objective="binary",
                random_state=RANDOM_STATE,
                n_jobs=1,
                verbosity=-1
            ),
            X_dev
        ),
        "params": {
            "model__n_estimators": randint(100, 600),
            "model__num_leaves": randint(7, 50),
            "model__max_depth": [-1, 3, 4, 5, 6, 8],
            "model__learning_rate": loguniform(0.01, 0.2),
            "model__subsample": uniform(0.65, 0.35),
            "model__colsample_bytree": uniform(0.65, 0.35),
            "model__min_child_samples": randint(10, 60),
            "model__reg_lambda": loguniform(1e-3, 20),
            "model__class_weight": [None, "balanced"]
        }
    }
}
