ext_eval = ext.copy()
ext_eval["prob"] = p_ext
ext_eval["age_group"] = pd.cut(
    ext_eval["RIDAGEYR"],
    bins=[17, 39, 59, np.inf],
    labels=["18-39", "40-59", "60+"]
)


def subgroup_bootstrap(
    data,
    group_var,
    y_col="ckd",
    p_col="prob",
    threshold=SCREENING_THRESHOLD,
    n_boot=1000,
    min_cases=10,
    min_controls=10,
    seed=RANDOM_STATE
):
    rng = np.random.default_rng(seed)
    rows = []

    for level, g in data.groupby(group_var, dropna=False, observed=False):
        y = g[y_col].to_numpy(dtype=int)
        p = g[p_col].to_numpy(dtype=float)
        n = len(g)
        cases = int(y.sum())
        controls = n - cases
        base = {
            "variable": group_var,
            "subgroup": str(level),
            "n": n,
            "cases": cases,
            "controls": controls
        }

        if cases < min_cases or controls < min_controls:
            rows.append({**base, "status": "insufficient_events"})
            continue

        point = metrics_from_prob(y, p, threshold)
        boot = []
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            yy = y[idx]
            pp = p[idx]
            if np.unique(yy).size < 2:
                continue
            boot.append(metrics_from_prob(yy, pp, threshold))

        boot = pd.DataFrame(boot)
        row = {**base, "status": "evaluated"}
        for metric in ["roc_auc", "sensitivity", "specificity", "pr_auc"]:
            row[metric] = point[metric]
            row[metric + "_ci_low"] = boot[metric].quantile(0.025)
            row[metric + "_ci_high"] = boot[metric].quantile(0.975)
        rows.append(row)

    return pd.DataFrame(rows)


subgroup_frames = []
for var in ["RIAGENDR", "RIDRETH1", "age_group", "has_diabetes_dx", "has_htn_dx"]:
    subgroup_frames.append(
        subgroup_bootstrap(ext_eval, var, n_boot=1000, seed=RANDOM_STATE + 900)
    )

subgroup_results = pd.concat(subgroup_frames, ignore_index=True)
display(subgroup_results.round(4))
subgroup_results.to_csv(
    RESULTS_DIR / "temporal_subgroup_performance_with_CI.csv", index=False
)


def subgroup_calibration_table(
    data,
    group_var,
    y_col="ckd",
    p_col="prob",
    min_cases=20,
    min_controls=20
):
    rows = []
    for level, g in data.groupby(group_var, dropna=False, observed=False):
        y = g[y_col].to_numpy(dtype=int)
        p = g[p_col].to_numpy(dtype=float)
        n = len(g)
        cases = int(y.sum())
        controls = n - cases
        row = {
            "variable": group_var,
            "subgroup": str(level),
            "n": n,
            "cases": cases,
            "controls": controls
        }

        if cases < min_cases or controls < min_controls:
            row["status"] = "insufficient_events"
            rows.append(row)
            continue

        intercept, slope = calibration_intercept_slope(y, p)
        row.update({
            "status": "evaluated",
            "brier": brier_score_loss(y, p),
            "calibration_intercept": intercept,
            "calibration_slope": slope,
            "ece": weighted_ece(y, p, 10)
        })
        rows.append(row)

    return pd.DataFrame(rows)


subgroup_calibration_frames = []
for var in ["RIAGENDR", "age_group", "has_diabetes_dx", "has_htn_dx"]:
    subgroup_calibration_frames.append(subgroup_calibration_table(ext_eval, var))

subgroup_calibration = pd.concat(subgroup_calibration_frames, ignore_index=True)
display(subgroup_calibration.round(4))
subgroup_calibration.to_csv(
    RESULTS_DIR / "temporal_subgroup_calibration.csv", index=False
)
