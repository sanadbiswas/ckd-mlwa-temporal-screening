w_dev = survey_dev["WTMEC2YR"].to_numpy()

weighted_dev_metrics = metrics_from_prob(
    y_dev,
    selected_internal_prob,
    threshold=SCREENING_THRESHOLD,
    sample_weight=w_dev
)

weighted_ext_metrics = metrics_from_prob(
    y_ext,
    p_ext,
    threshold=SCREENING_THRESHOLD,
    sample_weight=w_ext
)

survey_weighted_table = pd.DataFrame([
    {"cohort": "2017-2018 development OOF", "survey_weight": "WTMEC2YR", **weighted_dev_metrics},
    {"cohort": "2021-2023 temporal", "survey_weight": "WTPH2YR", **weighted_ext_metrics}
])

display(survey_weighted_table.round(4))
survey_weighted_table.to_csv(
    RESULTS_DIR / "survey_weighted_performance.csv", index=False
)
