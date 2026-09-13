if BEST_CALIBRATION == "Sigmoid":
    selected_internal_prob = sigmoid_oof
elif BEST_CALIBRATION == "Isotonic":
    selected_internal_prob = isotonic_oof
else:
    selected_internal_prob = raw_oof_fixed

SCREENING_THRESHOLD = threshold_for_target_sensitivity(
    y_dev,
    selected_internal_prob,
    target=0.80
)

internal_operating = metrics_from_prob(
    y_dev,
    selected_internal_prob,
    threshold=SCREENING_THRESHOLD
)

print("Frozen screening threshold:", round(SCREENING_THRESHOLD, 6))
print(pd.Series(internal_operating).round(4))
