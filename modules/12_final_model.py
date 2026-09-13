if BEST_CALIBRATION == "Sigmoid":
    FINAL_MODEL = clone(calibrated_sigmoid)
elif BEST_CALIBRATION == "Isotonic":
    FINAL_MODEL = clone(calibrated_isotonic)
else:
    FINAL_MODEL = clone(fixed_selected)

FINAL_MODEL.fit(X_dev, y_dev)
print("Final frozen model fitted.")
