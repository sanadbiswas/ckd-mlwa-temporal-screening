perm = permutation_importance(
    FINAL_MODEL,
    X_ext,
    y_ext,
    scoring="roc_auc",
    n_repeats=30,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

perm_df = pd.DataFrame({
    "feature": X_ext.columns,
    "importance_mean": perm.importances_mean,
    "importance_sd": perm.importances_std
}).sort_values("importance_mean", ascending=False)

display(perm_df.round(4))
perm_df.to_csv(RESULTS_DIR / "temporal_permutation_importance.csv", index=False)

feature_label_map = {
    "RIDAGEYR": "Age",
    "RIAGENDR": "Sex",
    "DMDEDUC2": "Education",
    "INDFMPIR": "Income-to-poverty ratio",
    "has_diabetes_dx": "Diagnosed diabetes",
    "has_htn_dx": "Diagnosed hypertension",
    "BMXBMI": "BMI",
    "BMXWAIST": "Waist circumference",
    "smoked_100_cigs": "Smoking history",
    "ALQ130": "Alcohol intake",
    "mean_sbp": "Mean systolic BP",
    "mean_dbp": "Mean diastolic BP"
}

plot_df = perm_df.sort_values("importance_mean", ascending=True).copy()
plot_df["feature_label"] = plot_df["feature"].map(feature_label_map).fillna(plot_df["feature"])
colors = plt.cm.tab20(np.linspace(0, 1, len(plot_df)))

fig, ax = plt.subplots(figsize=(10, 7.5))
bars = ax.barh(
    plot_df["feature_label"],
    plot_df["importance_mean"],
    xerr=plot_df["importance_sd"],
    color=colors,
    edgecolor="black",
    linewidth=0.7,
    error_kw={"elinewidth": 1.3, "capsize": 3, "capthick": 1.3}
)
ax.axvline(0, color="black", linestyle="--", linewidth=1.0)

for bar, val, err in zip(bars, plot_df["importance_mean"], plot_df["importance_sd"]):
    ax.text(
        val + err + 0.0025,
        bar.get_y() + bar.get_height() / 2,
        f"{val:.4f}",
        va="center",
        ha="left",
        fontsize=9
    )

ax.set_xlabel("Decrease in ROC-AUC after permutation", fontsize=13)
ax.set_ylabel("Predictor", fontsize=12)
ax.set_title("Temporal Permutation Feature Importance", fontsize=15, pad=12)
for spine in ax.spines.values():
    spine.set_linewidth(1.8)
    spine.set_color("black")
ax.tick_params(axis="x", labelsize=10, width=1.4, length=5)
ax.tick_params(axis="y", labelsize=10, width=1.4, length=5)
ax.grid(axis="x", linestyle="--", linewidth=0.7, alpha=0.45)
ax.set_axisbelow(True)
ax.set_xlim(
    min(plot_df["importance_mean"].min() - 0.004, -0.008),
    plot_df["importance_mean"].max() + plot_df["importance_sd"].max() + 0.025
)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "temporal_permutation_importance.png", dpi=600, bbox_inches="tight")
plt.savefig(RESULTS_DIR / "temporal_permutation_importance.eps", format="eps", bbox_inches="tight")
plt.savefig(RESULTS_DIR / "temporal_permutation_importance.svg", format="svg", bbox_inches="tight")
plt.show()
