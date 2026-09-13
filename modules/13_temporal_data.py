BASE_L = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles"


def read_l(filename):
    return read_xpt_url(f"{BASE_L}/{filename}")


def select_or_nan(frame, cols, label):
    present = [c for c in cols if c in frame.columns]
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        print(f"[{label}] missing: {missing}")
    out = frame[["SEQN"] + present].copy()
    for c in missing:
        out[c] = np.nan
    return out


def ckd_epi_2021(scr, age, sex_code):
    female = np.asarray(sex_code) == 2
    kappa = np.where(female, 0.7, 0.9)
    alpha = np.where(female, -0.241, -0.302)
    sex_mult = np.where(female, 1.012, 1.0)
    ratio = np.asarray(scr) / kappa
    return (
        142
        * np.minimum(ratio, 1) ** alpha
        * np.maximum(ratio, 1) ** (-1.200)
        * (0.9938 ** np.asarray(age))
        * sex_mult
    )


demo_l_full = read_l("DEMO_L.XPT")
demo_l = select_or_nan(
    demo_l_full,
    [
        "RIDAGEYR", "RIAGENDR", "RIDRETH1", "DMDEDUC2", "INDFMPIR",
        "WTMEC2YR", "SDMVPSU", "SDMVSTRA"
    ],
    "DEMO_L"
)

bio_l = select_or_nan(read_l("BIOPRO_L.XPT"), ["LBXSCR", "WTPH2YR"], "BIOPRO_L")
acr_l = select_or_nan(read_l("ALB_CR_L.XPT"), ["URDACT"], "ALB_CR_L")
bpq_l = select_or_nan(read_l("BPQ_L.XPT"), ["BPQ020"], "BPQ_L")
diq_l = select_or_nan(read_l("DIQ_L.XPT"), ["DIQ010"], "DIQ_L")
bmx_l = select_or_nan(read_l("BMX_L.XPT"), ["BMXBMI", "BMXWAIST"], "BMX_L")
smq_l = select_or_nan(read_l("SMQ_L.XPT"), ["SMQ020"], "SMQ_L")
alq_l = select_or_nan(read_l("ALQ_L.XPT"), ["ALQ130"], "ALQ_L")
bpx_l = select_or_nan(
    read_l("BPXO_L.XPT"),
    ["BPXOSY1", "BPXOSY2", "BPXOSY3", "BPXODI1", "BPXODI2", "BPXODI3"],
    "BPXO_L"
)

ext = (
    demo_l
    .merge(bio_l, on="SEQN", how="inner")
    .merge(acr_l, on="SEQN", how="inner")
    .merge(bpq_l, on="SEQN", how="left")
    .merge(diq_l, on="SEQN", how="left")
    .merge(bmx_l, on="SEQN", how="left")
    .merge(smq_l, on="SEQN", how="left")
    .merge(alq_l, on="SEQN", how="left")
    .merge(bpx_l, on="SEQN", how="left")
)

n_before_age = len(ext)
ext = ext[ext["RIDAGEYR"] >= 18].copy()
n_removed_age = n_before_age - len(ext)
print(f"Linked temporal source participants: {n_before_age:,}")
print(f"Excluded age <18: {n_removed_age:,}")
print(f"Adults after age filter: {len(ext):,}")

# Require both outcome-defining laboratory values before phenotype construction.
n_before_labs = len(ext)
ext = ext.dropna(subset=["LBXSCR", "URDACT"]).reset_index(drop=True)
n_removed_labs = n_before_labs - len(ext)
print(f"Excluded for missing serum creatinine or urinary ACR: {n_removed_labs:,}")
print(f"Temporal validation cohort retained: {len(ext):,}")

ext["egfr"] = ckd_epi_2021(ext["LBXSCR"], ext["RIDAGEYR"], ext["RIAGENDR"])
assert ext["LBXSCR"].notna().all() and ext["URDACT"].notna().all()

ext["ckd"] = ((ext["egfr"] < 60) | (ext["URDACT"] >= 30)).astype(int)
ext["low_egfr"] = (ext["egfr"] < 60).astype(int)
ext["albuminuria"] = (ext["URDACT"] >= 30).astype(int)

print("Temporal phenotype-positive cases:", int(ext["ckd"].sum()))
print("Temporal unweighted phenotype prevalence:", round(ext["ckd"].mean(), 6))

yn = {1: "yes", 2: "no", 7: np.nan, 9: np.nan}
ext["has_htn_dx"] = ext["BPQ020"].map(yn)
ext["has_diabetes_dx"] = ext["DIQ010"].map(yn)
ext["smoked_100_cigs"] = ext["SMQ020"].map(yn)
ext["RIAGENDR"] = ext["RIAGENDR"].map({1: "male", 2: "female"})
ext["RIDRETH1"] = ext["RIDRETH1"].map({
    1: "mexican_american", 2: "other_hispanic", 3: "non_hispanic_white",
    4: "non_hispanic_black", 5: "other_or_multiracial"
})
ext["DMDEDUC2"] = ext["DMDEDUC2"].map({
    1: "less_than_9th_grade", 2: "9th_to_11th_grade",
    3: "high_school_grad_or_GED", 4: "some_college_or_AA_degree",
    5: "college_grad_or_above", 7: np.nan, 9: np.nan
})
ext["ALQ130"] = ext["ALQ130"].replace({777: np.nan, 999: np.nan})
ext["mean_sbp"] = ext[["BPXOSY1", "BPXOSY2", "BPXOSY3"]].mean(axis=1)
ext["mean_dbp"] = ext[["BPXODI1", "BPXODI2", "BPXODI3"]].mean(axis=1)

X_ext = ext[PRIMARY_FEATURES].copy()
y_ext = ext["ckd"].astype(int).copy()
w_ext = ext["WTPH2YR"].copy()

print("Temporal validation n:", len(ext))
print("Temporal survey weight: WTPH2YR")
print("Positive temporal weights:", int((w_ext > 0).sum()))
print("Unweighted prevalence:", round(y_ext.mean(), 4))
print("Survey-weighted prevalence:", round(np.average(y_ext, weights=w_ext), 4))
