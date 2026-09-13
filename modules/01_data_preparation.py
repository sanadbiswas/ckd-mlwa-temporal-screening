# NHANES 2017-2018 data acquisition and preparation.

from pathlib import Path
PROJECT_ROOT = Path(globals().get("PROJECT_ROOT", Path.cwd())).resolve()

import io
import requests
import numpy as np
import pandas as pd

pd.set_option("display.max_columns", None)

BASE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles"
OUTPUT_FILE = PROJECT_ROOT / "nhanes_ckd_risk_clean.csv"

print("CDC/NCHS NHANES source:", BASE)


def read_nhanes_xpt(filename):
    """Download one NHANES XPT file from CDC/NCHS and return a DataFrame."""
    url = f"{BASE}/{filename}"
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return pd.read_sas(io.BytesIO(response.content), format="xport")


def load_development_cycle():
    # Demographics and socioeconomic variables
    demo = read_nhanes_xpt("DEMO_J.XPT")[[
        "SEQN",
        "RIDAGEYR",
        "RIAGENDR",
        "RIDRETH1",
        "DMDEDUC2",
        "INDFMPIR",
    ]]

    # Outcome-source laboratory variables
    bio = read_nhanes_xpt("BIOPRO_J.XPT")[[
        "SEQN",
        "LBXSCR",      # serum creatinine (mg/dL)
    ]]

    albcr = read_nhanes_xpt("ALB_CR_J.XPT")[[
        "SEQN",
        "URDACT",      # urine albumin-to-creatinine ratio (mg/g)
    ]]

    # Self-reported clinical history
    bpq = read_nhanes_xpt("BPQ_J.XPT")[[
        "SEQN",
        "BPQ020",
    ]].rename(columns={"BPQ020": "has_htn_dx"})

    diq = read_nhanes_xpt("DIQ_J.XPT")[[
        "SEQN",
        "DIQ010",
    ]].rename(columns={"DIQ010": "has_diabetes_dx"})

    # Anthropometric measures
    bmx = read_nhanes_xpt("BMX_J.XPT")[[
        "SEQN",
        "BMXBMI",
        "BMXWAIST",
    ]]

    # Smoking history
    smq = read_nhanes_xpt("SMQ_J.XPT")[[
        "SEQN",
        "SMQ020",
    ]].rename(columns={"SMQ020": "smoked_100_cigs"})

    # Physical-activity questionnaire variables retained for sensitivity analyses
    paq = read_nhanes_xpt("PAQ_J.XPT")[[
        "SEQN",
        "PAQ605",
        "PAQ620",
        "PAQ635",
        "PAQ650",
        "PAQ665",
    ]]

    # Alcohol use
    alq = read_nhanes_xpt("ALQ_J.XPT")[[
        "SEQN",
        "ALQ130",
    ]]

    # Manual auscultatory blood-pressure readings used in the 2017–2018 development cohort
    bpx = read_nhanes_xpt("BPX_J.XPT")[[
        "SEQN",
        "BPXSY1", "BPXSY2", "BPXSY3",
        "BPXDI1", "BPXDI2", "BPXDI3",
    ]]

    # Predictor/risk-factor side: left joins preserve participants when a predictor is missing.
    risk = (
        demo
        .merge(bpq, on="SEQN", how="left")
        .merge(diq, on="SEQN", how="left")
        .merge(bmx, on="SEQN", how="left")
        .merge(smq, on="SEQN", how="left")
        .merge(paq, on="SEQN", how="left")
        .merge(alq, on="SEQN", how="left")
        .merge(bpx, on="SEQN", how="left")
    )

    # Outcome-source laboratory files are required for phenotype construction.
    labs = bio.merge(albcr, on="SEQN", how="inner")

    # Keep participants present in the required laboratory files.
    return risk.merge(labs, on="SEQN", how="inner")


df = load_development_cycle()
print("Linked source dataset shape:", df.shape)


n_before = len(df)

df = (
    df.loc[df["RIDAGEYR"] >= 18]
      .reset_index(drop=True)
)

n_removed_age = n_before - len(df)

print(f"Source participants: {n_before:,}")
print(f"Excluded age <18: {n_removed_age:,}")
print(f"Adults after age filter: {len(df):,}")

# Require both outcome-defining laboratory values before phenotype construction.
n_before_labs = len(df)

df = (
    df.dropna(subset=["LBXSCR", "URDACT"])
      .reset_index(drop=True)
)

n_removed_labs = n_before_labs - len(df)

print(f"Excluded for missing serum creatinine or urinary ACR: {n_removed_labs:,}")
print(f"Development adults retained: {len(df):,}")


def ckd_epi_2021(scr, age, sex_code):
    """2021 race-free CKD-EPI creatinine equation."""
    is_female = (sex_code == 2)

    kappa = np.where(is_female, 0.7, 0.9)
    alpha = np.where(is_female, -0.241, -0.302)
    sex_multiplier = np.where(is_female, 1.012, 1.0)

    ratio = scr / kappa

    return (
        142
        * np.minimum(ratio, 1) ** alpha
        * np.maximum(ratio, 1) ** -1.200
        * (0.9938 ** age)
        * sex_multiplier
    )


df["egfr"] = ckd_epi_2021(
    df["LBXSCR"],
    df["RIDAGEYR"],
    df["RIAGENDR"],
)

# Confirm that outcome-defining laboratory values are complete.
assert df["LBXSCR"].notna().all() and df["URDACT"].notna().all(), (
    "Unexpected missing creatinine or ACR values reached outcome "
    "construction; the exclusion step above should have removed these."
)

df["ckd"] = (
    (df["egfr"] < 60) |
    (df["URDACT"] >= 30)
).astype(int)

print("Phenotype-positive cases:", int(df["ckd"].sum()))
print("Unweighted phenotype prevalence:", round(df["ckd"].mean(), 6))


df = df.drop(columns=[
    "LBXSCR",
    "URDACT",
    "egfr",
])

print("Shape after removing outcome-source laboratory variables:", df.shape)


# Yes/no questionnaire fields:
# 1 = Yes, 2 = No, 7 = Refused, 9 = Don't know
yes_no_map = {
    1: "yes",
    2: "no",
    7: np.nan,
    9: np.nan,
}

yes_no_columns = [
    "has_htn_dx",
    "has_diabetes_dx",
    "smoked_100_cigs",
    "PAQ605",
    "PAQ620",
    "PAQ635",
    "PAQ650",
    "PAQ665",
]

for col in yes_no_columns:
    df[col] = df[col].map(yes_no_map)


# Sex
df["RIAGENDR"] = df["RIAGENDR"].map({
    1: "male",
    2: "female",
})


# Race/ethnicity
df["RIDRETH1"] = df["RIDRETH1"].map({
    1: "mexican_american",
    2: "other_hispanic",
    3: "non_hispanic_white",
    4: "non_hispanic_black",
    5: "other_or_multiracial",
})


# Educational attainment.
# NHANES DMDEDUC2 is not administered in the same way to ages 18–19,
# so missing values in those participants are retained and handled later
# by the modeling preprocessing pipeline.
df["DMDEDUC2"] = df["DMDEDUC2"].map({
    1: "less_than_9th_grade",
    2: "9th_to_11th_grade",
    3: "high_school_grad_or_GED",
    4: "some_college_or_AA_degree",
    5: "college_grad_or_above",
    7: np.nan,
    9: np.nan,
})


# Alcohol: NHANES uses 777 = Refused and 999 = Don't know.
df["ALQ130"] = df["ALQ130"].replace({
    777: np.nan,
    999: np.nan,
})

print("Questionnaire recoding complete.")


expected_columns = [
    "SEQN",
    "RIDAGEYR",
    "RIAGENDR",
    "RIDRETH1",
    "DMDEDUC2",
    "INDFMPIR",
    "has_htn_dx",
    "has_diabetes_dx",
    "BMXBMI",
    "BMXWAIST",
    "smoked_100_cigs",
    "PAQ605",
    "PAQ620",
    "PAQ635",
    "PAQ650",
    "PAQ665",
    "ALQ130",
    "BPXSY1",
    "BPXSY2",
    "BPXSY3",
    "BPXDI1",
    "BPXDI2",
    "BPXDI3",
    "ckd",
]

assert list(df.columns) == expected_columns, "Unexpected output column order/schema."

print("Validation checks passed.")
print("Final shape:", df.shape)
print("CKD-compatible phenotype prevalence:", f"{df['ckd'].mean():.4%}")


missing_summary = (
    df.isna()
      .sum()
      .rename("missing_n")
      .to_frame()
)

missing_summary["missing_pct"] = (
    100 * missing_summary["missing_n"] / len(df)
)

missing_summary = (
    missing_summary
    .query("missing_n > 0")
    .sort_values("missing_n", ascending=False)
)

print("Per-variable missingness after outcome construction:")
print(missing_summary.to_string())


df.to_csv(OUTPUT_FILE, index=False)

# Use the regenerated dataset in downstream pipeline stages.
DATA_PATH = OUTPUT_FILE

print(f"Saved: {OUTPUT_FILE}")
print("Rows:", len(df))
print("Columns:", len(df.columns))
