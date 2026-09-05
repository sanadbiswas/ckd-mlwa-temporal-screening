import json
from pathlib import Path

NOTEBOOK = Path("CKD_MLWA_Research.ipynb")
REPO = "sanadbiswas/ckd-mlwa-temporal-screening"
DATA_URL = f"https://raw.githubusercontent.com/{REPO}/main/nhanes_ckd_risk_data.csv"

with NOTEBOOK.open("r", encoding="utf-8") as f:
    nb = json.load(f)

# -----------------------------------------------------------------------------
# 1. Final notebook title / reproducibility description
# -----------------------------------------------------------------------------
if nb.get("cells"):
    nb["cells"][0]["cell_type"] = "markdown"
    nb["cells"][0]["source"] = [
        "# CKD Screening Paper — Final Reproducibility Pipeline (v5)\n",
        "\n",
        "This notebook supports the manuscript *Temporal Validation of a Non-Laboratory Machine Learning Framework for Chronic Kidney Disease Screening in U.S. Adults.*\n",
        "\n",
        "It is designed to run directly in Google Colab or Jupyter without a Google Drive mount. The cleaned 2017–2018 development dataset is downloaded from the public GitHub repository, while the 2021–2023 temporal cohort is reconstructed from official CDC/NCHS files.\n",
        "\n",
        "Final survey-weight specification: `WTMEC2YR` for 2017–2018 development evaluation and `WTPH2YR` for August 2021–August 2023 temporal evaluation. Survey weights are used only for secondary population-weighted evaluation.\n",
    ]

# -----------------------------------------------------------------------------
# 2. Replace Google Drive setup with GitHub-hosted development data
# -----------------------------------------------------------------------------
setup_source = [
    "# =========================================================\n",
    "# 0. Google Colab / Jupyter setup — no Google Drive required\n",
    "# =========================================================\n",
    "\n",
    "!pip -q install -U lightgbm xgboost imbalanced-learn statsmodels requests\n",
    "\n",
    "from pathlib import Path\n",
    "import urllib.request\n",
    "\n",
    f"DATA_URL = \"{DATA_URL}\"\n",
    "ROOT = Path('/content') if Path('/content').exists() else Path.cwd()\n",
    "DATA_PATH = ROOT / 'nhanes_ckd_risk_data.csv'\n",
    "RESULTS_DIR = ROOT / 'mlwa_research_grade_results_v5'\n",
    "RESULTS_DIR.mkdir(parents=True, exist_ok=True)\n",
    "\n",
    "if not DATA_PATH.exists():\n",
    "    print('Downloading development dataset from GitHub...')\n",
    "    urllib.request.urlretrieve(DATA_URL, DATA_PATH)\n",
    "\n",
    "print('Dataset:', DATA_PATH)\n",
    "print('Exists:', DATA_PATH.exists())\n",
    "print('Results directory:', RESULTS_DIR)\n",
    "\n",
    "if not DATA_PATH.exists():\n",
    "    raise FileNotFoundError(f'Dataset not found: {DATA_PATH}')\n",
]

setup_done = False
for cell in nb.get("cells", []):
    if cell.get("cell_type") != "code":
        continue
    src = "".join(cell.get("source", []))
    if "drive.mount" in src or "Shuvo-Paper" in src or "nhanes_ckd_risk_clean.csv" in src or "DATA_URL" in src:
        cell["source"] = setup_source
        setup_done = True
        break

if not setup_done:
    nb["cells"].insert(1, {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": setup_source,
    })

# -----------------------------------------------------------------------------
# 3. Apply final v5 source-code revisions throughout notebook
# -----------------------------------------------------------------------------
for cell in nb.get("cells", []):
    src = "".join(cell.get("source", []))

    src = src.replace('from google.colab import drive\n', '')
    src = src.replace('drive.mount("/content/drive")\n', '')
    src = src.replace("drive.mount('/content/drive')\n", '')
    src = src.replace('/content/drive/MyDrive/Shuvo-Paper', '/content')
    src = src.replace('nhanes_ckd_risk_clean.csv', 'nhanes_ckd_risk_data.csv')
    # Normalize only legacy result-directory names, avoiding repeated _v5 suffixes.
    src = src.replace("mlwa_research_grade_results_v5_v5", "mlwa_research_grade_results_v5")
    src = src.replace("ROOT / 'mlwa_research_grade_results'", "ROOT / 'mlwa_research_grade_results_v5'")
    src = src.replace('ROOT / "mlwa_research_grade_results"', 'ROOT / "mlwa_research_grade_results_v5"')

    # Avoid nested CPU parallelism: RandomizedSearchCV owns parallelization.
    src = src.replace('RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)',
                      'RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1)')
    src = src.replace('random_state=RANDOM_STATE,\n                n_jobs=-1,\n                tree_method="hist"',
                      'random_state=RANDOM_STATE,\n                n_jobs=1,\n                tree_method="hist"')
    src = src.replace('LGBMClassifier(\n                objective="binary",\n                random_state=RANDOM_STATE,\n                n_jobs=-1,',
                      'LGBMClassifier(\n                objective="binary",\n                random_state=RANDOM_STATE,\n                n_jobs=1,')

    # Temporal survey weighting: WTPH2YR for 2021–2023.
    if 'BIOPRO_L.XPT' in src:
        src = src.replace(
            'select_or_nan(read_l("BIOPRO_L.XPT"), ["LBXSCR"], "BIOPRO_L")',
            'select_or_nan(read_l("BIOPRO_L.XPT"), ["LBXSCR", "WTPH2YR"], "BIOPRO_L")'
        )
    if 'w_ext = ext["WTMEC2YR"]' in src:
        src = src.replace('w_ext = ext["WTMEC2YR"].copy()',
                          'w_ext = ext["WTPH2YR"].copy()')
    if '"2021-2023 temporal", **weighted_ext_metrics' in src:
        src = src.replace(
            '{"cohort": "2017-2018 development OOF", **weighted_dev_metrics},\n'
            '    {"cohort": "2021-2023 temporal", **weighted_ext_metrics}',
            '{"cohort": "2017-2018 development OOF", "survey_weight": "WTMEC2YR", **weighted_dev_metrics},\n'
            '    {"cohort": "2021-2023 temporal", "survey_weight": "WTPH2YR", **weighted_ext_metrics}'
        )

    src = src.replace(
        'Because the study uses MEC examination/laboratory variables, `WTMEC2YR` is used\nfor population-weighted sensitivity analyses.',
        'For the 2017–2018 development cohort, `WTMEC2YR` is used for the secondary population-weighted evaluation. For August 2021–August 2023, `WTPH2YR` is used because the phenotype includes serum creatinine.'
    )

    if src != "".join(cell.get("source", [])):
        cell["source"] = src.splitlines(keepends=True)

# -----------------------------------------------------------------------------
# 4. Remove stale outputs / execution metadata
# -----------------------------------------------------------------------------
for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
        md = cell.get("metadata", {})
        md.pop("executionInfo", None)
        md.pop("outputId", None)
        if isinstance(md.get("colab"), dict):
            md["colab"].pop("base_uri", None)
        cell["metadata"] = md

# -----------------------------------------------------------------------------
# 5. Final metadata and checks
# -----------------------------------------------------------------------------
nb.setdefault("metadata", {})["revision_note"] = (
    "Final v5 GitHub-first reproducibility notebook: no Google Drive path, "
    "GitHub-hosted development data, WTPH2YR temporal survey weighting, and "
    "model-level n_jobs=1 for RF/XGBoost/LightGBM."
)

serialized = json.dumps(nb, ensure_ascii=False, indent=1)
for forbidden in [
    "/content/drive/MyDrive/Shuvo-Paper",
    "drive.mount(",
    "nhanes_ckd_risk_clean.csv",
    "mlwa_research_grade_results_v5_v5",
]:
    if forbidden in serialized:
        raise RuntimeError(f"Forbidden stale reference remains: {forbidden}")

if DATA_URL not in serialized:
    raise RuntimeError("GitHub development-data URL was not inserted")

NOTEBOOK.write_text(serialized, encoding="utf-8")
print(f"Updated {NOTEBOOK}")
print(f"Cells: {len(nb.get('cells', []))}")
print("GitHub-first setup: OK")
print("WTPH2YR temporal weighting revision: applied where matching cells were found")
