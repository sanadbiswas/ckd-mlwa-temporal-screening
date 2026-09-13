# Modular setup: paths, imports, and reproducibility constants.
from pathlib import Path

ROOT = Path(globals().get("PROJECT_ROOT", Path.cwd())).resolve()
RESULTS_DIR = ROOT / "mlwa_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Cleaned development dataset used by the analysis.
_clean_csv = ROOT / "nhanes_ckd_risk_clean.csv"
_clean_csv_gz = ROOT / "nhanes_ckd_risk_clean.csv.gz"
DATA_PATH = _clean_csv if _clean_csv.exists() else _clean_csv_gz

if not DATA_PATH.exists():
    raise FileNotFoundError(
        "Cleaned development data not found. Run modules/01_data_preparation.py "
        "to rebuild it from the public NHANES source files."
    )

print("Project root:", ROOT)
print("Dataset:", DATA_PATH)
print("Results directory:", RESULTS_DIR)

# Imports and reproducibility
import io
import json
import math
import warnings
import platform
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests

from scipy.stats import randint, uniform, loguniform

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    StratifiedKFold,
    RandomizedSearchCV,
    cross_val_predict
)
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN

RANDOM_STATE = 42
OUTER_FOLDS = 5
INNER_FOLDS = 5
N_ITER = 20
N_BOOT = 2000

np.random.seed(RANDOM_STATE)
warnings.filterwarnings("ignore")

print("Python:", platform.python_version())

from IPython.display import display
