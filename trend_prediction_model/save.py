"""
Train and Save Models — run this ONCE
-------------------------------------------
This trains the best model per target (based on our evaluation results)
and saves each one to disk using joblib. After running this once, the
separate export_handoff.py script just LOADS these saved models and
generates the output CSV — no training happens in that script.

Best model per target (from evaluation):
  - Claims       -> Linear Regression
  - Beneficiaries -> Linear Regression
  - Cost         -> XGBoost (log-transformed target)

OUTPUT: saved_models/model_clms.pkl
        saved_models/model_benes.pkl
        saved_models/model_cost.pkl
        saved_models/feature_columns.pkl   (so export script knows the
                                             exact feature order to use)
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

INPUT_PATH = "final/clean.csv"
MODEL_DIR = "saved_models"
RANDOM_STATE = 42

IDENTIFIER_COLUMNS = ["Gnrc_Name", "Brnd_Name", "Gnrc_Name_Code"]
TARGET_COLUMNS = [
    "TARGET_Tot_Clms_2024",
    "TARGET_Tot_Drug_Cst_2024",
    "TARGET_Tot_Benes_2024",
]


def build_xgboost_model() -> XGBRegressor:
    """Regularized for a small dataset — see train_models.py for full explanation."""
    return XGBRegressor(
        n_estimators=100,
        max_depth=3,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.6,
        reg_alpha=1.0,
        reg_lambda=2.0,
        random_state=RANDOM_STATE,
    )


def main():
    print(f"Loading {INPUT_PATH} ...")
    df = pd.read_csv(INPUT_PATH)

    feature_cols = [c for c in df.columns if c not in IDENTIFIER_COLUMNS and c not in TARGET_COLUMNS]
    X = df[feature_cols]

    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Training + saving Claims model (Linear Regression) ...")
    lr_clms = LinearRegression()
    lr_clms.fit(X, df["TARGET_Tot_Clms_2024"])
    joblib.dump(lr_clms, os.path.join(MODEL_DIR, "model_clms.pkl"))

    print("Training + saving Beneficiaries model (Linear Regression) ...")
    lr_benes = LinearRegression()
    lr_benes.fit(X, df["TARGET_Tot_Benes_2024"])
    joblib.dump(lr_benes, os.path.join(MODEL_DIR, "model_benes.pkl"))

    print("Training + saving Cost model (XGBoost, log-transformed) ...")
    y_cost_log = np.log1p(df["TARGET_Tot_Drug_Cst_2024"])
    xgb_cost = build_xgboost_model()
    xgb_cost.fit(X, y_cost_log)
    joblib.dump(xgb_cost, os.path.join(MODEL_DIR, "model_cost.pkl"))

    joblib.dump(feature_cols, os.path.join(MODEL_DIR, "feature_columns.pkl"))

    print(f"\nDone. Saved 3 models + feature list to {MODEL_DIR}/")
    print("You only need to run this script again if you retrain on new data.")
    print("For every-day exporting, just run export_handoff.py from now on.")


if __name__ == "__main__":
    main()