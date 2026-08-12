"""
Export Handoff CSV — for the Therapeutic Equivalence / Formulary Impact team
--------------------------------------------------------------------------------
This script does NOT train anything. It only:
  1. Loads the 3 already-trained models from saved_models/ (see save_models.py)
  2. Loads the model-ready dataset
  3. Generates predictions for all 132 drugs
  4. Adds a priority flag + explanatory note
  5. Saves the final handoff CSV

RUN ORDER
-----------
1. Run save_models.py ONCE (trains + saves the 3 models to disk)
2. Run this script (export_handoff.py) any time you need to regenerate
   the handoff CSV — this step is fast since no training happens here

OUTPUT: team_handoff/trend_prediction_handoff.csv
Columns: Gnrc_Name, Brnd_Name, Predicted_2024_Claims, Predicted_2024_Benes,
         Predicted_2024_Cost, Cost_Growth_Pct, Growth_Flag, Note
"""

import pandas as pd
import numpy as np
import joblib
import os

INPUT_PATH = "final/clean.csv"
MODEL_DIR = "saved_models"
OUTPUT_PATH = "team_handoff/trend_prediction_handoff.csv"

GROWTH_FLAG_PERCENTILE = 0.75


def main():
    # --- Load already-trained models (no training happens here) ---
    print(f"Loading saved models from {MODEL_DIR}/ ...")
    lr_clms = joblib.load(os.path.join(MODEL_DIR, "model_clms.pkl"))
    lr_benes = joblib.load(os.path.join(MODEL_DIR, "model_benes.pkl"))
    xgb_cost = joblib.load(os.path.join(MODEL_DIR, "model_cost.pkl"))
    feature_cols = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))

    # --- Load data ---
    print(f"Loading {INPUT_PATH} ...")
    df = pd.read_csv(INPUT_PATH)
    X = df[feature_cols]

    handoff = df[["Gnrc_Name", "Brnd_Name"]].copy()

    # --- Predict (no fitting — just .predict() using the loaded models) ---
    print("Generating predictions ...")
    handoff["Predicted_2024_Claims"] = lr_clms.predict(X).round(0).astype(int)
    handoff["Predicted_2024_Benes"] = lr_benes.predict(X).round(0).astype(int)
    handoff["Predicted_2024_Cost"] = np.expm1(xgb_cost.predict(X)).round(2)

    # --- Growth % vs. most recent known year (2023) ---
    handoff["Cost_Growth_Pct"] = (
        (handoff["Predicted_2024_Cost"] - df["Tot_Drug_Cst_2023"]) / df["Tot_Drug_Cst_2023"] * 100
    ).round(1)

    # --- Priority flag: high growth OR already a top-spend drug ---
    growth_threshold = handoff["Cost_Growth_Pct"].quantile(GROWTH_FLAG_PERCENTILE)
    cost_threshold = handoff["Predicted_2024_Cost"].quantile(GROWTH_FLAG_PERCENTILE)
    handoff["Growth_Flag"] = np.where(
        (handoff["Cost_Growth_Pct"] >= growth_threshold)
        | (handoff["Predicted_2024_Cost"] >= cost_threshold),
        "HIGH_GROWTH",
        "STABLE",
    )

    # --- Note for top-10% historical-cost drugs (regularization caveat) ---
    top_cost_cutoff = df["Tot_Drug_Cst_2023"].quantile(0.9)
    handoff["Note"] = np.where(
        df.set_index("Gnrc_Name").loc[handoff["Gnrc_Name"], "Tot_Drug_Cst_2023"].values >= top_cost_cutoff,
        "Top-10% historical cost — verify growth % against raw yearly values",
        "",
    )

    handoff = handoff.sort_values("Predicted_2024_Cost", ascending=False).reset_index(drop=True)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    handoff.to_csv(OUTPUT_PATH, index=False)

    print(f"\nDone. Saved {OUTPUT_PATH}")
    print(f"Shape: {handoff.shape[0]} rows x {handoff.shape[1]} columns")
    print(f"HIGH_GROWTH drugs flagged: {(handoff['Growth_Flag'] == 'HIGH_GROWTH').sum()} out of {len(handoff)}")
    print(f"\nPreview:")
    print(handoff.head(5).to_string(index=False))


if __name__ == "__main__":
    main()