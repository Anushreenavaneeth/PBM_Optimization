"""
Export Handoff CSV — for the Therapeutic Equivalence / Formulary Impact team
--------------------------------------------------------------------------------
This script does NOT train anything. It only:
  1. Loads the already-trained models from saved_models/ (see save_models.py)
  2. Loads the model-ready dataset
  3. Generates RAW ML predictions
  4. BLENDS raw predictions with a simple trend-extrapolation, weighted by
     each drug's beneficiary volume (see blend_predictions() below) — this
     reduces the gap between predicted and actual for low-volume/volatile
     specialty drugs, without discarding the ML model's signal entirely
  5. Adds a priority flag + explanatory note
  6. Saves the final handoff CSV

WHY BLENDING WAS ADDED
--------------------------
For high-volume common drugs, the ML model's raw predictions closely track
history. For low-volume specialty drugs (a few hundred patients or fewer),
raw ML predictions can jump further from the historical trend than is
useful, because these drugs have inherently noisier year-to-year data and
the model has less signal to learn a stable pattern from.

Blending pulls low-volume drug predictions toward a simple, stable trend
extrapolation (last known value x average historical growth rate), while
leaving high-volume drug predictions almost entirely as the ML model
produced them. This is a standard variance-reduction technique — it does
NOT fabricate accuracy, it just avoids over-trusting the ML model exactly
where it has the least data to be confident.

RUN ORDER
-----------
1. Run save_models.py ONCE (trains + saves the models to disk)
2. Run this script any time you need to regenerate the handoff CSV

OUTPUT: team_handoff/trend_prediction_handoff.csv
Columns: Gnrc_Name, Brnd_Name, Predicted_2024_Claims, Predicted_2024_Benes,
         Predicted_2024_Cost, Cost_Growth_Pct, Growth_Flag, Blend_Weight, Note
"""

import pandas as pd
import numpy as np
import joblib
import os

INPUT_PATH = "final/clean.csv"
MODEL_DIR = "saved_models"
OUTPUT_PATH = "team_handoff/trend_prediction_handoff.csv"

GROWTH_FLAG_PERCENTILE = 0.75

# Volume thresholds for blending: below LOW_VOLUME_CUTOFF, lean heavily on
# the simple trend extrapolation. Above HIGH_VOLUME_CUTOFF, trust the ML
# model almost entirely. In between, blend proportionally.
LOW_VOLUME_CUTOFF = 500
HIGH_VOLUME_CUTOFF = 50_000


def calculate_blend_weight(beneficiaries: pd.Series) -> pd.Series:
    """
    Returns a weight between 0 and 1 for how much to trust the ML model
    vs. the simple trend extrapolation, based on 2023 beneficiary volume.

    weight = 0   -> fully trust simple trend extrapolation (low volume)
    weight = 1   -> fully trust the ML model (high volume)
    """
    weight = (beneficiaries - LOW_VOLUME_CUTOFF) / (HIGH_VOLUME_CUTOFF - LOW_VOLUME_CUTOFF)
    return weight.clip(lower=0, upper=1)


def simple_trend_extrapolation(value_2023: pd.Series, avg_growth_rate: pd.Series) -> pd.Series:
    """A basic, stable estimate: last known value grown by its own historical average rate."""
    return value_2023 * (1 + avg_growth_rate)


def blend_predictions(ml_prediction: pd.Series, trend_prediction: pd.Series, weight: pd.Series) -> pd.Series:
    """weight=1 -> pure ML model. weight=0 -> pure simple trend extrapolation."""
    return weight * ml_prediction + (1 - weight) * trend_prediction


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

    # --- Generate RAW ML predictions ---
    print("Generating raw ML predictions ...")
    raw_clms = lr_clms.predict(X)
    raw_benes = lr_benes.predict(X)
    raw_cost = np.expm1(xgb_cost.predict(X))

    # --- Calculate blend weight per drug, based on 2023 beneficiary volume ---
    blend_weight = calculate_blend_weight(df["Tot_Benes_2023"])

    # --- Simple trend extrapolations (the "stable" fallback) ---
    trend_clms = simple_trend_extrapolation(df["Tot_Clms_2023"], df["avg_clms_growth"])
    trend_benes = simple_trend_extrapolation(df["Tot_Benes_2023"], df["avg_clms_growth"])  # benes tends to move with claims
    trend_cost = simple_trend_extrapolation(df["Tot_Drug_Cst_2023"], df["avg_cst_growth"])

    # --- Blend: high-volume drugs stay close to ML model, low-volume drugs
    # lean toward the simple, more stable trend extrapolation ---
    print("Blending ML predictions with trend extrapolation (volume-weighted) ...")
    handoff["Predicted_2024_Claims"] = blend_predictions(raw_clms, trend_clms, blend_weight).round(0).astype(int)
    handoff["Predicted_2024_Benes"] = blend_predictions(raw_benes, trend_benes, blend_weight).round(0).astype(int)
    handoff["Predicted_2024_Cost"] = blend_predictions(raw_cost, trend_cost, blend_weight).round(2)
    handoff["Blend_Weight"] = blend_weight.round(2)  # 1.0 = pure ML, 0.0 = pure trend extrapolation

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

    # --- Note: flag low-volume drugs where the blend leaned on trend extrapolation ---
    handoff["Note"] = np.where(
        handoff["Blend_Weight"] < 0.5,
        "Low patient volume — prediction blended with simple trend extrapolation for stability",
        "",
    )

    handoff = handoff.sort_values("Predicted_2024_Cost", ascending=False).reset_index(drop=True)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    handoff.to_csv(OUTPUT_PATH, index=False)

    print(f"\nDone. Saved {OUTPUT_PATH}")
    print(f"Shape: {handoff.shape[0]} rows x {handoff.shape[1]} columns")
    print(f"HIGH_GROWTH drugs flagged: {(handoff['Growth_Flag'] == 'HIGH_GROWTH').sum()} out of {len(handoff)}")
    print(f"Drugs with blend weight < 0.5 (leaning on trend extrapolation): "
          f"{(handoff['Blend_Weight'] < 0.5).sum()} out of {len(handoff)}")
    print(f"\nPreview:")
    print(handoff.head(5).to_string(index=False))


if __name__ == "__main__":
    main()