"""
PMPM Handoff File
-----------------

Creates a clean handoff file for the Therapeutic Equivalence
Optimization module.

Input:
    pmpm_tracking.csv

Output:
    pmpm_handoff.csv

The handoff contains only PMPM-related information.
Drug utilization predictions remain in the separate
trend_prediction_handoff.csv file.
"""

import os
import numpy as np
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_PATH = "pmpm_output/pmpm_tracking.csv"
OUTPUT_DIR = "team_handoff"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "pmpm_handoff.csv")

REFERENCE_YEAR = 2023
PREDICTION_YEAR = 2024


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("PMPM HANDOFF CREATION")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Load PMPM tracking data
    # --------------------------------------------------------
    print(f"\nLoading:\n{INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df):,} rows")

    # --------------------------------------------------------
    # 2. Separate actual 2023 and predicted 2024
    # --------------------------------------------------------
    actual_2023 = df[
        (df["year"] == REFERENCE_YEAR) & (df["source"] == "actual")
    ][["Gnrc_Name", "Brnd_Name", "PMPM"]].copy()

    actual_2023 = actual_2023.rename(columns={"PMPM": "PMPM_2023"})

    predicted_2024 = df[
        (df["year"] == PREDICTION_YEAR) & (df["source"] == "predicted")
    ][["Gnrc_Name", "Brnd_Name", "PMPM", "Plausible_Prediction"]].copy()

    predicted_2024 = predicted_2024.rename(columns={"PMPM": "Predicted_2024_PMPM"})

    # --------------------------------------------------------
    # 3. Merge 2023 actual + 2024 predicted
    # --------------------------------------------------------
    handoff = predicted_2024.merge(
        actual_2023[["Gnrc_Name", "PMPM_2023"]], on="Gnrc_Name", how="left"
    )

    # --------------------------------------------------------
    # 4. Calculate PMPM growth
    # --------------------------------------------------------
    handoff["PMPM_Growth"] = np.nan
    valid = (handoff["PMPM_2023"] > 0) & (handoff["Predicted_2024_PMPM"] > 0)

    handoff.loc[valid, "PMPM_Growth"] = (
        (handoff.loc[valid, "Predicted_2024_PMPM"] - handoff.loc[valid, "PMPM_2023"])
        / handoff.loc[valid, "PMPM_2023"]
    ) * 100

    # --------------------------------------------------------
    # 5. Create PMPM growth flag
    # --------------------------------------------------------
    """
    Growth classification:
        HIGH_INCREASE     : PMPM growth >= 10%
        MODERATE_INCREASE : 0% <= growth < 10%
        STABLE            : -5% <= growth < 0%
        DECREASE          : growth < -5%
        UNKNOWN           : Missing/unreliable prediction
    """

    def classify_growth(row):
        growth = row["PMPM_Growth"]

        if pd.isna(growth) or row["Plausible_Prediction"] is False:
            return "UNKNOWN"
        if growth >= 10:
            return "HIGH_INCREASE"
        elif growth >= 0:
            return "MODERATE_INCREASE"
        elif growth >= -5:
            return "STABLE"
        else:
            return "DECREASE"

    handoff["PMPM_Growth_Flag"] = handoff.apply(classify_growth, axis=1)

    # --------------------------------------------------------
    # 6. Select final handoff columns
    # --------------------------------------------------------
    handoff = handoff[
        [
            "Gnrc_Name",
            "Brnd_Name",
            "PMPM_2023",
            "Predicted_2024_PMPM",
            "PMPM_Growth",
            "PMPM_Growth_Flag",
            "Plausible_Prediction",
        ]
    ].copy()

    # --------------------------------------------------------
    # 7. Sort by highest predicted PMPM
    # --------------------------------------------------------
    handoff = handoff.sort_values(
        "Predicted_2024_PMPM", ascending=False, na_position="last"
    )

    # --------------------------------------------------------
    # 8. Save
    # --------------------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    handoff.to_csv(OUTPUT_PATH, index=False)

    # --------------------------------------------------------
    # 9. Summary
    # --------------------------------------------------------
    print("\nPMPM handoff created successfully.")
    print(f"\nOutput:\n{OUTPUT_PATH}")
    print(f"\nShape: {handoff.shape[0]:,} rows × {handoff.shape[1]} columns")

    print("\nPMPM Growth Flag distribution:")
    print(handoff["PMPM_Growth_Flag"].value_counts(dropna=False))

    print("\nPreview:")
    print(handoff.head(10).to_string(index=False))

    print("\nDone.")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()