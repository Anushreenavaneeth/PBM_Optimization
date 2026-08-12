"""
Feature Engineering for Drug Utilization Trend Prediction (v3 — TRIMMED)
---------------------------------------------------------------------------
INPUT:  processed_data/cms_partd_combined_clean.csv
        (long format: one row per drug PER YEAR — 132 drugs x 6 years = 792 rows)

OUTPUT: processed_data/cms_partd_model_ready.csv
        (wide format: one row per drug, trimmed feature set)

WHY THIS VERSION EXISTS (vs v2)
-----------------------------------
v2 produced 40 training features on only 132 rows (~3 rows per feature) —
a real overfitting risk for XGBoost on a dataset this small. v3 trims the
feature set while keeping the two genuinely useful additions from the
earlier review (identifier separation + utilization ratios):

  - DROPPED: raw Tot_Clms/Tot_Drug_Cst/Tot_Benes for 2019 and 2020
             (6 columns removed) — these are the years furthest from the
             2024 target and least likely to carry unique predictive signal
             once more recent years are included.
  - ADDED BACK (compact form): a single "5-year overall growth" feature
             per metric (2019 -> 2023), so the long-term trend isn't lost
             entirely — just compressed into 2 columns instead of 6 raw ones.
  - KEPT: raw values for 2021, 2022, 2023 (closest to the target year,
             most relevant for a lag-feature model).
  - KEPT: short-term year-over-year growth (2021->2022, 2022->2023).
  - TRIMMED: utilization ratios (claims_per_bene, cost_per_claim,
             cost_per_bene) now computed for 2022 and 2023 ONLY, not all
             5 feature years — these are the most recent, most relevant.

Result: ~23 training features on 132 rows (~5.7 rows per feature),
a meaningfully safer ratio than v2's ~3.3 rows per feature.

DOWNSTREAM NOTE (Therapeutic Equivalence handoff)
------------------------------------------------------
This output is one row per drug (Gnrc_Name), which is exactly the join
key the Therapeutic Equivalence teammate's output will need to match
against for the platform integration. Gnrc_Name and Brnd_Name are kept
as plain, clean identifier columns (not encoded, not dropped) specifically
so this file can be joined/merged downstream without extra cleanup.
"""

import pandas as pd
import numpy as np
import os

INPUT_PATH = "processed_data/cms_final.csv"
OUTPUT_PATH = "final/clean.csv"

ALL_YEARS = [2019, 2020, 2021, 2022, 2023]   # all available feature years
RAW_FEATURE_YEARS = [2021, 2022, 2023]        # years kept as raw columns
RATIO_YEARS = [2022, 2023]                    # years kept for ratio features
LONG_TERM_SPAN = (2019, 2023)                 # start/end for compact long-term growth
TARGET_YEAR = 2024

METRICS_TO_PIVOT = ["Tot_Clms", "Tot_Drug_Cst", "Tot_Benes"]

IDENTIFIER_COLUMNS = ["Gnrc_Name", "Brnd_Name", "Gnrc_Name_Code"]


def build_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot long format -> wide format for ALL years first (we need 2019/2020
    briefly to compute the long-term growth feature), then drop the raw
    2019/2020 columns afterward in main().
    """
    drug_info = df[["Gnrc_Name", "Brnd_Name"]].drop_duplicates(subset="Gnrc_Name")

    pivoted = df.pivot(index="Gnrc_Name", columns="year", values=METRICS_TO_PIVOT)
    pivoted.columns = [f"{metric}_{year}" for metric, year in pivoted.columns]
    pivoted = pivoted.reset_index()

    result = pivoted.merge(drug_info, on="Gnrc_Name", how="left")

    rename_map = {f"{m}_{TARGET_YEAR}": f"TARGET_{m}_{TARGET_YEAR}" for m in METRICS_TO_PIVOT}
    result = result.rename(columns=rename_map)

    return result


def add_short_term_growth(df: pd.DataFrame) -> pd.DataFrame:
    """Year-over-year growth between consecutive RAW_FEATURE_YEARS only (2021->22, 2022->23)."""
    df = df.copy()
    growth_cols = {"clms": [], "cst": []}

    for y1, y2 in zip(RAW_FEATURE_YEARS[:-1], RAW_FEATURE_YEARS[1:]):
        clms_col = f"clms_growth_{str(y1)[2:]}_{str(y2)[2:]}"
        cst_col = f"cst_growth_{str(y1)[2:]}_{str(y2)[2:]}"

        df[clms_col] = (df[f"Tot_Clms_{y2}"] - df[f"Tot_Clms_{y1}"]) / df[f"Tot_Clms_{y1}"]
        df[cst_col] = (df[f"Tot_Drug_Cst_{y2}"] - df[f"Tot_Drug_Cst_{y1}"]) / df[f"Tot_Drug_Cst_{y1}"]

        growth_cols["clms"].append(clms_col)
        growth_cols["cst"].append(cst_col)

    df["avg_clms_growth"] = df[growth_cols["clms"]].mean(axis=1)
    df["avg_cst_growth"] = df[growth_cols["cst"]].mean(axis=1)

    return df


def add_long_term_growth(df: pd.DataFrame) -> pd.DataFrame:
    """
    Single compact feature summarizing the FULL 2019->2023 trend, computed
    using the 2019/2020 raw columns BEFORE they get dropped. This keeps the
    long-term signal without keeping 6 extra raw columns.
    """
    df = df.copy()
    y1, y2 = LONG_TERM_SPAN

    df["clms_growth_5yr"] = (df[f"Tot_Clms_{y2}"] - df[f"Tot_Clms_{y1}"]) / df[f"Tot_Clms_{y1}"]
    df["cst_growth_5yr"] = (df[f"Tot_Drug_Cst_{y2}"] - df[f"Tot_Drug_Cst_{y1}"]) / df[f"Tot_Drug_Cst_{y1}"]

    return df


def add_utilization_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Ratio features (claims/bene, cost/claim, cost/bene) for RATIO_YEARS only (2022, 2023)."""
    df = df.copy()
    for year in RATIO_YEARS:
        clms_col, cst_col, bene_col = f"Tot_Clms_{year}", f"Tot_Drug_Cst_{year}", f"Tot_Benes_{year}"
        df[f"claims_per_bene_{year}"] = df[clms_col] / df[bene_col]
        df[f"cost_per_claim_{year}"] = df[cst_col] / df[clms_col]
        df[f"cost_per_bene_{year}"] = df[cst_col] / df[bene_col]
    return df


def drop_old_raw_years(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the raw 2019/2020 columns now that long-term growth has been computed from them."""
    years_to_drop = [y for y in ALL_YEARS if y not in RAW_FEATURE_YEARS]
    cols_to_drop = [f"{m}_{y}" for m in METRICS_TO_PIVOT for y in years_to_drop]
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    return df.drop(columns=cols_to_drop)


def encode_drug_name(df: pd.DataFrame) -> pd.DataFrame:
    """Identifier-only numeric code — excluded from training features (see get_feature_columns)."""
    df = df.copy()
    df["Gnrc_Name_Code"] = df["Gnrc_Name"].astype("category").cat.codes
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    exclude = set(IDENTIFIER_COLUMNS)
    return [c for c in df.columns if c not in exclude and not c.startswith("TARGET_")]


def main():
    print(f"Loading {INPUT_PATH} ...")
    df = pd.read_csv(INPUT_PATH)
    print(f"  Loaded {df.shape[0]} rows (long format)")

    print("\nPivoting all years into columns (temporarily, to compute long-term growth) ...")
    wide = build_lag_features(df)
    print(f"  Reshaped to {wide.shape[0]} rows")

    print("\nComputing long-term growth (2019 -> 2023) before dropping old raw years ...")
    wide = add_long_term_growth(wide)

    print("\nDropping raw 2019/2020 columns (kept only as the long-term growth feature above) ...")
    wide = drop_old_raw_years(wide)

    print("\nAdding short-term growth (2021->22, 2022->23) ...")
    wide = add_short_term_growth(wide)

    print("\nAdding utilization ratios for 2022, 2023 only ...")
    wide = add_utilization_ratios(wide)

    print("\nEncoding Gnrc_Name -> Gnrc_Name_Code (identifier only, NOT a training feature) ...")
    wide = encode_drug_name(wide)

    target_cols = [c for c in wide.columns if c.startswith("TARGET_")]
    feature_cols = get_feature_columns(wide)
    wide = wide[IDENTIFIER_COLUMNS + feature_cols + target_cols]

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    wide.to_csv(OUTPUT_PATH, index=False)

    print(f"\nDone. Saved model-ready dataset to {OUTPUT_PATH}")
    print(f"Final shape: {wide.shape[0]} rows x {wide.shape[1]} columns")
    print(f"Training features: {len(feature_cols)}  |  Rows-per-feature ratio: {wide.shape[0]/len(feature_cols):.1f} : 1")
    print(f"\nIdentifier columns (NOT used for training): {IDENTIFIER_COLUMNS}")
    print(f"\nFeature columns used for training:")
    for c in feature_cols:
        print(f"  - {c}")
    print(f"\nTarget columns: {target_cols}")


if __name__ == "__main__":
    main()