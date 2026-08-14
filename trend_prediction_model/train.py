"""
Model Training — Drug Utilization Trend Prediction
------------------------------------------------------
INPUT:  processed_data/cms_partd_model_ready.csv
        (132 drugs x 29 columns: identifiers + 23 features + 3 targets)

WHAT THIS SCRIPT DOES
------------------------
1. Loads the model-ready data, separates identifiers / features / targets
2. Splits into train/test (80/20) — but because 132 rows is small, ALSO
   runs 5-fold cross-validation, since a single train/test split leaves
   only ~26 test rows, too few to fully trust on their own
3. Trains a baseline Linear Regression model
4. Trains an XGBoost Regression model with regularization settings chosen
   specifically because this dataset is small relative to its feature
   count (see comments in build_xgboost_model())
5. Evaluates both with RMSE, MAE, R² — printed as a comparison table
6. Shows XGBoost feature importance (which features actually mattered)
7. Plots actual vs. predicted values for the test set
8. Saves a results CSV and the plot image

NOTE ON MULTIPLE TARGETS
----------------------------
We have 3 targets: TARGET_Tot_Clms_2024, TARGET_Tot_Drug_Cst_2024,
TARGET_Tot_Benes_2024. We train a SEPARATE model per target rather than
one multi-output model. This is deliberate: claims, cost, and beneficiaries
behave differently (e.g. cost per claim can rise even while claim volume
falls), so letting each target have its own model / its own tuned
hyperparameters gives more accurate predictions than forcing one model to
learn all three simultaneously. This is a standard, explainable choice —
worth stating plainly if asked in a viva.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import os

INPUT_PATH = "final/clean.csv"
OUTPUT_DIR = "model_outputs"

IDENTIFIER_COLUMNS = ["Gnrc_Name", "Brnd_Name", "Gnrc_Name_Code"]
TARGET_COLUMNS = [
    "TARGET_Tot_Clms_2024",
    "TARGET_Tot_Drug_Cst_2024",
    "TARGET_Tot_Benes_2024",
]

RANDOM_STATE = 42   # fixed seed so results are reproducible run-to-run
TEST_SIZE = 0.2     # 80/20 split
N_FOLDS = 5          # cross-validation folds


def build_xgboost_model() -> XGBRegressor:
    """
    XGBoost configured for the FULL dataset (1,451 rows, 23 features).

    UPDATED from the original small-dataset settings (132 rows). With
    ~11x more data now, the earlier heavy regularization (max_depth=3,
    min_child_weight=5, subsample=0.8, colsample_bytree=0.6) was overly
    conservative — it was shrinking predictions toward the "average" drug's
    behavior, which systematically understated continued growth for the
    highest-cost, fastest-growing drugs (e.g. Dulaglutide, Apixaban) —
    exactly the drugs a PBM cares about most. This showed up clearly once
    PMPM tracking was built: nearly every high-PMPM drug's predicted 2024
    value came in BELOW its actual 2023 value, despite a clear multi-year
    upward trend.

    Loosened settings below give the model more capacity to fit real
    patterns now that there's enough data to support it, while still
    keeping some regularization (not fully unconstrained) as a safety
    margin. Re-verify the train/test R2 gap after this change to confirm
    it hasn't reintroduced overfitting.

    - max_depth=5          : deeper trees -> more capacity (was 3)
    - n_estimators=200      : more trees, more data to justify it (was 100)
    - min_child_weight=2    : allows finer splits (was 5)
    - subsample=0.9         : each tree sees more of the data (was 0.8)
    - colsample_bytree=0.8  : each tree sees more features (was 0.6)
    - reg_alpha=0.3         : lighter L1 penalty (was 1.0)
    - reg_lambda=1.0        : lighter L2 penalty (was 2.0)
    """
    return XGBRegressor(
        n_estimators=200,
        max_depth=5,
        min_child_weight=2,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_alpha=0.3,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
    )


def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """Compute RMSE, MAE, R² for a fitted model on the test set."""
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    return {"model": model_name, "RMSE": rmse, "MAE": mae, "R2": r2, "predictions": preds}


def run_cross_validation(model, X, y, model_name: str) -> dict:
    """
    5-fold cross-validation — rotates which 20% of rows is used as the
    test set five times and averages the result. More reliable than a
    single train/test split when the test set would otherwise be tiny
    (~26 rows here).
    """
    kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    neg_mse_scores = cross_val_score(model, X, y, cv=kfold, scoring="neg_mean_squared_error")
    rmse_scores = np.sqrt(-neg_mse_scores)
    r2_scores = cross_val_score(model, X, y, cv=kfold, scoring="r2")
    return {
        "model": model_name,
        "cv_rmse_mean": rmse_scores.mean(),
        "cv_rmse_std": rmse_scores.std(),
        "cv_r2_mean": r2_scores.mean(),
        "cv_r2_std": r2_scores.std(),
    }


def train_and_evaluate_for_target(df: pd.DataFrame, feature_cols: list, target_col: str, log_transform: bool = False):
    """
    Full pipeline for ONE target column: split, train baseline + XGBoost,
    evaluate both, cross-validate both, plot actual vs predicted.

    log_transform: if True, the model is trained on log(1 + y) instead of
        raw y, and predictions are converted back with expm1() before
        evaluation/plotting. This is used for TARGET_Tot_Drug_Cst_2024
        specifically, because drug cost is heavily right-skewed — a
        handful of very expensive specialty drugs (e.g. Apixaban at
        ~$19.9B vs a median around $95M) can otherwise dominate training
        and even produce impossible NEGATIVE cost predictions for
        everything else. Log-transforming compresses that scale so the
        model isn't overwhelmed by a few extreme values, and guarantees
        predictions convert back to positive numbers only.
    """
    print(f"\n{'='*70}")
    print(f"TARGET: {target_col}" + (" (log-transformed)" if log_transform else ""))
    print(f"{'='*70}")

    X = df[feature_cols]
    y_raw = df[target_col]
    y = np.log1p(y_raw) if log_transform else y_raw

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"Train rows: {len(X_train)}  |  Test rows: {len(X_test)}")

    # --- Baseline: Linear Regression ---
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_results = evaluate_model(lr_model, X_test, y_test, "Linear Regression")
    lr_train_r2 = r2_score(y_train, lr_model.predict(X_train))

    # --- Main model: XGBoost ---
    xgb_model = build_xgboost_model()
    xgb_model.fit(X_train, y_train)
    xgb_results = evaluate_model(xgb_model, X_test, y_test, "XGBoost")
    xgb_train_r2 = r2_score(y_train, xgb_model.predict(X_train))

    # --- If log-transformed, convert predictions AND y_test back to the
    # original scale for reporting/plotting, so RMSE/MAE are in real
    # dollars (or claims/benes), not log-units which aren't interpretable ---
    if log_transform:
        y_test_original = np.expm1(y_test)
        lr_results["predictions"] = np.expm1(lr_results["predictions"])
        xgb_results["predictions"] = np.expm1(xgb_results["predictions"])
        # Recompute metrics on the original scale
        for r in (lr_results, xgb_results):
            r["RMSE"] = np.sqrt(mean_squared_error(y_test_original, r["predictions"]))
            r["MAE"] = mean_absolute_error(y_test_original, r["predictions"])
            r["R2"] = r2_score(y_test_original, r["predictions"])
        y_test_for_plot = y_test_original
    else:
        y_test_for_plot = y_test

    # --- Print single-split comparison ---
    print(f"\n--- Single train/test split results (original scale) ---")
    print(f"{'Model':<20} {'RMSE':>18} {'MAE':>18} {'R2':>8}")
    for r in [lr_results, xgb_results]:
        print(f"{r['model']:<20} {r['RMSE']:>18,.2f} {r['MAE']:>18,.2f} {r['R2']:>8.3f}")

    # --- Explicit overfitting check: TRAIN R2 vs TEST R2 ---
    # A large gap (e.g. train >> test) is the actual signature of overfitting.
    # A small gap (train and test close together) means the model generalizes,
    # regardless of how high the absolute R2 value is.
    print(f"\n--- Overfitting check: Train R2 vs Test R2 (large gap = overfitting) ---")
    print(f"{'Model':<20} {'Train R2':>12} {'Test R2':>12} {'Gap':>10}")
    print(f"{'Linear Regression':<20} {lr_train_r2:>12.4f} {lr_results['R2']:>12.4f} {lr_train_r2 - lr_results['R2']:>10.4f}")
    print(f"{'XGBoost':<20} {xgb_train_r2:>12.4f} {xgb_results['R2']:>12.4f} {xgb_train_r2 - xgb_results['R2']:>10.4f}")

    # --- Cross-validation (more reliable given small test set) ---
    # NOTE: CV is run on the (possibly log-transformed) y directly, so
    # CV R2 reflects fit quality on the scale the model actually trained on.
    lr_cv = run_cross_validation(LinearRegression(), X, y, "Linear Regression")
    xgb_cv = run_cross_validation(build_xgboost_model(), X, y, "XGBoost")

    print(f"\n--- 5-fold cross-validation results (more reliable on this small dataset) ---")
    print(f"{'Model':<20} {'CV RMSE (mean ± std)':>30} {'CV R2 (mean ± std)':>25}")
    for r in [lr_cv, xgb_cv]:
        rmse_str = f"{r['cv_rmse_mean']:,.2f} ± {r['cv_rmse_std']:,.2f}"
        r2_str = f"{r['cv_r2_mean']:.3f} ± {r['cv_r2_std']:.3f}"
        print(f"{r['model']:<20} {rmse_str:>30} {r2_str:>25}")

    # --- Feature importance from XGBoost ---
    importance = pd.Series(xgb_model.feature_importances_, index=feature_cols)
    importance = importance.sort_values(ascending=False)
    print(f"\n--- Top 10 most important features (XGBoost) ---")
    print(importance.head(10).to_string())

    # --- Plot: actual vs predicted using the BEST model for this target
    # (not always XGBoost) — so the plot matches what you'd actually
    # recommend/use, not just whichever model happened to be trained last.
    best_is_xgb = xgb_results["R2"] >= lr_results["R2"]
    best_model_name = "XGBoost" if best_is_xgb else "Linear Regression"
    best_predictions = xgb_results["predictions"] if best_is_xgb else lr_results["predictions"]

    plt.figure(figsize=(7, 6))
    plt.scatter(y_test_for_plot, best_predictions, alpha=0.7, edgecolor="k")
    min_val = min(y_test_for_plot.min(), best_predictions.min())
    max_val = max(y_test_for_plot.max(), best_predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect prediction")
    plt.xlabel(f"Actual {target_col}")
    plt.ylabel(f"Predicted {target_col}")
    plt.title(f"Actual vs Predicted — {target_col} ({best_model_name}, test set)")
    plt.legend()
    plt.tight_layout()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_path = os.path.join(OUTPUT_DIR, f"actual_vs_predicted_{target_col}.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\nSaved plot: {plot_path}")

    return {
        "target": target_col,
        "lr_single_split": lr_results,
        "xgb_single_split": xgb_results,
        "lr_cv": lr_cv,
        "xgb_cv": xgb_cv,
        "feature_importance": importance,
    }


def main():
    print(f"Loading {INPUT_PATH} ...")
    df = pd.read_csv(INPUT_PATH)
    print(f"  Loaded {df.shape[0]} rows, {df.shape[1]} columns")

    feature_cols = [c for c in df.columns if c not in IDENTIFIER_COLUMNS and c not in TARGET_COLUMNS]
    print(f"  Using {len(feature_cols)} features: {feature_cols}")

    all_results = []
    for target_col in TARGET_COLUMNS:
        # Cost is heavily right-skewed (see docstring above) -> log-transform it.
        # Claims and beneficiaries are far less skewed, so they're trained as-is.
        use_log = (target_col == "TARGET_Tot_Drug_Cst_2024")
        result = train_and_evaluate_for_target(df, feature_cols, target_col, log_transform=use_log)
        all_results.append(result)

    # --- Summary table across all 3 targets ---
    print(f"\n\n{'='*70}")
    print("SUMMARY — XGBoost vs Linear Regression across all targets")
    print(f"{'='*70}")
    summary_rows = []
    for r in all_results:
        summary_rows.append({
            "Target": r["target"],
            "LR Test RMSE": r["lr_single_split"]["RMSE"],
            "XGB Test RMSE": r["xgb_single_split"]["RMSE"],
            "LR CV R2": r["lr_cv"]["cv_r2_mean"],
            "XGB CV R2": r["xgb_cv"]["cv_r2_mean"],
        })
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary_path = os.path.join(OUTPUT_DIR, "model_comparison_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved summary: {summary_path}")


if __name__ == "__main__":
    main()