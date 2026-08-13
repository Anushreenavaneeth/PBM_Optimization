# ============================================================
# PHARMACY BENEFIT MANAGEMENT (PBM) OPTIMIZATION
# ORANGE BOOK - MACHINE LEARNING
#
# Objective:
# Train an ML model using FDA Orange Book product information
# to predict the TE_Code classification.
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from xgboost import XGBClassifier


warnings.filterwarnings("ignore")


# ============================================================
# 2. FILE PATH
# ============================================================

DATA_PATH = "Preprocess/Preprocessed ORANGEBOOK/products.csv"

OUTPUT_DIR = "orangebook_models"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 3. LOAD ORANGE BOOK DATA
# ============================================================

print("\n" + "=" * 70)
print("LOADING ORANGE BOOK DATA")
print("=" * 70)

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"\nFile not found:\n{DATA_PATH}\n"
        "Make sure you are running this file from "
        "the PBM_OPTIMIZATION-MAIN folder."
    )

df = pd.read_csv(DATA_PATH, low_memory=False)

print("\nDataset loaded successfully.")

print("Rows    :", df.shape[0])
print("Columns :", df.shape[1])


# ============================================================
# 4. DISPLAY COLUMNS
# ============================================================

print("\n" + "=" * 70)
print("DATASET COLUMNS")
print("=" * 70)

for i, column in enumerate(df.columns, start=1):
    print(f"{i}. {column}")


# ============================================================
# 5. DISPLAY FIRST ROWS
# ============================================================

print("\n" + "=" * 70)
print("FIRST 5 ROWS")
print("=" * 70)

print(df.head())


# ============================================================
# 6. CHECK MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

missing_values = df.isnull().sum()

print(
    missing_values[
        missing_values > 0
    ].sort_values(ascending=False)
)


# ============================================================
# 7. CHECK TE_CODE
# ============================================================

TARGET = "TE_Code"

if TARGET not in df.columns:
    raise ValueError(
        "TE_Code column is not available in the dataset."
    )


print("\n" + "=" * 70)
print("TE CODE DISTRIBUTION")
print("=" * 70)

print(
    df[TARGET].value_counts(dropna=False)
)


# ============================================================
# 8. CLEAN TARGET COLUMN
# ============================================================

print("\nCleaning TE_Code...")

# Remove rows where target is missing
df = df.dropna(
    subset=[TARGET]
).copy()

# Convert target to string
df[TARGET] = (
    df[TARGET]
    .astype(str)
    .str.strip()
    .str.upper()
)

# Remove empty values
df = df[
    (df[TARGET] != "") &
    (df[TARGET] != "NAN")
].copy()


print("\nDataset after target cleaning:")
print(df.shape)


# ============================================================
# 9. TARGET DISTRIBUTION AFTER CLEANING
# ============================================================

print("\n" + "=" * 70)
print("CLEANED TE CODE DISTRIBUTION")
print("=" * 70)

te_counts = df[TARGET].value_counts()

print(te_counts)


# ============================================================
# 10. HANDLE VERY RARE CLASSES
# ============================================================

# A stratified train/test split requires at least 2 samples
# in each class.
#
# We retain classes with >= 2 samples.

MIN_SAMPLES = 2

valid_classes = te_counts[
    te_counts >= MIN_SAMPLES
].index

df = df[
    df[TARGET].isin(valid_classes)
].copy()


print("\nClasses used for training:")

print(
    df[TARGET].value_counts()
)


# ============================================================
# 11. SELECT FEATURES
# ============================================================

# These are the useful Orange Book product characteristics.
#
# We intentionally DO NOT use:
#
# TE_Code       -> TARGET
# Appl_No       -> identifier
# Product_No    -> identifier
#
# We also avoid Trade_Name and Applicant initially because
# they have very high cardinality and can cause the model
# to memorize individual products/manufacturers.

candidate_features = [
    "Ingredient",
    "DF;Route",
    "Strength",
    "Appl_Type",
    "RLD",
    "RS",
    "Type",
    "Dosage_Form",
    "Route_Of_Administration",
    "Approved_Prior_To_1982"
]


# Keep only columns that actually exist
features = [
    col
    for col in candidate_features
    if col in df.columns
]


print("\n" + "=" * 70)
print("FEATURES SELECTED")
print("=" * 70)

for feature in features:
    print("-", feature)


# ============================================================
# 12. CREATE X AND y
# ============================================================

X = df[features].copy()

y = df[TARGET].copy()


print("\nFeature matrix shape:")
print(X.shape)

print("\nTarget shape:")
print(y.shape)


# ============================================================
# 13. IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
# ============================================================

numeric_features = X.select_dtypes(
    include=[
        "int64",
        "int32",
        "float64",
        "float32",
        "bool"
    ]
).columns.tolist()


categorical_features = X.select_dtypes(
    include=[
        "object",
        "category"
    ]
).columns.tolist()


print("\n" + "=" * 70)
print("FEATURE TYPES")
print("=" * 70)

print("\nNumerical features:")
print(numeric_features)

print("\nCategorical features:")
print(categorical_features)


# ============================================================
# 14. PREPROCESSING PIPELINE
# ============================================================

# Numerical:
# Missing values -> median
#
# Categorical:
# Missing values -> most frequent
# Then One-Hot Encoding

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        )
    ]
)


categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),

        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),

        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# ============================================================
# 15. ENCODE TE_CODE
# ============================================================

# XGBoost requires numerical class labels.

classes = sorted(
    y.unique()
)


class_to_id = {
    class_name: index
    for index, class_name in enumerate(classes)
}


id_to_class = {
    index: class_name
    for class_name, index in class_to_id.items()
}


y_encoded = y.map(
    class_to_id
).astype(int)


print("\n" + "=" * 70)
print("TARGET ENCODING")
print("=" * 70)

for class_name, class_id in class_to_id.items():
    print(
        f"{class_name} -> {class_id}"
    )


# ============================================================
# 16. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)


print("\n" + "=" * 70)
print("TRAIN / TEST DATA")
print("=" * 70)

print(
    "Training samples:",
    len(X_train)
)

print(
    "Testing samples :",
    len(X_test)
)


# ============================================================
# 17. RANDOM FOREST MODEL
# ============================================================

print("\n" + "=" * 70)
print("TRAINING RANDOM FOREST")
print("=" * 70)


rf_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),

        (
            "classifier",
            RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced"
            )
        )
    ]
)


rf_model.fit(
    X_train,
    y_train
)


rf_predictions = rf_model.predict(
    X_test
)


print("Random Forest training completed.")


# ============================================================
# 18. XGBOOST MODEL
# ============================================================

print("\n" + "=" * 70)
print("TRAINING XGBOOST")
print("=" * 70)


xgb_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),

        (
            "classifier",
            XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="multi:softprob",
                eval_metric="mlogloss",
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)


xgb_model.fit(
    X_train,
    y_train
)


xgb_predictions = xgb_model.predict(
    X_test
)


print("XGBoost training completed.")


# ============================================================
# 19. EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    model_name,
    y_true,
    predictions
):

    accuracy = accuracy_score(
        y_true,
        predictions
    )

    precision = precision_score(
        y_true,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        average="weighted",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_true,
        predictions,
        average="weighted",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_true,
        predictions,
        average="macro",
        zero_division=0
    )

    print("\n" + "=" * 70)
    print(model_name)
    print("=" * 70)

    print(
        f"Accuracy       : {accuracy:.4f}"
    )

    print(
        f"Precision      : {precision:.4f}"
    )

    print(
        f"Recall         : {recall:.4f}"
    )

    print(
        f"Weighted F1    : {weighted_f1:.4f}"
    )

    print(
        f"Macro F1       : {macro_f1:.4f}"
    )

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Weighted_F1": weighted_f1,
        "Macro_F1": macro_f1
    }


# ============================================================
# 20. EVALUATE RANDOM FOREST
# ============================================================

rf_results = evaluate_model(
    "Random Forest",
    y_test,
    rf_predictions
)


# ============================================================
# 21. EVALUATE XGBOOST
# ============================================================

xgb_results = evaluate_model(
    "XGBoost",
    y_test,
    xgb_predictions
)


# ============================================================
# 22. COMPARE MODELS
# ============================================================

results = pd.DataFrame(
    [
        rf_results,
        xgb_results
    ]
)


print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(results)


# Save comparison
results.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "model_comparison.csv"
    ),
    index=False
)


# ============================================================
# 23. SELECT BEST MODEL
# ============================================================

best_index = results[
    "Macro_F1"
].idxmax()


best_model_name = results.loc[
    best_index,
    "Model"
]


if best_model_name == "XGBoost":

    best_model = xgb_model
    best_predictions = xgb_predictions

else:

    best_model = rf_model
    best_predictions = rf_predictions


print("\n" + "=" * 70)

print(
    "BEST MODEL:",
    best_model_name
)

print("=" * 70)


# ============================================================
# 24. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)


target_names = [
    id_to_class[i]
    for i in range(
        len(classes)
    )
]


print(
    classification_report(
        y_test,
        best_predictions,
        labels=list(
            range(len(classes))
        ),
        target_names=target_names,
        zero_division=0
    )
)


# ============================================================
# 25. CONFUSION MATRIX
# ============================================================

print("\nGenerating confusion matrix...")


cm = confusion_matrix(
    y_test,
    best_predictions,
    labels=list(
        range(len(classes))
    )
)


fig, ax = plt.subplots(
    figsize=(12, 10)
)


display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=target_names
)


display.plot(
    ax=ax,
    xticks_rotation=90,
    values_format="d"
)


plt.title(
    "Orange Book TE Code - Confusion Matrix"
)

plt.tight_layout()


confusion_path = os.path.join(
    OUTPUT_DIR,
    "confusion_matrix.png"
)


plt.savefig(
    confusion_path,
    dpi=300
)


plt.show()


# ============================================================
# 26. FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)


try:

    fitted_preprocessor = (
        best_model
        .named_steps["preprocessor"]
    )

    fitted_classifier = (
        best_model
        .named_steps["classifier"]
    )


    feature_names = (
        fitted_preprocessor
        .get_feature_names_out()
    )


    importances = (
        fitted_classifier
        .feature_importances_
    )


    feature_importance = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": importances
        }
    )


    feature_importance = (
        feature_importance
        .sort_values(
            "Importance",
            ascending=False
        )
    )


    print(
        feature_importance.head(20)
    )


    # Save feature importance
    feature_importance.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "feature_importance.csv"
        ),
        index=False
    )


    # Plot top 20
    top_features = (
        feature_importance
        .head(20)
        .sort_values(
            "Importance"
        )
    )


    plt.figure(
        figsize=(10, 8)
    )


    plt.barh(
        top_features["Feature"],
        top_features["Importance"]
    )


    plt.xlabel(
        "Importance"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        "Top 20 Orange Book Features"
    )


    plt.tight_layout()


    feature_plot_path = os.path.join(
        OUTPUT_DIR,
        "feature_importance.png"
    )


    plt.savefig(
        feature_plot_path,
        dpi=300
    )


    plt.show()


except Exception as error:

    print(
        "Could not generate feature importance."
    )

    print(error)


# ============================================================
# 27. SAVE TRAINED MODEL
# ============================================================

model_path = os.path.join(
    OUTPUT_DIR,
    "orangebook_te_model.pkl"
)


joblib.dump(
    best_model,
    model_path
)


# ============================================================
# 28. SAVE TARGET MAPPING
# ============================================================

mapping_path = os.path.join(
    OUTPUT_DIR,
    "te_code_mapping.pkl"
)


joblib.dump(
    {
        "class_to_id": class_to_id,
        "id_to_class": id_to_class
    },
    mapping_path
)


# ============================================================
# 29. SAVE CLEAN DATA
# ============================================================

clean_data_path = os.path.join(
    OUTPUT_DIR,
    "orangebook_cleaned.csv"
)


df.to_csv(
    clean_data_path,
    index=False
)


# ============================================================
# 30. FINAL SUMMARY
# ============================================================

print("\n\n")
print("=" * 70)
print("ORANGE BOOK ML TRAINING COMPLETED")
print("=" * 70)


print(
    "\nOriginal dataset rows:",
    pd.read_csv(
        DATA_PATH,
        low_memory=False
    ).shape[0]
)


print(
    "Training rows:",
    len(X_train)
)


print(
    "Testing rows:",
    len(X_test)
)


print(
    "\nBest model:",
    best_model_name
)


print(
    "\nOutput files:"
)


print(
    "1. Trained model:"
)

print(
    "   ",
    model_path
)


print(
    "2. TE Code mapping:"
)

print(
    "   ",
    mapping_path
)


print(
    "3. Model comparison:"
)

print(
    "   ",
    os.path.join(
        OUTPUT_DIR,
        "model_comparison.csv"
    )
)


print(
    "4. Cleaned dataset:"
)

print(
    "   ",
    clean_data_path
)


print(
    "5. Confusion matrix:"
)

print(
    "   ",
    confusion_path
)


print(
    "6. Feature importance:"
)

print(
    "   ",
    os.path.join(
        OUTPUT_DIR,
        "feature_importance.csv"
    )
)


print("\n" + "=" * 70)
print("DONE")
print("=" * 70)