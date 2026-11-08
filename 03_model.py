"""
CareLoop Clinics – No-Show Guard
PART 3: MACHINE LEARNING (Tasks 3.1–3.5)

Input:
    appointments_clean.csv

Outputs:
    features_task_3_1.csv
    task_3_2_predictions.csv
    task_3_4_model_comparison.csv
    test_risk_scores.csv

Install dependencies if needed:
    pip install pandas numpy scikit-learn

Run:
    python part3_machine_learning.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score, cross_val_predict
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------
INPUT_FILE = "appointments_clean.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.20

NUMERIC_FEATURES = [
    "age", "distance_km", "fee_inr", "lead_days",
    "prior_no_shows", "reminder_sent"
]
CATEGORICAL_FEATURES = ["booking_channel", "visit_type"]
TARGET = "no_show"


def make_preprocessor():
    """Build preprocessing that is fitted only on the training data."""
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES)
    ])


def safe_roc_auc(y_true, probabilities):
    """Return ROC-AUC, or NaN if the target has only one class."""
    try:
        return roc_auc_score(y_true, probabilities)
    except ValueError:
        return np.nan


def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    """Fit model and return train/test metrics."""
    model.fit(X_train, y_train)

    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
    test_probabilities = model.predict_proba(X_test)[:, 1]

    return {
        "model": name,
        "train_accuracy": accuracy_score(y_train, train_predictions),
        "test_accuracy": accuracy_score(y_test, test_predictions),
        "test_precision": precision_score(y_test, test_predictions, zero_division=0),
        "test_recall": recall_score(y_test, test_predictions, zero_division=0),
        "test_f1": f1_score(y_test, test_predictions, zero_division=0),
        "test_roc_auc": safe_roc_auc(y_test, test_probabilities),
        "model_object": model
    }


# ============================================================
# TASK 3.1 – LOAD DATA, SELECT FEATURES, SPLIT, BASELINE
# ============================================================
print("\n" + "=" * 65)
print("TASK 3.1 – DATA PREPARATION AND BASELINE")
print("=" * 65)

df = pd.read_csv(INPUT_FILE)

required_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET, "appointment_id"]
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise ValueError(
        f"Missing required columns in {INPUT_FILE}: {missing_columns}"
    )

# Convert target and numeric fields safely.
df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
for col in NUMERIC_FEATURES:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=[TARGET]).copy()
df[TARGET] = df[TARGET].astype(int)

if not set(df[TARGET].unique()).issubset({0, 1}):
    raise ValueError("The no_show target must contain only 0 and 1.")

# Keep appointment IDs separate; do not use them as model features.
appointment_ids = df["appointment_id"].astype(str)
X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
y = df[TARGET].copy()

# Export the prepared modelling dataset for inspection.
features_export = X.copy()
features_export.insert(0, "appointment_id", appointment_ids.values)
features_export[TARGET] = y.values
features_export.to_csv("features_task_3_1.csv", index=False)

X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
    X, y, appointment_ids,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

baseline_predictions = np.zeros(len(y_test), dtype=int)
baseline_accuracy = accuracy_score(y_test, baseline_predictions)

print(f"Total usable appointments: {len(df)}")
print(f"Training rows: {len(X_train)}")
print(f"Testing rows: {len(X_test)}")
print(f"No-shows in full dataset: {int(y.sum())}")
print(f"No-shows in test set: {int(y_test.sum())}")
print(f"Baseline accuracy (predict every appointment as attended): {baseline_accuracy:.4f}")


# ============================================================
# TASK 3.2 – LOGISTIC REGRESSION
# ============================================================
print("\n" + "=" * 65)
print("TASK 3.2 – LOGISTIC REGRESSION")
print("=" * 65)

logistic_model = Pipeline([
    ("preprocessor", make_preprocessor()),
    ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced"))
])

logistic_model.fit(X_train, y_train)
lr_predictions = logistic_model.predict(X_test)
lr_probabilities = logistic_model.predict_proba(X_test)[:, 1]

print("Accuracy :", round(accuracy_score(y_test, lr_predictions), 4))
print("Precision:", round(precision_score(y_test, lr_predictions, zero_division=0), 4))
print("Recall   :", round(recall_score(y_test, lr_predictions, zero_division=0), 4))
print("F1-score :", round(f1_score(y_test, lr_predictions, zero_division=0), 4))
print("ROC-AUC  :", round(safe_roc_auc(y_test, lr_probabilities), 4))
print("\nConfusion matrix:")
print(confusion_matrix(y_test, lr_predictions))
print("\nClassification report:")
print(classification_report(y_test, lr_predictions, zero_division=0))

task_3_2_output = pd.DataFrame({
    "appointment_id": ids_test.values,
    "actual_no_show": y_test.values,
    "predicted_no_show": lr_predictions,
    "risk_probability": lr_probabilities
})
task_3_2_output.to_csv("task_3_2_predictions.csv", index=False)
print("Saved: task_3_2_predictions.csv")


# ============================================================
# TASK 3.3 – DECISION TREE: UNRESTRICTED VS DEPTH 3
# ============================================================
print("\n" + "=" * 65)
print("TASK 3.3 – DECISION TREE COMPARISON")
print("=" * 65)

tree_models = {
    "Decision Tree – unrestricted": Pipeline([
        ("preprocessor", make_preprocessor()),
        ("classifier", DecisionTreeClassifier(random_state=RANDOM_STATE))
    ]),
    "Decision Tree – max_depth=3": Pipeline([
        ("preprocessor", make_preprocessor()),
        ("classifier", DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE))
    ])
}

for name, model in tree_models.items():
    model.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    print(f"\n{name}")
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Test accuracy : {test_acc:.4f}")
    print(f"Train-test gap: {train_acc - test_acc:.4f}")


# ============================================================
# TASK 3.4 – COMPARE LOGISTIC, DECISION TREE, RANDOM FOREST
# ============================================================
print("\n" + "=" * 65)
print("TASK 3.4 – MODEL COMPARISON")
print("=" * 65)

depth3_tree = Pipeline([
    ("preprocessor", make_preprocessor()),
    ("classifier", DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE))
])

random_forest = Pipeline([
    ("preprocessor", make_preprocessor()),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1
    ))
])

comparison_models = [
    ("Logistic Regression", logistic_model),
    ("Decision Tree (depth 3)", depth3_tree),
    ("Random Forest (100 trees)", random_forest)
]

comparison_rows = []
fitted_models = {}

for name, model in comparison_models:
    metrics = evaluate_model(name, model, X_train, y_train, X_test, y_test)
    fitted_models[name] = metrics.pop("model_object")
    comparison_rows.append(metrics)

comparison_df = pd.DataFrame(comparison_rows)
comparison_df.to_csv("task_3_4_model_comparison.csv", index=False)
print(comparison_df.to_string(index=False))
print("Saved: task_3_4_model_comparison.csv")


# ============================================================
# TASK 3.5 – CROSS-VALIDATION, RANDOM FOREST TUNING, THRESHOLD
# ============================================================
print("\n" + "=" * 65)
print("TASK 3.5 – HYPERPARAMETER TUNING AND FINAL RISK SCORES")
print("=" * 65)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# Cross-validation scores for Logistic Regression and depth-3 Tree.
lr_cv_scores = cross_val_score(
    logistic_model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1
)
tree_cv_scores = cross_val_score(
    depth3_tree, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1
)

# Tune Random Forest using 5-fold CV.
rf_pipeline = Pipeline([
    ("preprocessor", make_preprocessor()),
    ("classifier", RandomForestClassifier(
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1
    ))
])

param_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [3, 5, 8]
}

grid_search = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1,
    refit=True
)
grid_search.fit(X_train, y_train)

print("Logistic Regression mean CV ROC-AUC:", round(lr_cv_scores.mean(), 4))
print("Decision Tree mean CV ROC-AUC      :", round(tree_cv_scores.mean(), 4))
print("Tuned Random Forest best CV ROC-AUC :", round(grid_search.best_score_, 4))
print("Best Random Forest parameters      :", grid_search.best_params_)

cv_results = pd.DataFrame([
    {"model": "Logistic Regression", "mean_cv_roc_auc": lr_cv_scores.mean()},
    {"model": "Decision Tree (depth 3)", "mean_cv_roc_auc": tree_cv_scores.mean()},
    {"model": "Tuned Random Forest", "mean_cv_roc_auc": grid_search.best_score_}
]).sort_values("mean_cv_roc_auc", ascending=False)

print("\nCross-validation comparison:")
print(cv_results.to_string(index=False))

# Select the model with the highest mean CV ROC-AUC.
winner_name = cv_results.iloc[0]["model"]
if winner_name == "Logistic Regression":
    winner = logistic_model
elif winner_name == "Decision Tree (depth 3)":
    winner = depth3_tree
else:
    winner = grid_search.best_estimator_

print(f"\nSelected model: {winner_name}")

# Obtain out-of-fold probabilities on training data to choose a threshold
# without using the held-out test set.
oof_probabilities = cross_val_predict(
    winner, X_train, y_train,
    cv=cv,
    method="predict_proba",
    n_jobs=-1
)[:, 1]

threshold_results = []
for threshold in [0.35, 0.50]:
    oof_predictions = (oof_probabilities >= threshold).astype(int)
    threshold_results.append({
        "threshold": threshold,
        "f1": f1_score(y_train, oof_predictions, zero_division=0),
        "recall": recall_score(y_train, oof_predictions, zero_division=0)
    })

threshold_df = pd.DataFrame(threshold_results)
# Highest F1 wins; if tied, choose the higher recall.
threshold_df = threshold_df.sort_values(
    ["f1", "recall"], ascending=[False, False]
)
chosen_threshold = float(threshold_df.iloc[0]["threshold"])

print("\nThreshold selection using training out-of-fold predictions:")
print(threshold_df.to_string(index=False))
print("Chosen threshold:", chosen_threshold)

# Fit final selected model on the complete training set.
winner.fit(X_train, y_train)
test_risk_probabilities = winner.predict_proba(X_test)[:, 1]
test_flags = (test_risk_probabilities >= chosen_threshold).astype(int)

print("\nFinal held-out test metrics:")
print("Accuracy :", round(accuracy_score(y_test, test_flags), 4))
print("Precision:", round(precision_score(y_test, test_flags, zero_division=0), 4))
print("Recall   :", round(recall_score(y_test, test_flags, zero_division=0), 4))
print("F1-score :", round(f1_score(y_test, test_flags, zero_division=0), 4))
print("ROC-AUC  :", round(safe_roc_auc(y_test, test_risk_probabilities), 4))
print("\nConfusion matrix:")
print(confusion_matrix(y_test, test_flags))

risk_scores = pd.DataFrame({
    "appointment_id": ids_test.values,
    "no_show": y_test.values,
    "risk_prob": test_risk_probabilities,
    "flag": test_flags
})
risk_scores.to_csv("test_risk_scores.csv", index=False)

print("\nSaved: test_risk_scores.csv")
print("\nTASKS 3.1–3.5 COMPLETED")
print("Generated files:")
print("- features_task_3_1.csv")
print("- task_3_2_predictions.csv")
print("- task_3_4_model_comparison.csv")
print("- test_risk_scores.csv")
print("\nNote: Keep the test set separate. The test metrics are final evaluation results.")
