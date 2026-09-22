import pandas as pd

from pathlib import Path

import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv(
    "churn_classification_project/data/churn.csv"
)

X = df.drop(columns="churn")
y = df["churn"]


# ============================================================
# 2. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42,
)


# ============================================================
# 3. CROSS-VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)


# ============================================================
# 4. MODEL PIPELINE
# ============================================================

pipeline = Pipeline([
    (
        "scaler",
        StandardScaler(),
    ),
    (
        "model",
        LogisticRegression(
            max_iter=1000,
        ),
    ),
])


# ============================================================
# 5. HYPERPARAMETER TUNING
# ============================================================

param_grid = {
    "model__C": [
        0.01,
        0.1,
        1,
        10,
        100,
    ]
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring={
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
    },
    refit="average_precision",
    cv=cv,
    n_jobs=-1,
)

grid_search.fit(
    X_train,
    y_train,
)

best_model = grid_search.best_estimator_
best_index = grid_search.best_index_

cv_roc_auc = grid_search.cv_results_[
    "mean_test_roc_auc"
][best_index]

cv_ap = grid_search.cv_results_[
    "mean_test_average_precision"
][best_index]


print("\n=== BEST MODEL ===")

print(
    "Best params:",
    grid_search.best_params_,
)

print(
    f"CV ROC-AUC: {cv_roc_auc:.4f}"
)

print(
    f"CV AP:      {cv_ap:.4f}"
)


# ============================================================
# 6. OOF PREDICTIONS
#    Used for threshold selection without using Test
# ============================================================

oof_proba = cross_val_predict(
    best_model,
    X_train,
    y_train,
    cv=cv,
    method="predict_proba",
    n_jobs=-1,
)[:, 1]


# ============================================================
# 7. THRESHOLD SELECTION
# ============================================================

thresholds = [
    0.03,
    0.05,
    0.07,
    0.10,
    0.15,
    0.20,
]

threshold_results = []

for threshold in thresholds:

    y_oof_pred = (
        oof_proba >= threshold
    ).astype(int)

    precision = precision_score(
        y_train,
        y_oof_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_train,
        y_oof_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_train,
        y_oof_pred,
        zero_division=0,
    )

    predicted_positive = (
        y_oof_pred.sum()
    )

    threshold_results.append({
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predicted_positive": predicted_positive,
    })


threshold_df = pd.DataFrame(
    threshold_results
)


print("\n=== THRESHOLD COMPARISON ===")

print(
    threshold_df.to_string(
        index=False
    )
)


# Select threshold with maximum F1

best_row = threshold_df.loc[
    threshold_df["f1"].idxmax()
]

best_threshold = float(
    best_row["threshold"]
)


print("\n=== SELECTED THRESHOLD ===")

print(
    f"Threshold: {best_threshold:.2f}"
)

print(
    f"OOF Precision: "
    f"{best_row['precision']:.4f}"
)

print(
    f"OOF Recall:    "
    f"{best_row['recall']:.4f}"
)

print(
    f"OOF F1:        "
    f"{best_row['f1']:.4f}"
)


# ============================================================
# 8. FINAL TEST EVALUATION
# ============================================================

y_test_proba = best_model.predict_proba(
    X_test
)[:, 1]


test_roc_auc = roc_auc_score(
    y_test,
    y_test_proba,
)

test_ap = average_precision_score(
    y_test,
    y_test_proba,
)


# Apply selected threshold

y_test_pred = (
    y_test_proba >= best_threshold
).astype(int)


print("\n=== FINAL TEST RESULTS ===")

print(
    f"Test ROC-AUC: {test_roc_auc:.4f}"
)

print(
    f"Test AP:      {test_ap:.4f}"
)

print(
    f"Threshold:    {best_threshold:.2f}"
)


print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_test_pred,
    )
)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_test_pred,
        digits=4,
        zero_division=0,
    )
)

# ============================================================
# 9. SAVE MODEL
# ============================================================

models_dir = Path(
    "churn_classification_project/models"
)

models_dir.mkdir(
    parents=True,
    exist_ok=True,
)

model_artifact = {
    "model": best_model,
    "threshold": best_threshold,
    "features": X.columns.tolist(),
}

model_path = models_dir / "churn_model.joblib"

joblib.dump(
    model_artifact,
    model_path,
)

print(
    f"\nModel saved to: {model_path}"
)