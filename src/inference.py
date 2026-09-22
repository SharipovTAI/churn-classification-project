from pathlib import Path

import joblib
import pandas as pd


# ============================================================
# 1. LOAD SAVED MODEL
# ============================================================

project_root = Path(__file__).resolve().parents[1]

model_path = (
    project_root
    / "models"
    / "churn_model.joblib"
)

artifact = joblib.load(model_path)

model = artifact["model"]
threshold = artifact["threshold"]
features = artifact["features"]


# ============================================================
# 2. NEW CLIENT
# ============================================================

client = {
    "age": 42,
    "monthly_spend": 85,
    "days_since_last_order": 73,
    "support_tickets": 4,
    "tenure_months": 8,
}


# ============================================================
# 3. PREPARE INPUT
# ============================================================

X_new = pd.DataFrame(
    [client],
    columns=features,
)


# ============================================================
# 4. PREDICT PROBABILITY
# ============================================================

churn_probability = model.predict_proba(
    X_new
)[0, 1]


# ============================================================
# 5. APPLY SAVED THRESHOLD
# ============================================================

prediction = int(
    churn_probability >= threshold
)


# ============================================================
# 6. OUTPUT
# ============================================================

print(
    f"Churn probability: "
    f"{churn_probability:.4f}"
)

print(
    f"Threshold:         "
    f"{threshold:.4f}"
)

print(
    f"Prediction:        "
    f"{prediction}"
)

if prediction == 1:
    print("Result: CHURN RISK")
else:
    print("Result: NO CHURN RISK")