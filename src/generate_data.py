import numpy as np
import pandas as pd

# Этап 3. Данные
rng = np.random.default_rng(42)

n = 5000

age = rng.integers(18, 75, size=n)
monthly_spend = rng.normal(120, 40, size=n).clip(10, 400)
days_since_last_order = rng.exponential(30, size=n).clip(0, 300)
support_tickets = rng.poisson(1.5, size=n)
tenure_months = rng.integers(1, 120, size=n)

logit = (
    -2.5
    + 0.015 * (age - 40)
    + 0.012 * (days_since_last_order - 30)
    + 0.35 * support_tickets
    - 0.012 * tenure_months
    - 0.003 * monthly_spend
)

probability = 1 / (1 + np.exp(-logit))

churn = rng.binomial(1, probability)

df = pd.DataFrame({
    "age": age,
    "monthly_spend": monthly_spend,
    "days_since_last_order": days_since_last_order,
    "support_tickets": support_tickets,
    "tenure_months": tenure_months,
    "churn": churn
})

df.to_csv("churn_classification_project/data/churn.csv", index=False)

print(df.head())
print()
print(df["churn"].value_counts())
print()
print(df["churn"].value_counts(normalize=True))