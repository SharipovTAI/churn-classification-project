# Churn Classification Project

A small end-to-end machine learning project for predicting customer churn.

The goal of the project was not only to train a classifier, but also to build a correct ML workflow:

- train/test separation;
- preprocessing with `Pipeline`;
- cross-validation;
- model comparison;
- hyperparameter tuning;
- evaluation under class imbalance;
- threshold selection;
- model serialization;
- inference on new customers.

---

## Dataset

The dataset contains 5,000 synthetic customer records.

### Features

- `age`
- `monthly_spend`
- `days_since_last_order`
- `support_tickets`
- `tenure_months`

### Target

- `churn = 0` — customer stays
- `churn = 1` — customer leaves

### Class distribution

```text
Class 0: 4701 (94.02%)
Class 1:  299 (5.98%)

The dataset is strongly imbalanced, so accuracy alone is not a useful primary metric.

Project Structure
churn_classification_project/
│
├── data/
│   └── churn.csv
│
├── models/
│   └── churn_model.joblib
│
├── src/
│   ├── train.py
│   └── inference.py
│
├── .gitignore
├── requirements.txt
└── README.md

The trained .joblib model can be excluded from Git because it can be recreated by running train.py.

ML Workflow
1. Train/Test Split

The dataset was divided into:

80% training data
20% test data

A stratified split was used to preserve the churn class ratio.

The test set was kept separate from model selection and hyperparameter tuning.

2. Preprocessing Pipeline

Logistic Regression uses standardized numerical features.

The preprocessing and model were combined into a single Scikit-Learn Pipeline:

Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])

Using a pipeline prevents data leakage during cross-validation because StandardScaler is fitted only on the training part of each fold.

Model Comparison

Three baseline models were compared using 5-fold Stratified Cross-Validation:

Model	ROC-AUC	Average Precision
Logistic Regression	0.7231	0.1616
Decision Tree	0.5366	0.0693
Random Forest	0.6359	0.1100

Logistic Regression performed best.

The synthetic target was generated using a logistic-style probability function, so a strong result from Logistic Regression was expected.

Overfitting Analysis

Train and validation metrics were compared.

Logistic Regression
Train ROC-AUC:      0.7293
Validation ROC-AUC: 0.7231
Gap:                0.0062

The small gap indicates good generalization.

Decision Tree
Train ROC-AUC:      1.0000
Validation ROC-AUC: 0.5366
Gap:                0.4634

The unrestricted Decision Tree strongly overfit the training data.

Random Forest
Train ROC-AUC:      1.0000
Validation ROC-AUC: 0.6359
Gap:                0.3641

Random Forest generalized better than one Decision Tree, but still showed significant overfitting.

Hyperparameter Tuning

GridSearchCV was used for hyperparameter optimization.

The main selection metric was Average Precision, because the positive class represents only about 6% of the dataset.

Best Logistic Regression
C = 0.01

CV ROC-AUC = 0.7235
CV AP      = 0.1645
Best Decision Tree
max_depth = 4
min_samples_leaf = 1
min_samples_split = 2

CV ROC-AUC = 0.6692
CV AP      = 0.1139
Best Random Forest
max_depth = 5
max_features = sqrt
min_samples_leaf = 1

CV ROC-AUC = 0.6980
CV AP      = 0.1338

Logistic Regression remained the best model after tuning.

Threshold Selection

Using the default classification threshold of 0.5 resulted in:

TN = 940
FP = 0
FN = 60
TP = 0

The model predicted every test customer as class 0.

This happened because the maximum predicted churn probability was only about:

0.254

However, ranking metrics were still useful:

Test ROC-AUC = 0.7402
Test AP      = 0.1679

This demonstrated an important distinction:

Good ranking quality does not guarantee that the default threshold of 0.5 is appropriate.

OOF Threshold Search

Out-of-Fold predictions from the training set were used to choose a threshold without optimizing it directly on the test set.

Several thresholds were compared:

Threshold	Precision	Recall	F1
0.03	0.0650	0.9623	0.1218
0.05	0.0929	0.8117	0.1667
0.07	0.1355	0.6067	0.2215
0.10	0.1654	0.2803	0.2081
0.15	0.1739	0.0669	0.0967
0.20	0.1923	0.0209	0.0377

Because this is an educational project without a real business cost function, the threshold with the highest F1 score was selected:

threshold = 0.07

In a real churn system, the threshold should instead be selected using business costs and constraints.

Final Test Results

Using the selected threshold:

Test ROC-AUC = 0.7402
Test AP      = 0.1679
Threshold    = 0.07

Confusion matrix:

[[698 242]
 [ 26  34]]

For the churn class:

Precision = 0.1232
Recall    = 0.5667
F1-score  = 0.2024

The model detected:

34 / 60 = 56.7%

of the churn customers in the test set.

The churn rate in the full test population was approximately 6%, while among customers predicted as high-risk it was approximately 12.3%.

So the model concentrated churn customers roughly twice as strongly as random selection.

Model Serialization

The final artifact contains:

{
    "model": best_model,
    "threshold": best_threshold,
    "features": feature_names
}

It is saved using joblib.

The complete Scikit-Learn pipeline is stored, including:

StandardScaler
      ↓
LogisticRegression

Therefore preprocessing does not have to be manually repeated during inference.

Inference

Example customer:

client = {
    "age": 42,
    "monthly_spend": 85,
    "days_since_last_order": 73,
    "support_tickets": 4,
    "tenure_months": 8,
}

Example output:

Churn probability: 0.2121
Threshold:         0.0700
Prediction:        1
Result: CHURN RISK
Installation

Create and activate a virtual environment.

Windows:

python -m venv .venv
.venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt
Training

Run:

python src/train.py

The script:

loads the dataset;
creates Train/Test split;
performs cross-validation;
tunes Logistic Regression;
generates OOF predictions;
selects the classification threshold;
evaluates the final model;
saves the trained model.
Inference

After training:

python src/inference.py

The script loads the saved model and performs prediction for a new customer without retraining.

Main Lessons

This project demonstrates several important ML concepts:

accuracy can be misleading for imbalanced classification;
ROC-AUC and Average Precision measure different aspects of model quality;
preprocessing should be performed inside a pipeline;
cross-validation should be used for model selection;
train/validation gaps can reveal overfitting;
a more complex model is not necessarily a better model;
hyperparameters and classification thresholds solve different problems;
the default threshold of 0.5 is not always appropriate;
the test set should be reserved for final evaluation;
the complete preprocessing + model pipeline should be saved for inference.
Tech Stack
Python
pandas
NumPy
Scikit-Learn
joblib
Git

Я бы ещё сделал один маленький Git-шаг: теперь README — отдельное осмысленное изменение, поэтому после сохранения:

```bash
git status
git add README.md
git commit -m "Add project documentation"
git push