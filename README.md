# Credit Default Prediction – ML Assignment 2 (BITS WILP)

This repository features six machine learning classification models built on the “Default of Credit Card Clients” dataset, along with an interactive Streamlit application for model evaluation and visualization.

- Live App: https://mlassignment2-yzc6kwff2uxvxjvvzv79uw.streamlit.app/
- Dataset: https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset
- Typical file: `UCI_Credit_Card.csv`
- Target column: `default.payment.next.month` (binary 0/1)

---

## Problem Statement
Forecast whether a credit card customer is likely to default on their payment in the upcoming month using demographic information, credit history, and billing and repayment data.

## Dataset Description
- Source: UCI/Kaggle (see link above)
- Instances: ~30,000
- Features: ≥ 23 (includes demographic, credit, bill amounts, and payment amounts)
- Target: `default.payment.next.month` (0 = no default, 1 = default)

## Models Used
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (kNN)
4. Naive Bayes (Gaussian)
5. Random Forest (Ensemble)
6. XGBoost (Ensemble)

## Comparison Table (fill after evaluation)

| ML Model Name       | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------------|----------|-----|-----------|--------|----|-----|
| Logistic Regression | 0.8078 | 0.7076 | 0.6883 | 0.2396 | 0.3555 | 0.3251 |
| Decision Tree       | 0.7152 | 0.6079 | 0.3704 | 0.4115 | 0.3899 | 0.2052 |
| kNN                 | 0.7928 | 0.7013 | 0.5487 | 0.3564 | 0.4322 | 0.3233 |
| Naive Bayes         | 0.7525 | 0.7249 | 0.4515 | 0.5539 | 0.4975 | 0.3386 |
| Random Forest       | 0.8133 | 0.7540 | 0.6356 | 0.3655 | 0.4641 | 0.3812 |
| XGBoost             | 0.8183 | 0.7747 | 0.6604 | 0.3677 | 0.4724 | 0.3966 |

## Observations (fill after evaluation)

| ML Model Name       | Observation about model performance |
|---------------------|-------------------------------------|
| Logistic Regression | High precision but low recall, misses many defaults |
| Decision Tree       | Moderate accuracy with balanced metrics, overfits possibly |
| kNN                 | Decent performance, sensitive to feature scaling |
| Naive Bayes         | High recall but low precision, many false positives |
| Random Forest       | Strong ensemble performance with good AUC |
| XGBoost             | Best overall with highest accuracy and AUC |

## Project Structure
```
project-root/
├─ app.py                          # Streamlit entrypoint
├─ requirements.txt
├─ README.md
├─ PLAN.md
├─ model/                          # persisted models + metrics
│  ├─ preprocessor.joblib
│  ├─ logistic_regression.joblib
│  ├─ decision_tree.joblib
│  ├─ knn.joblib
│  ├─ naive_bayes.joblib
│  ├─ random_forest.joblib
│  ├─ xgboost.joblib
│  └─ metrics_summary.csv
├─ src/
│  ├─ __init__.py
│  ├─ data.py                      # load/validate data
│  ├─ preprocess.py                # ColumnTransformer pipeline
│  ├─ models.py                    # model builders
│  ├─ evaluate.py                  # metrics utilities
│  └─ train.py                     # train/evaluate/save
└─ notebooks/                      # optional for EDA
```

## How to Run Locally
1) Create and activate a virtual environment
```
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```
2) Install dependencies
```
pip install -r requirements.txt
```
3) (Offline) Train models and save artifacts
```
python -m src.train --data /path/to/UCI_Credit_Card.csv --target default.payment.next.month
```
4) Run the Streamlit app
```
streamlit run app.py
```

## Deployment (Streamlit Community Cloud)
- Ensure `app.py` is at repo root and `requirements.txt` includes all dependencies.
- On Streamlit Cloud: New App → select repo → branch → `app.py` → Deploy.

## Notes
- The Streamlit app is inference-only. Upload test data with the same schema as training data.
- Unseen categorical values are handled by `OneHotEncoder(handle_unknown='ignore')`.
- For this dataset, most features are numeric; some integer-coded fields may represent categories.
