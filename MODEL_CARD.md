# Model Card — Customer Churn Prediction (Production)

## Model
- Name: RandomForest (class_weight='balanced')
- Persisted artifact: `reports/model_production.joblib`
- Version: v1.0
- Date: 2026-08-16

## Overview
This model predicts customer churn (binary). It was trained on the Telco Customer Churn dataset (public IBM example) that was downloaded and preprocessed as part of this project.

## Intended use
- Predict whether a customer will churn (1) or not (0).
- For batch scoring and exploratory analysis. Not intended for direct real-time production without further validation.

## Metrics (test set)
- Accuracy: ~0.766
- Precision (positive class): ~0.55
- Recall (positive class): ~0.65
- F1 (positive class): ~0.597
- ROC AUC: ~0.827

Full evaluation artifacts: `reports/imbalance_results.json`, `reports/metrics.json`, `reports/grid_search_results.json`.

## Data
- Source: IBM Telco Customer Churn (public example)
- Processed files: `data/processed/train.csv`, `data/processed/train_processed.csv`

## Preprocessing
- Numeric features: median imputation + standard scaling
- Categorical features: most-frequent imputation + one-hot encoding
- Class imbalance handling: `class_weight='balanced'` used for production model

## Limitations and caveats
- Trained on a public dataset; performance may differ on your production data.
- Sensitive to feature schema; ensure the same columns and types when scoring.
- No formal fairness or production-grade validation performed.

## Reproducibility
Run the download, preprocessing, grid-search, and evaluation scripts in order:

- `scripts/download_and_train_telco.py` — download + baseline training
- `scripts/grid_search_pipeline.py` — reproducible pipeline + GridSearchCV
- `scripts/evaluate_model.py` — hold-out evaluation

Artifacts and reports are in the `reports/` folder.
