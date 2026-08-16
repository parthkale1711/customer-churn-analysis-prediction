import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.pipeline import Pipeline as SklearnPipeline


def load_processed(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def build_preprocessor(X: pd.DataFrame):
    num_cols = X.select_dtypes(include=['number']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    num_transform = SklearnPipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    cat_transform = SklearnPipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer([('num', num_transform, num_cols), ('cat', cat_transform, cat_cols)])
    return preprocessor, num_cols, cat_cols


def evaluate_strategy(pipe, X_train, X_test, y_train, y_test):
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_proba = None
    try:
        y_proba = pipe.predict_proba(X_test)[:, 1]
    except Exception:
        y_proba = None

    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1': float(f1_score(y_test, y_pred, zero_division=0)),
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'roc_auc': float(roc_auc_score(y_test, y_proba)) if y_proba is not None else None
    }
    return metrics, pipe


def main():
    processed = Path('data/processed/train.csv')
    if not processed.exists():
        print('Processed CSV not found. Run download_and_train_telco first.')
        return

    df = load_processed(processed)
    if 'churn' not in df.columns:
        print('Target `churn` not found in processed CSV')
        return

    X = df.drop(columns=['churn'])
    y = df['churn'].map({1:1, 0:0, 'Yes':1, 'No':0}).astype(int)

    preprocessor, num_cols, cat_cols = build_preprocessor(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    results = {}

    # Strategy A: no handling
    pipe_a = SklearnPipeline([('pre', preprocessor), ('clf', RandomForestClassifier(n_estimators=200, random_state=42))])
    metrics_a, fitted_a = evaluate_strategy(pipe_a, X_train, X_test, y_train, y_test)
    results['no_handling'] = metrics_a
    joblib.dump(fitted_a, Path('reports') / 'model_no_handling.joblib')

    # Strategy B: class_weight='balanced'
    pipe_b = SklearnPipeline([('pre', preprocessor), ('clf', RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42))])
    metrics_b, fitted_b = evaluate_strategy(pipe_b, X_train, X_test, y_train, y_test)
    results['class_weight_balanced'] = metrics_b
    joblib.dump(fitted_b, Path('reports') / 'model_class_weight.joblib')

    # Strategy C: SMOTE oversampling
    pipe_c = ImbPipeline([('pre', preprocessor), ('smote', SMOTE(random_state=42)), ('clf', RandomForestClassifier(n_estimators=200, random_state=42))])
    metrics_c, fitted_c = evaluate_strategy(pipe_c, X_train, X_test, y_train, y_test)
    results['smote'] = metrics_c
    joblib.dump(fitted_c, Path('reports') / 'model_smote.joblib')

    Path('reports').mkdir(parents=True, exist_ok=True)
    with open(Path('reports') / 'imbalance_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print('Imbalance strategies evaluated. Results saved to reports/imbalance_results.json')


if __name__ == '__main__':
    main()
