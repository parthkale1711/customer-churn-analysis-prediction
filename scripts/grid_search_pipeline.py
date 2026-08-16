import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def build_pipeline(numeric_cols, categorical_cols):
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('clf', RandomForestClassifier(random_state=42))
    ])
    return pipe


def main():
    processed_csv = Path('data/processed/train.csv')
    if not processed_csv.exists():
        print('Processed CSV not found at data/processed/train.csv. Run download_and_train_telco first.')
        return

    df = load_data(processed_csv)
    if 'churn' not in df.columns:
        print('Target column `churn` not found in processed CSV')
        return

    X = df.drop(columns=['churn'])
    y = df['churn'].map({1:1, 0:0, 'Yes':1, 'No':0}).astype(int)

    numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    pipe = build_pipeline(numeric_cols, categorical_cols)

    param_grid = {
        'clf__n_estimators': [100, 200],
        'clf__max_depth': [None, 10, 20],
        'clf__class_weight': [None, 'balanced']
    }

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    gs = GridSearchCV(pipe, param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=1)
    gs.fit(X_train, y_train)

    best = gs.best_estimator_
    y_pred = best.predict(X_test)
    y_proba = None
    try:
        y_proba = best.predict_proba(X_test)[:, 1]
    except Exception:
        y_proba = None

    results = {
        'best_params': gs.best_params_,
        'test_accuracy': float(accuracy_score(y_test, y_pred)),
        'test_roc_auc': float(roc_auc_score(y_test, y_proba)) if y_proba is not None else None,
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }

    reports = Path('reports')
    reports.mkdir(parents=True, exist_ok=True)
    joblib.dump(best, reports / 'model_grid_best.joblib')
    with open(reports / 'grid_search_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print('Grid search complete. Best params:', gs.best_params_)
    print('Test accuracy:', results['test_accuracy'])
    print('Test ROC AUC:', results['test_roc_auc'])


if __name__ == '__main__':
    main()
