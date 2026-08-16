import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    classification_report,
)


def load_processed(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def evaluate(df: pd.DataFrame, target_col: str = 'churn') -> dict:
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # split: 60% train, 20% val, 20% test
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)

    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate on validation and test
    y_val_pred = clf.predict(X_val)
    y_test_pred = clf.predict(X_test)

    y_test_proba = None
    try:
        y_test_proba = clf.predict_proba(X_test)[:, 1]
    except Exception:
        y_test_proba = None

    metrics = {}
    for split, y_true, y_pred in [('val', y_val, y_val_pred), ('test', y_test, y_test_pred)]:
        m = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, zero_division=0)),
            'f1': float(f1_score(y_true, y_pred, zero_division=0)),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
            'classification_report': classification_report(y_true, y_pred, output_dict=True),
        }
        metrics[split] = m

    if y_test_proba is not None:
        try:
            metrics['test']['roc_auc'] = float(roc_auc_score(y_test, y_test_proba))
        except Exception:
            metrics['test']['roc_auc'] = None

    return clf, metrics, (X_test, y_test, y_test_proba)


def save_metrics(metrics: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)


def plot_roc(y_test, y_proba, out_png: Path) -> None:
    if y_proba is None:
        print('No probability scores available; skipping ROC plot')
        return
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, label='ROC')
    plt.plot([0,1], [0,1], '--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png)
    plt.close()


def main():
    processed = Path('data/processed/train_processed.csv')
    if not processed.exists():
        processed = Path('data/processed/train.csv')
    if not processed.exists():
        print('No processed dataset found. Run preprocessing first.')
        return

    df = load_processed(processed)
    clf, metrics, (X_test, y_test, y_test_proba) = evaluate(df, target_col='churn')

    reports = Path('reports')
    reports.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, reports / 'model_evaluated.joblib')
    save_metrics(metrics, reports / 'metrics.json')
    plot_roc(y_test, y_test_proba, reports / 'roc_curve.png')

    print('Evaluation complete. Metrics saved to reports/metrics.json')
    print('Model saved to reports/model_evaluated.joblib')


if __name__ == '__main__':
    main()
