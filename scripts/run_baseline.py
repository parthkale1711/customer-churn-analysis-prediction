import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import joblib
from src.churn_prediction.train import train_from_df


def main():
    processed_path = Path('data/processed/train_processed.csv')
    if not processed_path.exists():
        # fallback to original processed file
        processed_path = Path('data/processed/train.csv')
    if not processed_path.exists():
        print('No processed data found at data/processed/train_processed.csv or data/processed/train.csv')
        return

    df = pd.read_csv(processed_path)
    clf, acc = train_from_df(df, target_col='churn')
    out = Path('reports')
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, out / 'model.joblib')
    print(f'Baseline trained — accuracy: {acc:.4f}')
    print('Model saved to reports/model.joblib')


if __name__ == '__main__':
    main()
