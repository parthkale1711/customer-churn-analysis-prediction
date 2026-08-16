import sys
import pandas as pd
from pathlib import Path

# Ensure project root is on sys.path so `src` is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.churn_prediction.data import load_and_process


def main():
    Path('data/processed').mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        'customer_id': [1, 2, 3, 4],
        'age': [25, 34, 45, 52],
        'monthly_charges': [29.99, 49.95, 70.1, 15.0],
        'contract': ['month-to-month', 'one-year', 'two-year', 'month-to-month'],
        'churn': [0, 1, 0, 1]
    })
    src = 'data/processed/train.csv'
    df.to_csv(src, index=False)
    X, y, processed = load_and_process(src, target_col='churn', save_processed_path='data/processed/train_processed.csv')
    print('X shape:', X.shape)
    print('y shape:', y.shape)
    print('processed columns:', processed.columns.tolist())
    print('processed head:\n', processed.head().to_dict())


if __name__ == '__main__':
    main()
