import sys
from pathlib import Path
import urllib.request

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.churn_prediction.data import load_and_process
from src.churn_prediction.train import train_from_df
import joblib
import pandas as pd


TELCO_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
)


def download(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {out_path}")
    urllib.request.urlretrieve(url, out_path)


def normalize_and_save(raw_path: Path, processed_path: Path) -> None:
    df = pd.read_csv(raw_path)
    # normalize column names: lower, replace spaces and punctuation with underscore
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace(r"[^0-9a-zA-Z_]+", "", regex=True)
    )
    # normalize churn values to 0/1 if present
    if 'churn' in df.columns:
        df['churn'] = df['churn'].map({'Yes': 1, 'No': 0}).fillna(df['churn'])
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path, index=False)
    print(f"Saved normalized CSV to {processed_path}")


def main():
    raw = Path('data/raw/telco.csv')
    processed = Path('data/processed/train.csv')
    download(TELCO_URL, raw)
    normalize_and_save(raw, processed)

    # process with pipeline and save processed copy
    X, y, processed_df = load_and_process(str(processed), target_col='churn', save_processed_path=str(Path('data/processed/train_processed.csv')))

    # combine X and y for train_from_df convenience
    df_for_train = processed_df.copy()

    clf, acc = train_from_df(df_for_train, target_col='churn')
    out = Path('reports')
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, out / 'model_telco.joblib')
    print(f'Trained Telco baseline — accuracy: {acc:.4f}')
    print('Model saved to reports/model_telco.joblib')


if __name__ == '__main__':
    main()
