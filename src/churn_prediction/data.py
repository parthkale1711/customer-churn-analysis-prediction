from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from sklearn.impute import SimpleImputer


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def save_csv(df: pd.DataFrame, path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates()


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if num_cols:
        num_imputer = SimpleImputer(strategy="median")
        df[num_cols] = num_imputer.fit_transform(df[num_cols])

    if cat_cols:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

    return df


def encode_categoricals(df: pd.DataFrame, drop_first: bool = True) -> pd.DataFrame:
    return pd.get_dummies(df, drop_first=drop_first)


def build_X_y(df: pd.DataFrame, target_col: str = "churn") -> Tuple[pd.DataFrame, pd.Series]:
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")
    y = df[target_col]
    X = df.drop(columns=[target_col])
    return X, y


def load_and_process(path: str, target_col: str = "churn", save_processed_path: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    df = load_csv(path)
    df = drop_duplicates(df)
    df = impute_missing(df)
    processed = encode_categoricals(df)
    X, y = build_X_y(processed, target_col=target_col)
    if save_processed_path:
        save_csv(processed, save_processed_path)
    return X, y, processed


if __name__ == "__main__":
    import os
    source = os.environ.get("DATA_CSV", "data/processed/train.csv")
    out = os.environ.get("PROCESSED_OUT", "data/processed/train_processed.csv")
    try:
        X, y, processed = load_and_process(source, target_col="churn", save_processed_path=out)
        print("Processed shape:", processed.shape)
    except Exception as exc:
        print("Failed to load/process data:", exc)
