import joblib
import pandas as pd


def predict_from_df(df, model_path='reports/model.joblib'):
    model = joblib.load(model_path)
    return model.predict(df)
