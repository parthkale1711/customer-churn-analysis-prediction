from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import pandas as pd


def train_from_df(df, target_col='churn'):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    return clf, acc


if __name__ == '__main__':
    # placeholder: expects a CSV path in DATA_CSV env or default data/processed/train.csv
    import os
    path = os.environ.get('DATA_CSV', 'data/processed/train.csv')
    if not os.path.exists(path):
        print('Training data not found at', path)
    else:
        df = pd.read_csv(path)
        clf, acc = train_from_df(df)
        print('Accuracy:', acc)
        joblib.dump(clf, 'reports/model.joblib')
