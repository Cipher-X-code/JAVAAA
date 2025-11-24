"""Train a simple XGBoost multiclass model and save it to `predictor/model.joblib`.
"""
import joblib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from prepare_data import build_features

OUT = Path(__file__).parent / "model.joblib"


def train():
    df = build_features()
    X = df.drop(columns=["label", "date", "HomeTeam", "AwayTeam", "odds_home", "odds_draw", "odds_away"])
    y = df["label"]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        use_label_encoder=False,
        eval_metric="mlogloss",
        n_estimators=100,
        max_depth=4,
        random_state=42,
    )
    model.fit(X_train, y_train)

    joblib.dump(model, OUT)
    print(f"Saved model to {OUT}")


if __name__ == "__main__":
    train()
