
import os
import pickle
import sys

import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from features.feature_engineering import load_raw_claims, get_feature_matrix

MODEL_PATH = os.path.join(os.path.dirname(__file__), "saved", "fraud_classifier.pkl")


def main():
    df = load_raw_claims()
    print(f"Loaded {len(df)} claims")

    X, claim_ids = get_feature_matrix(df, fit=True)  # fits + saves the shared encoder
    y = df["fraud_reported"].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    fraud_ratio = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    print(f"Class imbalance ratio (negative:positive) = {fraud_ratio:.1f}:1")

    model = XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        scale_pos_weight=fraud_ratio,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n--- Fraud classifier evaluation ---")
    print(classification_report(y_test, y_pred, target_names=["not_fraud", "fraud"]))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.3f}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"\nSaved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
