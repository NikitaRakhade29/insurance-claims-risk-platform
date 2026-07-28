
import os
import pickle
import sys

from sklearn.ensemble import IsolationForest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from features.feature_engineering import load_raw_claims, get_feature_matrix

MODEL_PATH = os.path.join(os.path.dirname(__file__), "saved", "anomaly_detector.pkl")


def main():
    df = load_raw_claims()
    X, claim_ids = get_feature_matrix(df, fit=False)  # reuse the encoder the classifier fit

    model = IsolationForest(
        n_estimators=200,
        contamination=0.06,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X)

    scores = model.decision_function(X)  # higher = more normal
    # flip and normalize to 0-1 so higher = more anomalous, matching fraud_probability's scale
    normalized = (scores.max() - scores) / (scores.max() - scores.min())

    flagged = (normalized >= 0.7).sum()
    print(f"Trained on {len(df)} claims, {flagged} flagged as anomalous (score >= 0.7)")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "score_min": scores.min(), "score_max": scores.max()}, f)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
