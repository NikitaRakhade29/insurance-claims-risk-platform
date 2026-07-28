
import os
import pickle
import sys

import pandas as pd
import numpy as np
import psycopg2

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from features.feature_engineering import (
    load_raw_claims, compute_flags, write_flags, get_feature_matrix,
    compute_risk_index, PG_CONN_STR,
)

FRAUD_MODEL_PATH = os.path.join(os.path.dirname(__file__), "saved", "fraud_classifier.pkl")
ANOMALY_MODEL_PATH = os.path.join(os.path.dirname(__file__), "saved", "anomaly_detector.pkl")


def main():
    df = load_raw_claims()
    print(f"Scoring {len(df)} claims")

    # 1. explainable flags -> claim_features table
    flags = compute_flags(df)
    write_flags(flags)

    # 2. fraud classifier
    with open(FRAUD_MODEL_PATH, "rb") as f:
        fraud_model = pickle.load(f)
    X, claim_ids = get_feature_matrix(df, fit=False)
    fraud_probability = fraud_model.predict_proba(X)[:, 1]

    # 3. anomaly detector
    with open(ANOMALY_MODEL_PATH, "rb") as f:
        anomaly_bundle = pickle.load(f)
    anomaly_model = anomaly_bundle["model"]
    raw_scores = anomaly_model.decision_function(X)
    anomaly_score = (anomaly_bundle["score_max"] - raw_scores) / (
        anomaly_bundle["score_max"] - anomaly_bundle["score_min"]
    )

    # 4. transparent risk index, from the flags we just computed
    risk_index = flags.apply(compute_risk_index, axis=1)

    # 5. Percentile-based triage thresholds, not fixed cutoffs.
    #    Fixed thresholds (e.g. "0.7 = RED") don't account for the
    #    actual shape of the score distribution and can flag an
    #    unusable fraction of claims. Instead: RED = top ~8% combined
    #    risk, YELLOW = next ~12% -- a realistic, actionable triage
    #    load for a human adjuster, whatever the raw score values are.
    combined_score = pd.Series(np.maximum(fraud_probability, anomaly_score))
    red_cutoff = combined_score.quantile(0.92)
    yellow_cutoff = combined_score.quantile(0.80)

    def assign_triage(score):
        if score >= red_cutoff:
            return "RED"
        elif score >= yellow_cutoff:
            return "YELLOW"
        return "GREEN"

    triage_flag = combined_score.apply(assign_triage)
    print(f"Triage thresholds: RED >= {red_cutoff:.3f}, YELLOW >= {yellow_cutoff:.3f}")
    print(f"Distribution: RED={sum(triage_flag=='RED')}, YELLOW={sum(triage_flag=='YELLOW')}, GREEN={sum(triage_flag=='GREEN')}")

    results = pd.DataFrame({
        "claim_id": claim_ids,
        "fraud_probability": fraud_probability,
        "anomaly_score": anomaly_score,
        "customer_risk_index": risk_index.values,
        "triage_flag": triage_flag.values,
    })

    conn = psycopg2.connect(PG_CONN_STR)
    cur = conn.cursor()
    for _, row in results.iterrows():
        cur.execute(
            """
            INSERT INTO claims.model_scores
                (claim_id, fraud_probability, anomaly_score, customer_risk_index, triage_flag)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (claim_id) DO UPDATE SET
                fraud_probability = EXCLUDED.fraud_probability,
                anomaly_score = EXCLUDED.anomaly_score,
                customer_risk_index = EXCLUDED.customer_risk_index,
                triage_flag = EXCLUDED.triage_flag,
                scored_at = now()
            """,
            (row.claim_id, float(row.fraud_probability), float(row.anomaly_score),
             float(row.customer_risk_index), row.triage_flag),
        )
    conn.commit()
    cur.close()
    conn.close()

    print(f"Wrote {len(results)} scores.")


if __name__ == "__main__":
    main()