
import argparse
import os
import sys

import pandas as pd
import psycopg2

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from llm_reasoning.reasoning_client import get_rationale
from features.feature_engineering import PG_CONN_STR


def fetch_claims_needing_rationale(batch_size: int) -> pd.DataFrame:
    conn = psycopg2.connect(PG_CONN_STR)
    query = """
        SELECT r.*, s.fraud_probability, s.anomaly_score, s.customer_risk_index
        FROM claims.raw_claims r
        JOIN claims.model_scores s ON r.claim_id = s.claim_id
        LEFT JOIN claims.llm_rationale l ON r.claim_id = l.claim_id
        WHERE l.claim_id IS NULL
          AND s.triage_flag IN ('RED', 'YELLOW')
        ORDER BY GREATEST(s.fraud_probability, s.anomaly_score) DESC
        LIMIT %s
    """
    df = pd.read_sql(query, conn, params=(batch_size,))
    conn.close()
    return df


def write_rationale(claim_id: str, result: dict) -> None:
    conn = psycopg2.connect(PG_CONN_STR)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO claims.llm_rationale
            (claim_id, discrepancy_detected, flagged_reasons, recommended_action, rationale_text)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (claim_id) DO UPDATE SET
            discrepancy_detected = EXCLUDED.discrepancy_detected,
            flagged_reasons = EXCLUDED.flagged_reasons,
            recommended_action = EXCLUDED.recommended_action,
            rationale_text = EXCLUDED.rationale_text,
            generated_at = now()
        """,
        (claim_id, result["discrepancy_detected"], result["flagged_reasons"],
         result["recommended_action"], result["rationale_text"]),
    )
    conn.commit()
    cur.close()
    conn.close()


def main(batch_size: int):
    claims = fetch_claims_needing_rationale(batch_size)
    print(f"Generating rationale for {len(claims)} high-risk claims")

    for i, row in claims.iterrows():
        try:
            result = get_rationale(row.to_dict())
            write_rationale(row.claim_id, result)
            print(f"  [{i+1}/{len(claims)}] {row.claim_id} -> {result['recommended_action']}")
        except Exception as e:
            print(f"  [{i+1}/{len(claims)}] {row.claim_id} FAILED after all retries: {e}")
            print(f"    Skipping -- re-run this script later to retry just this claim.")
            continue

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=50)
    args = parser.parse_args()
    main(args.batch_size)