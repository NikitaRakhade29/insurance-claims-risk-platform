
import os
import pickle

import duckdb
import pandas as pd
import psycopg2
from dotenv import load_dotenv
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

load_dotenv()  # reads .env in the project root automatically

PG_CONN_STR = os.environ.get(
    "DATABASE_URL", "postgresql://airflow:airflow@localhost:5432/airflow"
)
ENCODER_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "saved", "encoder.pkl")

CATEGORICAL_COLS = [
    "make", "accident_area", "sex", "marital_status", "fault", "policy_type",
    "vehicle_category", "vehicle_price_bucket", "days_policy_accident",
    "days_policy_claim", "past_number_of_claims", "age_of_vehicle",
    "age_of_policyholder", "police_report_filed", "witness_present",
    "agent_type", "number_of_suppliments", "address_change_claim",
    "number_of_cars", "base_policy",
]
NUMERIC_COLS = ["age", "deductible", "driver_rating"]


def load_raw_claims() -> pd.DataFrame:
    conn = psycopg2.connect(PG_CONN_STR)
    df = pd.read_sql("SELECT * FROM claims.raw_claims", conn)
    conn.close()
    return df


def compute_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Explainable rule-based flags computed with DuckDB SQL over the
    real columns -- these are what a human adjuster reads directly,
    separate from the black-box model scores."""
    con = duckdb.connect()
    con.register("raw", df)
    flags = con.execute("""
        SELECT
            claim_id,
            customer_id,
            CASE past_number_of_claims
                WHEN 'none' THEN 0
                WHEN '1' THEN 1
                WHEN '2 to 4' THEN 3
                WHEN 'more than 4' THEN 5
                ELSE 0
            END AS past_claims_numeric,
            (address_change_claim IN ('under 6 months', '1 year')) AS address_change_recent,
            (deductible >= 400) AS high_deductible_flag,
            (driver_rating <= 1) AS low_driver_rating_flag
        FROM raw
    """).fetchdf()
    con.close()
    return flags


def write_flags(flags: pd.DataFrame) -> None:
    conn = psycopg2.connect(PG_CONN_STR)
    cur = conn.cursor()
    for _, row in flags.iterrows():
        cur.execute(
            """
            INSERT INTO claims.claim_features
                (claim_id, customer_id, past_claims_numeric, address_change_recent,
                 high_deductible_flag, low_driver_rating_flag)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (claim_id) DO UPDATE SET
                past_claims_numeric = EXCLUDED.past_claims_numeric,
                address_change_recent = EXCLUDED.address_change_recent,
                high_deductible_flag = EXCLUDED.high_deductible_flag,
                low_driver_rating_flag = EXCLUDED.low_driver_rating_flag,
                computed_at = now()
            """,
            (
                row.claim_id, row.customer_id, int(row.past_claims_numeric),
                bool(row.address_change_recent), bool(row.high_deductible_flag),
                bool(row.low_driver_rating_flag),
            ),
        )
    conn.commit()
    cur.close()
    conn.close()


def get_feature_matrix(df: pd.DataFrame, fit: bool = False):
    """Returns (X, claim_ids). Fits + saves the encoder if fit=True or
    no saved encoder exists yet; otherwise loads and transforms only."""
    if fit or not os.path.exists(ENCODER_PATH):
        encoder = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
            ],
            remainder="passthrough",
        )
        X = encoder.fit_transform(df[CATEGORICAL_COLS + NUMERIC_COLS])
        os.makedirs(os.path.dirname(ENCODER_PATH), exist_ok=True)
        with open(ENCODER_PATH, "wb") as f:
            pickle.dump(encoder, f)
    else:
        with open(ENCODER_PATH, "rb") as f:
            encoder = pickle.load(f)
        X = encoder.transform(df[CATEGORICAL_COLS + NUMERIC_COLS])

    return X, df["claim_id"].values


def compute_risk_index(flags_row: pd.Series) -> float:
    """Transparent, explainable business-rule risk score -- deliberately
    NOT a black-box model, so it complements the fraud classifier rather
    than duplicating it. Weighted composite of real per-claim signals."""
    score = (
        0.40 * min(flags_row.past_claims_numeric / 5.0, 1.0)
        + 0.20 * float(flags_row.address_change_recent)
        + 0.20 * float(flags_row.high_deductible_flag)
        + 0.20 * float(flags_row.low_driver_rating_flag)
    )
    return round(min(score, 1.0), 4)
