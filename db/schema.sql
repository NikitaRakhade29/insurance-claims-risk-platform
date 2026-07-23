
CREATE SCHEMA IF NOT EXISTS claims;

-- Raw claim events, one row per event consumed from Kafka
CREATE TABLE IF NOT EXISTS claims.raw_claims (
    claim_id                TEXT PRIMARY KEY,       -- PolicyNumber
    customer_id             TEXT NOT NULL,          -- PolicyNumber (one claim per policy in this dataset)
    make                    TEXT,
    accident_area           TEXT,
    sex                     TEXT,
    marital_status           TEXT,
    age                     INTEGER,
    fault                   TEXT,
    policy_type             TEXT,
    vehicle_category         TEXT,
    vehicle_price_bucket     TEXT,                  -- real categorical bucket, not a dollar amount
    deductible               NUMERIC,
    driver_rating            INTEGER,
    days_policy_accident     TEXT,
    days_policy_claim        TEXT,
    past_number_of_claims    TEXT,
    age_of_vehicle           TEXT,
    age_of_policyholder      TEXT,
    police_report_filed      TEXT,
    witness_present          TEXT,
    agent_type               TEXT,
    number_of_suppliments    TEXT,
    address_change_claim     TEXT,
    number_of_cars           TEXT,
    base_policy              TEXT,
    incident_period          TEXT,                  -- e.g. '1994-Dec', real Year+Month, no fabricated day
    fraud_reported            INTEGER,                -- real ground-truth label (FraudFound_P)
    raw_payload               JSONB,
    ingested_at                TIMESTAMP DEFAULT now()
);

-- Engineered features, one row per claim
CREATE TABLE IF NOT EXISTS claims.claim_features (
    claim_id                    TEXT PRIMARY KEY REFERENCES claims.raw_claims(claim_id),
    customer_id                  TEXT NOT NULL,
    past_claims_numeric          INTEGER,             -- parsed from past_number_of_claims bucket
    address_change_recent        BOOLEAN,             -- parsed from address_change_claim
    high_deductible_flag         BOOLEAN,
    low_driver_rating_flag       BOOLEAN,
    computed_at                   TIMESTAMP DEFAULT now()
);

-- Model outputs, one row per claim
CREATE TABLE IF NOT EXISTS claims.model_scores (
    claim_id                TEXT PRIMARY KEY REFERENCES claims.raw_claims(claim_id),
    fraud_probability       NUMERIC,   -- XGBoost, trained on real fraud_reported label
    anomaly_score           NUMERIC,   -- Isolation Forest on real feature set
    customer_risk_index     NUMERIC,   -- composed from real per-claim risk signals
    scored_at                TIMESTAMP DEFAULT now()
);

-- LLM reasoning output, one row per claim
CREATE TABLE IF NOT EXISTS claims.llm_rationale (
    claim_id             TEXT PRIMARY KEY REFERENCES claims.raw_claims(claim_id),
    discrepancy_detected BOOLEAN,
    flagged_reasons       TEXT[],
    recommended_action   TEXT,
    rationale_text        TEXT,
    generated_at          TIMESTAMP DEFAULT now()
);

-- Final combined view the Streamlit dashboard reads from
CREATE OR REPLACE VIEW claims.claim_risk_summary AS
SELECT
    r.claim_id,
    r.customer_id,
    r.vehicle_category,
    r.fault,
    r.base_policy,
    r.fraud_reported,
    s.fraud_probability,
    s.anomaly_score,
    s.customer_risk_index,
    l.recommended_action,
    l.flagged_reasons,
    l.rationale_text,
    CASE
        WHEN s.fraud_probability >= 0.7 OR s.anomaly_score >= 0.7 THEN 'RED'
        WHEN s.fraud_probability >= 0.4 OR s.anomaly_score >= 0.4 THEN 'YELLOW'
        ELSE 'GREEN'
    END AS triage_flag
FROM claims.raw_claims r
LEFT JOIN claims.model_scores s ON r.claim_id = s.claim_id
LEFT JOIN claims.llm_rationale l ON r.claim_id = l.claim_id;
