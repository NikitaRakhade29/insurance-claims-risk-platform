
from __future__ import annotations

import json
from datetime import datetime

import psycopg2
from airflow.decorators import dag, task
from kafka import KafkaConsumer

TOPIC = "claims-events"
KAFKA_BOOTSTRAP = "kafka:19092"
PG_CONN = dict(
    host="postgres", dbname="airflow", user="airflow", password="airflow"
)

COLUMNS = [
    "claim_id", "customer_id", "make", "accident_area", "sex", "marital_status",
    "age", "fault", "policy_type", "vehicle_category", "vehicle_price_bucket",
    "deductible", "driver_rating", "days_policy_accident", "days_policy_claim",
    "past_number_of_claims", "age_of_vehicle", "age_of_policyholder",
    "police_report_filed", "witness_present", "agent_type",
    "number_of_suppliments", "address_change_claim", "number_of_cars",
    "base_policy", "incident_period", "fraud_reported",
]


@dag(
    dag_id="ingest_claims_from_kafka",
    schedule="*/10 * * * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args={"retries": 3, "retry_delay": 30},
    tags=["claims", "kafka", "ingestion"],
)
def ingest_claims_dag():

    @task
    def consume_and_load(poll_timeout_ms: int = 8000) -> int:
        consumer = KafkaConsumer(
            TOPIC,
            bootstrap_servers=KAFKA_BOOTSTRAP,
            auto_offset_reset="earliest",
            enable_auto_commit=True,
            group_id="claims-ingestion-group",
            consumer_timeout_ms=poll_timeout_ms,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        )

        rows_loaded = 0
        conn = psycopg2.connect(**PG_CONN)
        cur = conn.cursor()

        placeholders = ", ".join(["%s"] * len(COLUMNS))
        col_list = ", ".join(COLUMNS)
        insert_sql = f"""
            INSERT INTO claims.raw_claims ({col_list}, raw_payload)
            VALUES ({placeholders}, %s)
            ON CONFLICT (claim_id) DO NOTHING
        """

        for message in consumer:
            claim = message.value
            values = [str(claim.get(c)) if claim.get(c) is not None else None for c in COLUMNS]
            cur.execute(insert_sql, values + [json.dumps(claim)])
            rows_loaded += 1

        conn.commit()
        cur.close()
        conn.close()
        consumer.close()

        print(f"Loaded {rows_loaded} claim events into raw_claims")
        return rows_loaded

    consume_and_load()


ingest_claims_dag()
