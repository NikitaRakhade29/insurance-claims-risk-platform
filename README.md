# infrastructure + streaming ingestion

## 0. Get the dataset
Download **Vehicle Insurance Claim Fraud Detection**:
https://www.kaggle.com/datasets/shivamb/vehicle-claim-fraud-detection

Save the CSV as `data/raw/claims.csv`.

**Important:** this dataset's column names won't exactly match the
`raw_claims` schema (`claim_id`, `customer_id`, `incident_type`,
`claim_amount`, `policy_premium`, `incident_date`, `claim_filed_date`).
Open the CSV, check the real column names, and either:
- rename columns in a quick pandas script before running the producer, or
- adjust `kafka/producer.py` and `airflow/dags/ingest_claims_dag.py` to
  match the actual column names.

This mapping step is normal real-world data engineering — source
systems never match your target schema exactly.

## 1. Start the stack
Docker Desktop must be running.

```bash
docker compose up -d --build
```

This starts:
- `postgres` — warehouse + Airflow metadata DB (schema.sql auto-applies on first boot)
- `kafka` — single-broker KRaft mode (no Zookeeper)
- `airflow-init` — creates the admin user, then exits
- `airflow-webserver` — http://localhost:8080 (login: admin / admin)
- `airflow-scheduler`

Give it 1-2 minutes on first boot. Check status:
```bash
docker compose ps
```

## 2. Verify Postgres schema applied
```bash
docker exec -it $(docker compose ps -q postgres) psql -U airflow -d airflow -c "\dt claims.*"
```
You should see `raw_claims`, `claim_features`, `model_scores`, `llm_rationale`.

## 3. Install local dependencies (for running the producer from your machine)
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 4. Stream the claims into Kafka
```bash
python kafka/producer.py --csv data/raw/claims.csv --delay 0.2
```
This replays every row as a Kafka event onto the `claims-events` topic —
your "live" claim stream for the rest of the pipeline.

## 5. Turn on the ingestion DAG
- Open http://localhost:8080
- Un-pause `ingest_claims_from_kafka`
- Trigger it manually once (don't wait for the schedule) to confirm it consumes
  from Kafka and loads rows into `claims.raw_claims`

Verify:
```bash
docker exec -it $(docker compose ps -q postgres) psql -U airflow -d airflow -c "SELECT count(*) FROM claims.raw_claims;"
```

