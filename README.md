# Insurance Claims Risk & Fraud Intelligence Platform

A Kafka-streamed, Airflow-orchestrated insurance claims platform that scores fraud risk, billing anomalies, and customer risk, then uses an LLM (Gemini) to turn those scores into a clear, adjuster-ready rationale.

## The problem

Insurance adjusters manually review every incoming claim at roughly the same pace, regardless of how suspicious it actually is. This lets real fraud slip through while clean claims sit in the same slow queue as risky ones. This platform automates the first-pass triage: it scores every claim on three independent signals and tells an adjuster which handful genuinely need their attention first.

## Architecture

```
 fraud_oracle.csv (Kaggle)
          │
          ▼
  preprocess_dataset.py  ──►  data/raw/claims.csv
          │
          ▼
  kafka/producer.py  ──►  Kafka topic "claims-events"  (single-broker, KRaft mode)
          │
          ▼
  Airflow DAG: ingest_claims_from_kafka  ──►  Postgres: claims.raw_claims
          │
          ▼
  features/feature_engineering.py  (DuckDB — explainable flags + encoded feature matrix)
          │
          ├──► models/train_fraud_classifier.py   (XGBoost, calibrated, real fraud_reported label)
          ├──► models/train_anomaly_detector.py   (Isolation Forest, unsupervised)
          └──► models/score_all.py                (percentile-based RED/YELLOW/GREEN triage)
                          │
                          ▼
                claims.model_scores (Postgres)
                          │
                          ▼
       llm_reasoning/generate_rationale.py  ──►  Gemini API (strict JSON reasoning)
                          │
                          ▼
                claims.llm_rationale (Postgres)
                          │
                          ▼
              dashboard/app.py  (Streamlit adjuster console)
```

Airflow DAG `score_and_reason_claims` chains scoring + reasoning together for scheduled runs; both also run standalone for local development.

## Tech stack

| Layer | Technology | Why |
|---|---|---|
| Streaming ingestion | Apache Kafka (KRaft, single broker) | Real-time-style claim ingestion, no Zookeeper needed |
| Orchestration | Apache Airflow (LocalExecutor) | Scheduling, retries, dependency management |
| Feature engineering | DuckDB | Fast in-process analytical SQL, no cluster needed at this data scale |
| ML | Scikit-learn, XGBoost | Fraud classifier (supervised, calibrated), anomaly detector (unsupervised) |
| Reasoning | Gemini API (also supports Claude/GPT) | Turns model scores into structured, adjuster-ready rationale |
| Storage | PostgreSQL | Claims warehouse + model scores + rationale |
| UI | Streamlit | Adjuster triage console |
| Infra | Docker Compose | Runs the full stack locally |

## Repository structure

```
insurance-claims-risk-platform/
├── .streamlit/
│   └── config.toml              # dark theme, hides Streamlit's Deploy button
├── docker-compose.yml
├── requirements.txt
├── preprocess_dataset.py        # maps fraud_oracle.csv -> claims.csv (real columns only)
├── db/
│   └── schema.sql
├── kafka/
│   └── producer.py
├── airflow/dags/
│   ├── ingest_claims_dag.py
│   └── scoring_dag.py
├── features/
│   └── feature_engineering.py
├── models/
│   ├── train_fraud_classifier.py
│   ├── train_anomaly_detector.py
│   ├── score_all.py
│   └── saved/                   # trained model artifacts (gitignored)
├── llm_reasoning/
│   ├── prompt_template.py
│   ├── reasoning_client.py
│   └── generate_rationale.py
├── dashboard/
│   └── app.py
├── data/raw/                    # dataset (gitignored if large)
└── assets/
    └── screenshots/             # demo images — see below
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate              # Windows
pip install -r requirements.txt
cp .env.example .env               # fill in GEMINI_API_KEY, DATABASE_URL etc.

docker compose up -d --build
docker compose ps                  # confirm all containers running

python preprocess_dataset.py --input data/raw/fraud_oracle.csv --output data/raw/claims.csv
python kafka/producer.py --csv data/raw/claims.csv --delay 0.05
# In Airflow UI (localhost:8080): un-pause + trigger ingest_claims_from_kafka

python models/train_fraud_classifier.py
python models/train_anomaly_detector.py
python models/score_all.py
python llm_reasoning/generate_rationale.py --batch-size 20

streamlit run dashboard/app.py
```

## Where to upload screenshots

Create the folder if it doesn't exist, then drop your PNG/JPG files in:

```
insurance-claims-risk-platform/assets/screenshots/
```

Suggested filenames (matches what's referenced below — rename yours to match, or update the paths):
- `dashboard-overview.png` — the top of the console showing the four metric cards
- `dashboard-claim-detail.png` — an expanded claim card with the AI rationale visible
- `airflow-dag-graph.png` — the Airflow UI showing the DAG graph view
- `training-metrics.png` — terminal output of the classifier's precision/recall/ROC-AUC

Then reference them here (or in your project report) like this:
```markdown
![Dashboard overview](assets/screenshots/dashboard-overview.png)
![Claim detail with AI rationale](assets/screenshots/dashboard-claim-detail.png)
```

## Known limitations (worth stating openly in a viva)

- **One claim per policy in this dataset** — `customer_id` and `claim_id` are effectively the same thing, so "customer risk index" reflects a single claim's own signals, not multi-claim history.
- **`claim_amount` / `policy_premium` don't exist in the source data** — only a `vehicle_price_bucket` category does. Nothing was fabricated to fill this gap; the schema simply doesn't include invented monetary fields.
- **Gemini free tier caps at ~20 requests/day/model** — `generate_rationale.py` only processes RED/YELLOW claims in batches and skips ones already covered, so rationale coverage grows incrementally across runs/days unless billing is enabled.
- **Triage thresholds are percentile-based** (top ~8% RED, next ~12% YELLOW), not fixed probability cutoffs — this keeps the flagged queue at a realistic, actionable size regardless of the raw score distribution.
