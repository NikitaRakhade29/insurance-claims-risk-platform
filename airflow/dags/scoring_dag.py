
from __future__ import annotations

from datetime import datetime

from airflow.decorators import dag, task


@dag(
    dag_id="score_and_reason_claims",
    schedule="0 * * * *",  # hourly
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args={"retries": 2, "retry_delay": 60},
    tags=["claims", "ml", "llm"],
)
def scoring_dag():

    @task
    def run_scoring():
        import sys
        sys.path.append("/opt/airflow")
        from models.score_all import main as score_main
        score_main()

    @task
    def run_llm_reasoning():
        import sys
        sys.path.append("/opt/airflow")
        from llm_reasoning.generate_rationale import main as reasoning_main
        reasoning_main(batch_size=50)

    run_scoring() >> run_llm_reasoning()


scoring_dag()
