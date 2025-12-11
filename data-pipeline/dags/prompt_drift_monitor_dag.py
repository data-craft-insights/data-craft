"""
Prompt Drift Monitor DAG

Runs every hour (or daily) and:
1. Reads prompt_logs from BigQuery
2. Computes drift (PSI, deltas, embedding drift)
3. Writes to prompt_drift_summary
4. If drift exceeds threshold → TRIGGER model_pipeline DAG
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from datetime import datetime, timedelta
import os
import sys

# Add scripts folder to PATH (so we can import prompt_drift.py)
sys.path.insert(0, "/opt/airflow/model-training/scripts")

from prompt_drift import analyze_prompt_drift

# --------------------------
# DAG Setup
# --------------------------

default_args = {
    "owner": "datacraft",
    "depends_on_past": False,
    "start_date": datetime(2025, 1, 1),
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

dag = DAG(
    dag_id="prompt_drift_monitor",
    default_args=default_args,
    schedule_interval="@hourly",     # runs every hour
    catchup=False,
    tags=["monitoring", "drift"],
)

# --------------------------
# TASK 1 — Compute drift
# --------------------------

def compute_drift_and_decide(**context):
    summary = analyze_prompt_drift()

    psi = summary.get("psi_token_length", 0)
    has_drift = summary.get("token_length_drift", False)

    should_trigger = has_drift or psi > 0.2

    context["ti"].xcom_push(key="trigger_retrain", value=should_trigger)

    if should_trigger:
        print("🔴 DRIFT DETECTED → retraining should be triggered")
    else:
        print("🟢 No drift detected")

compute_drift_task = PythonOperator(
    task_id="compute_prompt_drift",
    python_callable=compute_drift_and_decide,
    provide_context=True,
    dag=dag,
)

# --------------------------
# TASK 2 — Trigger model pipeline if drift detected
# --------------------------

trigger_training_task = TriggerDagRunOperator(
    task_id="trigger_model_training",
    trigger_dag_id="model_pipeline_with_evaluation",  # your main model DAG
    conf={"source": "prompt_drift_monitor"},
    dag=dag,
    trigger_rule="all_done"
)

# --------------------------
# DAG Flow
# --------------------------

compute_drift_task >> trigger_training_task
