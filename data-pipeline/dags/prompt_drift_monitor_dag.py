"""
Prompt Drift Monitor DAG

Runs on a schedule (e.g., hourly) and:
1. Reads prompt_logs from BigQuery
2. Computes drift (PSI, code/share deltas, embedding drift)
3. Writes a row into prompt_drift_summary
4. If any drift signal exceeds thresholds -> TRIGGER model_pipeline_with_evaluation DAG
"""

from datetime import datetime, timedelta
import sys

from airflow import DAG
from airflow.operators.python import ShortCircuitOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

# Make sure Airflow can import prompt_drift.py from scripts folder
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
    schedule_interval="@hourly",   # or "@daily" if you prefer
    catchup=False,
    tags=["monitoring", "drift"],
)


# --------------------------
# TASK 1 — Compute drift & decide
# --------------------------

def compute_drift_and_decide(**context) -> bool:
    """
    1. Run analyze_prompt_drift() from prompt_drift.py
    2. Read PSI, delta_has_code, delta_has_sql, embedding metrics
    3. Decide whether to trigger retraining
    4. Return True/False for ShortCircuitOperator
    """
    summary = analyze_prompt_drift()

    # --- Read metrics from summary (with safe defaults) ---
    psi = float(summary.get("psi_token_length", 0.0))
    token_length_drift = bool(summary.get("token_length_drift", False))

    delta_code = float(summary.get("delta_has_code", 0.0))
    delta_sql = float(summary.get("delta_has_sql", 0.0))

    emb_dist = summary.get("embedding_cosine_distance", None)
    emb_drift = bool(summary.get("embedding_drift", False))

    # --- Thresholds (you can tune these / later move to YAML config) ---
    PSI_THRESHOLD = 0.2             # same as in prompt_drift.py
    CODE_DELTA_THRESHOLD = 0.15     # 15 percentage points change in has_code
    SQL_DELTA_THRESHOLD = 0.15      # 15 percentage points change in has_sql
    EMBEDDING_DIST_THRESHOLD = 0.1  # consistent with prompt_drift.py idea

    # --- Individual flags ---
    psi_flag = psi > PSI_THRESHOLD or token_length_drift

    code_flag = abs(delta_code) > CODE_DELTA_THRESHOLD
    sql_flag = abs(delta_sql) > SQL_DELTA_THRESHOLD

    if emb_dist is not None:
        emb_flag = emb_drift or (float(emb_dist) > EMBEDDING_DIST_THRESHOLD)
    else:
        emb_flag = False

    # --- Final decision: trigger if ANY signal shows drift ---
    should_trigger = psi_flag or code_flag or sql_flag or emb_flag

    # Push for debugging / observability if you want to inspect later in XCom
    ti = context["ti"]
    ti.xcom_push(key="trigger_retrain", value=should_trigger)
    ti.xcom_push(key="psi_flag", value=psi_flag)
    ti.xcom_push(key="code_flag", value=code_flag)
    ti.xcom_push(key="sql_flag", value=sql_flag)
    ti.xcom_push(key="emb_flag", value=emb_flag)

    print("====== Prompt Drift Decision ======")
    print(f"PSI: {psi:.4f}, token_length_drift={token_length_drift}, psi_flag={psi_flag}")
    print(f"Δ has_code: {delta_code:.4f}, code_flag={code_flag}")
    print(f"Δ has_sql:  {delta_sql:.4f}, sql_flag={sql_flag}")
    print(f"Embedding distance: {emb_dist}, emb_drift={emb_drift}, emb_flag={emb_flag}")
    print(f"=> should_trigger = {should_trigger}")
    print("===================================")

    if should_trigger:
        print("🔴 DRIFT DETECTED → retraining will be triggered")
    else:
        print("🟢 No significant drift detected → model pipeline will NOT be triggered")

    # ShortCircuitOperator uses this return value to decide
    # whether to run downstream tasks.
    return should_trigger


compute_drift_task = ShortCircuitOperator(
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
    trigger_dag_id="model_pipeline_with_evaluation",  # your main model DAG id
    conf={"source": "prompt_drift_monitor"},
    dag=dag,
)


# --------------------------
# DAG Flow
# --------------------------

# If compute_drift_task returns False, ShortCircuitOperator will SKIP
# trigger_training_task. If it returns True, it will run it.
compute_drift_task >> trigger_training_task
