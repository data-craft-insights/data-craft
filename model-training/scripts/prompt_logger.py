# scripts/prompt_logger.py

"""
Utility to log LLM prompts into BigQuery (for prompt drift monitoring).
"""

import uuid
from datetime import datetime, timezone

from google.cloud import bigquery

PROJECT_ID = "datacraft-478300"      # <- your GCP project
DATASET_ID = "datacraft_ml"          # <- your dataset
TABLE_ID = "prompt_logs"             # <- table you just created


def log_prompt(
    prompt_text: str,
    model_name: str,
    user_id: str = "airflow_pipeline",
    channel: str = "airflow",
    task_type: str = "sql_viz_generation",
):
    """
    Log one prompt into BigQuery. Embedding is left empty for now (can be added later).
    """
    client = bigquery.Client(project=PROJECT_ID)

    token_length = len(prompt_text.split())
    has_code = "```" in prompt_text or "def " in prompt_text
    has_sql = "SELECT " in prompt_text.upper() or "FROM " in prompt_text.upper()

    row = {
        "prompt_id": str(uuid.uuid4()),
        "ts": datetime.now(timezone.utc).isoformat(),
        "user_id": user_id,
        "channel": channel,
        "model_name": model_name,
        "task_type": task_type,
        "prompt_text": prompt_text,
        "token_length": token_length,
        "has_code": has_code,
        "has_sql": has_sql,
        "embedding": [],   # ARRAY<FLOAT64>, empty for now
    }

    table_ref = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"
    errors = client.insert_rows_json(table_ref, [row])
    if errors:
        # Don't kill the pipeline if logging fails
        print(f"[prompt_logger] Error logging prompt: {errors}")
