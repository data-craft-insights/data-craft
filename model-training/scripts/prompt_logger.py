# scripts/prompt_logger.py

import uuid
from datetime import datetime, timezone

from google.cloud import bigquery
import vertexai
from vertexai.preview.language_models import TextEmbeddingModel

PROJECT_ID = "datacraft-data-pipeline"
REGION = "us-central1"

DATASET_ID = "datacraft_ml"
TABLE_ID = "prompt_logs"

def log_prompt(
    prompt_text: str,
    model_name: str,
    user_id: str = "airflow_pipeline",
    channel: str = "airflow",
    task_type: str = "sql_viz_generation",
):
    """Log one prompt into BigQuery with real semantic embeddings."""

    # --- initialize vertex ai ---
    vertexai.init(project=PROJECT_ID, location=REGION)

    # --- generate embedding ---
    try:
        embedder = TextEmbeddingModel.from_pretrained("text-embedding-004")
        embedding = embedder.get_embeddings([prompt_text])[0].values
    except Exception as e:
        print(f"[prompt_logger] Embedding generation failed: {e}")
        embedding = []  # fallback

    # --- other metadata ---
    token_length = len(prompt_text.split())
    has_code = ("```" in prompt_text) or ("def " in prompt_text)
    has_sql = ("SELECT " in prompt_text.upper()) or ("FROM " in prompt_text.upper())

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
        "embedding": embedding,  # 👈 actual embedding array (768 floats)
    }

    table_ref = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"
    client = bigquery.Client(project=PROJECT_ID)

    print(f"[prompt_logger] inserting into {table_ref}")
    errors = client.insert_rows_json(table_ref, [row])

    if errors:
        print(f"[prompt_logger] Error logging prompt: {errors}")
