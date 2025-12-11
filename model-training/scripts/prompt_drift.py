# model-training/scripts/prompt_drift.py
"""
Prompt Drift Analysis Script

Reads prompt_logs from BigQuery, splits into baseline vs current windows,
computes drift metrics (PSI, binary deltas, embedding drift), and writes
a summary row into prompt_drift_summary.

This script is triggered by a separate Drift Monitor DAG — NOT the
model training DAG.
"""

import os
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from google.cloud import bigquery

from utils import setup_logging, setup_gcp_credentials


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "datacraft-data-pipeline")
DATASET_ID = os.environ.get("BQ_DATASET", "datacraft_ml")

PROMPT_LOGS_TABLE = f"{PROJECT_ID}.{DATASET_ID}.prompt_logs"
DRIFT_SUMMARY_TABLE = f"{PROJECT_ID}.{DATASET_ID}.prompt_drift_summary"

PSI_THRESHOLD = 0.2
EMBEDDING_DRIFT_THRESHOLD = 0.1


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------

def make_json_safe(obj: Any):
    """Convert any object to JSON-serializable format."""
    if isinstance(obj, datetime):
        return obj.isoformat()

    if isinstance(obj, (np.integer,)):
        return int(obj)

    if isinstance(obj, (np.floating,)):
        return float(obj)

    if isinstance(obj, (np.bool_)):
        return bool(obj)

    if isinstance(obj, list):
        return [make_json_safe(o) for o in obj]

    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}

    return obj


def load_prompt_logs(client, logger) -> pd.DataFrame:
    """Load all prompt logs from BigQuery."""
    query = f"""
        SELECT *
        FROM `{PROMPT_LOGS_TABLE}`
        ORDER BY ts
    """
    logger.info(f"Loading data from {PROMPT_LOGS_TABLE} …")
    df = client.query(query).to_dataframe()
    logger.info(f"Loaded {len(df)} rows.")
    return df


def split_baseline_current(df: pd.DataFrame, frac=0.7):
    """First 70% → baseline, last 30% → current."""
    if df.empty:
        return df, df

    df = df.sort_values("ts")
    cutoff = int(len(df) * frac)
    cutoff = max(1, cutoff)

    baseline = df.iloc[:cutoff].copy()
    current = df.iloc[cutoff:].copy()

    return baseline, current


def compute_psi(baseline, current, bins=10) -> float:
    """Compute Population Stability Index."""
    baseline = np.array(baseline, dtype=float)
    current = np.array(current, dtype=float)

    if len(baseline) == 0 or len(current) == 0:
        return 0.0

    cuts = np.percentile(baseline, np.linspace(0, 100, bins + 1))
    cuts = np.unique(cuts)

    if len(cuts) <= 1:
        return 0.0

    b_counts, _ = np.histogram(baseline, bins=cuts)
    c_counts, _ = np.histogram(current, bins=cuts)

    b_dist = b_counts / max(b_counts.sum(), 1)
    c_dist = c_counts / max(c_counts.sum(), 1)

    eps = 1e-6
    b_dist = np.where(b_dist == 0, eps, b_dist)
    c_dist = np.where(c_dist == 0, eps, c_dist)

    psi_vals = (b_dist - c_dist) * np.log(b_dist / c_dist)
    return float(np.sum(psi_vals))


def compute_binary_delta(b, c):
    """Difference in means for binary features."""
    if b.empty or c.empty:
        return 0.0
    return float(c.mean() - b.mean())

def _compute_embedding_drift(
    baseline: pd.Series,
    current: pd.Series,
    logger
) -> Tuple[Optional[float], bool]:
    """
    Compute cosine distance between baseline and current embedding centroids.
    If embeddings are missing / null, returns (None, False).
    Now robust to:
      - nulls
      - empty lists
      - mixed shapes
    """
    # Drop nulls first
    b_nonnull = baseline.dropna()
    c_nonnull = current.dropna()

    if b_nonnull.empty or c_nonnull.empty:
        logger.info("No non-null embeddings found; skipping embedding drift.")
        return None, False

    def to_vec(x):
        """
        Convert one embedding cell into a 1D numpy vector.
        Handles:
          - list of floats
          - numpy array
        Skips:
          - empty lists/arrays
        """
        try:
            arr = np.asarray(x, dtype=float).ravel()
            if arr.size == 0:
                return None
            return arr
        except Exception:
            return None

    # Convert and filter
    b_vecs_list = [to_vec(x) for x in b_nonnull]
    c_vecs_list = [to_vec(x) for x in c_nonnull]

    b_vecs_list = [v for v in b_vecs_list if v is not None]
    c_vecs_list = [v for v in c_vecs_list if v is not None]

    if not b_vecs_list or not c_vecs_list:
        logger.info("No valid (non-empty) embeddings after conversion; skipping drift.")
        return None, False

    # Ensure all vectors are same dimension
    dim = b_vecs_list[0].shape[0]
    b_vecs_list = [v for v in b_vecs_list if v.shape[0] == dim]
    c_vecs_list = [v for v in c_vecs_list if v.shape[0] == dim]

    if not b_vecs_list or not c_vecs_list:
        logger.info("No embeddings with consistent dimension; skipping drift.")
        return None, False

    b_vecs = np.vstack(b_vecs_list)
    c_vecs = np.vstack(c_vecs_list)

    b_centroid = b_vecs.mean(axis=0)
    c_centroid = c_vecs.mean(axis=0)

    # cosine distance = 1 - cosine similarity
    denom = (np.linalg.norm(b_centroid) * np.linalg.norm(c_centroid))
    if denom == 0:
        logger.info("Zero-norm centroid encountered; skipping embedding drift.")
        return None, False

    cos_sim = float(np.dot(b_centroid, c_centroid) / denom)
    cos_dist = 1.0 * 1.0 - cos_sim

    drift = cos_dist > EMBEDDING_DRIFT_THRESHOLD
    return cos_dist, drift



# ---------------------------------------------------------
# MAIN FUNCTION
# ---------------------------------------------------------

def analyze_prompt_drift():
    logger = setup_logging("PromptDrift")
    logger.info("🚀 Starting Prompt Drift Analysis…")

    setup_gcp_credentials(None, logger)
    client = bigquery.Client(project=PROJECT_ID)

    df = load_prompt_logs(client, logger)
    if df.empty:
        logger.warning("No data available.")
        return {"status": "empty"}

    baseline, current = split_baseline_current(df)

    psi = compute_psi(baseline.token_length, current.token_length)
    delta_code = compute_binary_delta(baseline.has_code, current.has_code)
    delta_sql = compute_binary_delta(baseline.has_sql, current.has_sql)

    emb_dist, emb_drift = _compute_embedding_drift(
        baseline.embedding,
        current.embedding,
        logger,
    )

    summary = {
        "run_ts": datetime.utcnow(),
        "psi_token_length": psi,
        "token_length_drift": psi > PSI_THRESHOLD,
        "delta_has_code": delta_code,
        "delta_has_sql": delta_sql,
        "embedding_cosine_distance": emb_dist,
        "embedding_drift": emb_drift,
        "baseline_period_start": baseline.ts.min().to_pydatetime(),
        "baseline_period_end": baseline.ts.max().to_pydatetime(),
        "current_period_start": current.ts.min().to_pydatetime(),
        "current_period_end": current.ts.max().to_pydatetime(),
        "baseline_count": len(baseline),
        "current_count": len(current),
    }

    safe_row = make_json_safe(summary)

    logger.info(f"Writing summary to {DRIFT_SUMMARY_TABLE} …")
    errors = client.insert_rows_json(DRIFT_SUMMARY_TABLE, [safe_row])

    if errors:
        logger.error(f"Insert errors: {errors}")
    else:
        logger.info("✓ Drift summary inserted successfully.")

    return summary


if __name__ == "__main__":
    result = analyze_prompt_drift()
    print(result)
