# CI/CD Pipeline for Model Training

Automated pipeline that triggers model training, validates quality, and deploys to production.

## Pipeline Flow

```
Push to GitHub
    ↓
Trigger Airflow DAG
    ↓
Download Outputs from GCS
    ↓
Validate Model Performance
    ↓
Check Model Bias
    ↓
Compare with Production & Deploy
    ↓
Send Notifications
```

## Pipeline Steps

### 1. Trigger DAG
**Job:** `trigger_dag`  
**Script:** `ci-cd/scripts/trigger_dag.py`

**What it does:**
- Triggers Airflow DAG `model_pipeline_with_evaluation`
- Waits for DAG completion (max 60 minutes)
- Fails if DAG fails or times out

**Files used:**
- None (uses Airflow API)

**Screenshot:**
<!-- Add screenshot of trigger_dag job here -->

---

### 2. Download Outputs
**Job:** `download_outputs`  
**Script:** `ci-cd/scripts/download_outputs.py`

**What it does:**
- Downloads latest best model responses from GCS
- Source: `best_model_responses/{timestamp}_{model}/` in GCS bucket
- Extracts to `outputs/best-model-responses/` directory
- Uploads as GitHub artifact for subsequent jobs

**Files used:**
- `ci-cd/config/ci_cd_config.yaml` (GCP bucket configuration)
- Downloads: `summary.json`, `model_selection_report.json` from GCS

**Screenshot:**
<!-- Add screenshot of download_outputs job here -->

---

### 3. Validate Model Performance
**Job:** `validate`  
**Script:** `ci-cd/scripts/validate_model.py`

**What it does:**
- Validates performance metrics (composite_score, performance_score, success_rate)
- Validates execution metrics (execution_success_rate, result_validity_rate, overall_accuracy)
- Checks against thresholds in `validation_thresholds.yaml`
- Fails if any threshold not met

**Files used:**
- `outputs/best-model-responses/model_selection_report.json` (performance metrics)
- `outputs/best-model-responses/summary.json` (execution metrics: total_queries, successful_queries, failed_queries)
- `ci-cd/config/validation_thresholds.yaml` (thresholds)

**Screenshot:**
<!-- Add screenshot of validate job here -->

---

### 4. Check Model Bias
**Job:** `check_bias`  
**Script:** `ci-cd/scripts/check_bias.py`

**What it does:**
- Validates bias_score from model selection report
- Checks bias severity level (LOW, MEDIUM, HIGH)
- Fails if bias exceeds thresholds

**Files used:**
- `outputs/best-model-responses/model_selection_report.json` (bias_score)
- `ci-cd/config/validation_thresholds.yaml` (bias thresholds)

**Screenshot:**
<!-- Add screenshot of check_bias job here -->

---

### 5. Compare with Production & Deploy
**Job:** `compare_and_deploy`  
**Scripts:** `ci-cd/scripts/rollback_manager.py`, `ci-cd/scripts/push_to_registry.py`

**What it does:**
- Compares current model metrics with production model
- Blocks deployment if new model is worse (rollback protection)
- Deploys to production if better or equal
- Uploads to `models/{timestamp}_{commit}/` in GCS with commit SHA tag

**Files used:**
- `outputs/best-model-responses/model_selection_report.json` (current model metrics)
- Production model from GCS: `models/` (previous deployment)
- `ci-cd/config/ci_cd_config.yaml` (rollback configuration)

**Screenshot:**
<!-- Add screenshot of compare_and_deploy job here -->

---

### 6. Send Notifications
**Job:** `notify`  
**Script:** `ci-cd/scripts/send_notification.py`

**What it does:**
- Sends email notification with pipeline status
- Includes best model name, scores, and metrics
- Runs even if previous jobs fail

**Files used:**
- `outputs/best-model-responses/model_selection_report.json` (model metrics)
- `outputs/best-model-responses/summary.json` (execution stats)
- `ci-cd/config/ci_cd_config.yaml` (email configuration)

**Screenshot:**
<!-- Add screenshot of notify job here -->

---

## Files Used by Pipeline

### Input Files (from GCS)
- `summary.json` - Query execution summary (total_queries, successful_queries, failed_queries)
- `model_selection_report.json` - Model metrics (composite_score, performance_score, bias_score, success_rate)

### Configuration Files
- `ci-cd/config/ci_cd_config.yaml` - GCP settings, email config, rollback settings
- `ci-cd/config/validation_thresholds.yaml` - Quality thresholds (performance, bias, execution)

### Output Files (generated)
- `outputs/validation/validation_report.json` - Validation results
- `outputs/validation/bias_check_report.json` - Bias check results
- `outputs/validation/model_comparison_report.json` - Production comparison results

## Pipeline Triggers

Pipeline triggers on push to `main` branch when these files change:
- `model-training/data/user_queries.txt`
- `model-training/scripts/**`
- `ci-cd/**`
- `.github/workflows/model-training-ci-cd.yml`

Can also be triggered manually via GitHub Actions UI.
