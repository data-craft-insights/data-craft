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

<img width="1469" height="591" alt="Screenshot 2025-11-25 at 8 25 34 PM" src="https://github.com/user-attachments/assets/e40bbbbe-6897-4a10-a1fc-d53c6051315c" />


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
<img width="1117" height="529" alt="Screenshot 2025-11-25 at 1 07 27 PM" src="https://github.com/user-attachments/assets/79931382-c38b-4fd0-b84b-f4f746d4c849" />


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
<img width="677" height="539" alt="Screenshot 2025-11-25 at 1 36 29 PM" src="https://github.com/user-attachments/assets/e70209f9-0651-4f19-964d-6b9843bfd244" />

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
<img width="655" height="299" alt="Screenshot 2025-11-25 at 1 45 18 PM" src="https://github.com/user-attachments/assets/cfe35620-8df9-498a-b17f-e59d3682defc" />

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
<img width="689" height="283" alt="Screenshot 2025-11-25 at 1 45 52 PM" src="https://github.com/user-attachments/assets/87603c17-c29e-4d55-ba60-d667bda31dc4" />

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
<img width="677" height="271" alt="Screenshot 2025-11-25 at 1 48 43 PM" src="https://github.com/user-attachments/assets/f886ea67-fe31-4dd9-973d-a5ba751319c1" />

<img width="673" height="432" alt="Screenshot 2025-11-25 at 1 46 42 PM" src="https://github.com/user-attachments/assets/75e97d12-2922-4bf0-984b-8fddeeab142b" />

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
<img width="683" height="184" alt="Screenshot 2025-11-25 at 1 49 25 PM" src="https://github.com/user-attachments/assets/ab37dec6-eac7-44e7-9c4e-ba192c296d68" />
<img width="745" height="642" alt="Screenshot 2025-11-25 at 1 50 02 PM" src="https://github.com/user-attachments/assets/191d0bcf-5936-4392-b78b-17f2c8134149" />


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
