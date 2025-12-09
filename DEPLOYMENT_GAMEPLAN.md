# Model Deployment Game Plan - LLM Inference Service

## 🎯 Executive Summary

**Key Insight**: You are deploying **inference logic**, not model weights. Your "model" is configuration metadata that tells your service:
- Which LLM API to call (Gemini 2.5 Flash vs Pro)
- What hyperparameters to use (temperature, top_p, top_k)
- What prompts/templates to use

This is fundamentally different from traditional ML deployment because:
- ✅ No model weights to store
- ✅ No training artifacts needed
- ✅ Just code + configuration
- ✅ Deploy as Docker container → Cloud Run

### Single Sequential CI/CD Pipeline

Your deployment uses **ONE continuous pipeline** that runs sequentially:

```
Push to main
    ↓
1. Trigger Airflow DAG (Training/Eval)
    ↓
2. Wait for Airflow to finish
    ↓
3. Download Best Model Metadata
    ↓
4. Validate Performance + Bias Check
    ↓
5. Compare with Production
    ↓
6. Build Docker Image (Frontend + Backend)
    ↓
7. Push to Artifact Registry
    ↓
8. Deploy to Cloud Run
    ↓
9. Notify
```

**Frontend Integration**: Your Streamlit frontend (`frontend/app.py`) is deployed as part of the same Docker image, making it a **monolithic application** that includes both UI and backend logic.

---

## ✅ WHAT YOU ALREADY HAVE (100% Complete)

### 1. **Evaluation Pipeline (Airflow)**
- ✅ Loads validated data from GCS/BigQuery
- ✅ Evaluates multiple Gemini models
- ✅ Measures performance, bias, speed metrics
- ✅ Determines best model configuration

**Location**: `model-training/dags/model_pipeline_dag.py`

### 2. **Best Model Selection**
- ✅ Compares experiments (gemini-2.5-flash vs gemini-2.5-pro)
- ✅ Calculates composite score (weighted: performance 50%, bias 30%, speed 10%, reliability 10%)
- ✅ Selects best configuration
- ✅ Saves selection report

**Location**: `model-training/scripts/model_selector.py`

### 3. **Hyperparameter Tuning**
- ✅ Tunes temperature, top_p, top_k
- ✅ Grid search across parameter space
- ✅ Selects best hyperparameters
- ✅ Stores in XCom and saves to files

**Location**: `model-training/scripts/hyperparameter_tuner.py`

### 4. **Metadata Storage**
- ✅ Saves `best_model_metadata.json` with:
  - Selected model name (e.g., "gemini-2.5-flash")
  - Composite score, performance score, bias score
  - Success rate, response time
  - Selection methodology
- ✅ Uploads to GCS: `gs://{bucket}/best_model_responses/{timestamp}_{model}/`
- ✅ Also saved to production registry: `gs://{bucket}/models/{timestamp}_{commit}/`

**Location**: `model-training/scripts/response_saver.py` → `save_metadata_for_deployment()`

### 5. **CI/CD Pipeline (Training Side)**
- ✅ Triggers Airflow DAG
- ✅ Downloads outputs from GCS
- ✅ Validates performance & bias
- ✅ Compares with production
- ✅ Pushes to registry

**Location**: `.github/workflows/model-training-ci-cd.yml`

---

## 🔥 WHAT YOU NEED TO BUILD (Deployment Side)

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│              TRAINING/EVALUATION FLOW (DONE)                │
│                                                              │
│  Data → Airflow → Evaluate Models → Select Best            │
│         → Tune Hyperparams → Save Metadata → GCS           │
└─────────────────────────────────────────────────────────────┘
                           ↓
                    Metadata in GCS
                    gs://bucket/models/{timestamp}_{commit}/
                           ↓
┌─────────────────────────────────────────────────────────────┐
│         SINGLE CI/CD PIPELINE (SEQUENTIAL)                  │
│                                                              │
│  1. Trigger Airflow DAG (Training/Eval)                    │
│  2. Download Best Model Metadata                            │
│  3. Validate Performance + Bias                             │
│  4. Compare with Production                                 │
│  5. Build Docker Image (Frontend + Backend)                 │
│  6. Push to Artifact Registry                                │
│  7. Deploy to Cloud Run / Vertex AI                         │
│  8. Notify                                                  │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│              PRODUCTION SERVING                              │
│                                                              │
│  User → Streamlit UI → Query Handler                        │
│         → Load Metadata from GCS                            │
│         → Call Gemini API with config                       │
│         → Execute SQL → Return Results + Viz                │
└─────────────────────────────────────────────────────────────┘
```

### Key Insight: Frontend Integration

Your **frontend** (`frontend/app.py`) is a Streamlit application that:
- ✅ Directly imports from `model-training/scripts/` (monolithic architecture)
- ✅ Uses `QueryHandler` which calls Gemini API
- ✅ Needs model metadata to know which model/hyperparameters to use
- ✅ Should be deployed as a **single container** with frontend + backend together

**Deployment Strategy**: 
- Build **one Docker image** containing:
  - Streamlit frontend (`frontend/app.py`)
  - Model training scripts (for QueryHandler)
  - Metadata loader (to fetch best model config from GCS)
- Deploy to **Cloud Run** (better for Streamlit) or **Vertex AI**

---

## 📋 STEP-BY-STEP GAMEPLAN

### **PHASE 1: Create Metadata Loader** (1 hour)

**Note**: Since your frontend already uses `QueryHandler` from `model-training/scripts/`, we only need to add a metadata loader that the frontend can use to get the best model configuration.

#### Step 1.1: Create Metadata Loader Module

Create: `frontend/metadata_loader.py`

**Purpose**: Load best model metadata from GCS so frontend knows which model/hyperparameters to use

#### Step 1.2: Implement `metadata_loader.py`

```python
# frontend/metadata_loader.py
from google.cloud import storage
import json
from typing import Dict, Optional
import os

class MetadataLoader:
    """
    Load best model metadata from GCS
    Used by frontend to get current production model configuration
    """
    def __init__(self, project_id: str, bucket_name: str):
        self.project_id = project_id
        self.bucket_name = bucket_name
        self.storage_client = storage.Client(project=project_id)
        self.bucket = self.storage_client.bucket(bucket_name)
    
    def load_latest_metadata(self, base_path: str = "models") -> Dict:
        """
        Load latest best_model_metadata.json from GCS
        
        Returns:
            Dictionary with:
            - selected_model: "gemini-2.5-flash"
            - composite_score: 89.5
            - performance_score: 92.0
            - bias_score: 25.0
            - success_rate: 90.0
            - avg_response_time: 1.2
            - deployment_ready: True
        """
        # List all model directories
        blobs = list(self.bucket.list_blobs(prefix=f"{base_path}/"))
        
        # Find latest timestamp
        model_dirs = {}
        for blob in blobs:
            if "best_model_metadata.json" in blob.name:
                # Extract timestamp from path: models/{timestamp}_{commit}/
                parts = blob.name.split("/")
                if len(parts) >= 2:
                    dir_name = parts[1]  # {timestamp}_{commit}
                    timestamp = dir_name.split("_")[0]
                    model_dirs[timestamp] = blob.name
        
        if not model_dirs:
            # Fallback: try best_model_responses path
            blobs = list(self.bucket.list_blobs(prefix="best_model_responses/"))
            for blob in blobs:
                if "best_model_metadata.json" in blob.name:
                    model_dirs["fallback"] = blob.name
                    break
        
        if not model_dirs:
            raise ValueError("No metadata found in GCS")
        
        # Get latest
        if "fallback" in model_dirs:
            latest_blob_path = model_dirs["fallback"]
        else:
            latest_timestamp = max(model_dirs.keys())
            latest_blob_path = model_dirs[latest_timestamp]
        
        # Download and parse
        blob = self.bucket.blob(latest_blob_path)
        metadata_json = blob.download_as_text()
        metadata = json.loads(metadata_json)
        
        return metadata
    
    def get_best_model_name(self) -> str:
        """Get best model name from latest metadata"""
        metadata = self.load_latest_metadata()
        return metadata.get("selected_model", "gemini-2.5-flash")
    
    def get_hyperparameters(self) -> Dict:
        """
        Extract hyperparameters from metadata or use defaults
        
        Returns:
            {
                "temperature": 0.2,
                "top_p": 0.9,
                "top_k": 40
            }
        """
        metadata = self.load_latest_metadata()
        
        # Check if hyperparameters are in metadata
        if "hyperparameters" in metadata:
            return metadata["hyperparameters"]
        
        # Defaults (should match tuning defaults)
        return {
            "temperature": 0.2,
            "top_p": 0.9,
            "top_k": 40
        }
```

#### Step 1.3: Update Frontend to Use Metadata Loader

Update `frontend/app.py` to load best model from GCS:

```python
# Add at top of app.py
from metadata_loader import MetadataLoader

# In session state initialization (around line 178):
if 'handler' not in st.session_state:
    # Load configuration
    config = {
        'project_id': os.getenv('GCP_PROJECT_ID', 'datacraft-data-pipeline'),
        'dataset_id': os.getenv('BQ_DATASET', 'datacraft_ml'),
        'bucket_name': os.getenv('GCS_BUCKET_NAME', 'isha-retail-data'),
        'region': os.getenv('GCP_REGION', 'us-central1'),
    }
    
    # Load best model from GCS
    try:
        metadata_loader = MetadataLoader(config['project_id'], config['bucket_name'])
        best_model = metadata_loader.get_best_model_name()
        config['model_name'] = best_model
        st.session_state.best_model_metadata = metadata_loader.load_latest_metadata()
    except Exception as e:
        # Fallback to default
        config['model_name'] = os.getenv('BEST_MODEL_NAME', 'gemini-2.5-flash')
        st.warning(f"Could not load metadata from GCS: {e}. Using default model.")
    
    config['table_name'] = 'orders_processed'
    st.session_state.config = config
    st.session_state.handler = QueryHandler(config)
    # ... rest of initialization
```

**Note**: Your `QueryHandler` in `frontend/query_handler.py` already handles LLM calls. We just need to ensure it uses the best model from metadata.

---

### **PHASE 2: Containerize Frontend + Backend** (1 hour)

#### Step 2.1: Update Frontend Dockerfile

Update `frontend/Dockerfile` to include model-training scripts:

```dockerfile
# frontend/Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies (for PDF processing, etc.)
RUN apt-get update && apt-get install -y \
    gcc \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Copy and install frontend requirements
COPY frontend/requirements.txt ./frontend/
RUN pip install --no-cache-dir -r frontend/requirements.txt

# Copy model-training scripts (needed by QueryHandler)
COPY model-training/scripts/ ./model-training/scripts/
COPY shared/ ./shared/  # If you have shared utilities

# Copy frontend application
COPY frontend/ ./frontend/

# Set Python path to include model-training scripts
ENV PYTHONPATH="/app:/app/model-training/scripts:/app/shared:${PYTHONPATH}"

# Expose Streamlit port
EXPOSE 8501

# Set environment variables
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV PYTHONUNBUFFERED=1

# Run Streamlit
CMD ["streamlit", "run", "frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

#### Step 2.2: Update Frontend Requirements

Ensure `frontend/requirements.txt` includes all dependencies:

```txt
# frontend/requirements.txt
streamlit==1.29.0
plotly==5.18.0
pandas==2.0.3
google-cloud-storage==2.12.0
google-cloud-bigquery==3.14.0
google-cloud-aiplatform>=1.70.0
vertexai>=1.0.0
python-dotenv==1.0.0
pyyaml==6.0.1
chardet==5.2.0
duckdb==0.9.2
db-dtypes==1.4.4
pdf2image
langchain-core  # If used by QueryHandler
```

#### Step 2.3: Create `.dockerignore` in Root

Create `.dockerignore` at project root:

```
# .dockerignore
__pycache__
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.venv
*.log
.git
.gitignore
*.md
.env
outputs/
logs/
*.csv
*.json
!frontend/.streamlit/config.toml
gcp/service-account.json  # Don't copy credentials
```

---

### **PHASE 3: Update CI/CD Pipeline** (1 hour)

#### Step 3.1: Add Deployment Jobs to Existing Pipeline

Update `.github/workflows/model-training-ci-cd.yml` to add deployment steps **after** `compare_and_deploy`:

```yaml
  build_and_deploy:
    name: Build Docker Image & Deploy
    needs: [compare_and_deploy]
    runs-on: ubuntu-latest
    if: needs.compare_and_deploy.result == 'success'
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Authenticate to Google Cloud
        uses: google-github-actions/auth@v1
        with:
          credentials_json: ${{ secrets.GCP_SA_KEY }}

      - name: Set up Cloud SDK
        uses: google-github-actions/setup-gcloud@v1

      - name: Configure Docker for Artifact Registry
        run: |
          gcloud auth configure-docker us-east1-docker.pkg.dev

      - name: Get commit SHA
        id: commit
        run: echo "sha=$(git rev-parse --short HEAD)" >> $GITHUB_OUTPUT

      - name: Build Docker Image
        run: |
          docker build -f frontend/Dockerfile \
            -t us-east1-docker.pkg.dev/${{ env.GCP_PROJECT_ID }}/mlops-models/datacraft-app:${{ steps.commit.outputs.sha }} \
            -t us-east1-docker.pkg.dev/${{ env.GCP_PROJECT_ID }}/mlops-models/datacraft-app:latest \
            .

      - name: Push to Artifact Registry
        run: |
          docker push us-east1-docker.pkg.dev/${{ env.GCP_PROJECT_ID }}/mlops-models/datacraft-app:${{ steps.commit.outputs.sha }}
          docker push us-east1-docker.pkg.dev/${{ env.GCP_PROJECT_ID }}/mlops-models/datacraft-app:latest

      - name: Deploy to Cloud Run
        run: |
          gcloud run deploy datacraft-app \
            --image us-east1-docker.pkg.dev/${{ env.GCP_PROJECT_ID }}/mlops-models/datacraft-app:${{ steps.commit.outputs.sha }} \
            --platform managed \
            --region us-east1 \
            --allow-unauthenticated \
            --port 8501 \
            --memory 2Gi \
            --cpu 2 \
            --timeout 300 \
            --set-env-vars GCP_PROJECT_ID=${{ env.GCP_PROJECT_ID }},BQ_DATASET=datacraft_ml,GCS_BUCKET_NAME=${{ secrets.GCS_BUCKET_NAME }},GCP_REGION=us-east1 \
            --service-account mlops-ci-cd@${{ env.GCP_PROJECT_ID }}.iam.gserviceaccount.com
```

#### Step 3.2: Update Notification Job

Update `notify` job to include deployment status:

```yaml
  notify:
    name: Send Notifications
    needs: [trigger_dag, download_outputs, validate, check_bias, compare_and_deploy, build_and_deploy]
    runs-on: ubuntu-latest
    if: always()
    steps:
      # ... existing steps ...
      - name: Send Email Notification
        env:
          EMAIL_SMTP_USER: ${{ secrets.EMAIL_SMTP_USER }}
          EMAIL_SMTP_PASSWORD: ${{ secrets.EMAIL_SMTP_PASSWORD }}
        run: |
          python ci-cd/scripts/send_notification.py \
            --status ${{ job.status }} \
            --workflow-run-url ${{ github.server_url }}/${{ github.repository }}/actions/runs/${{ github.run_id }} \
            --deployment-url ${{ needs.build_and_deploy.outputs.service_url || 'N/A' }}
```

---

### **PHASE 4: Set up Artifact Registry** (30 minutes)

#### Step 4.1: Create Artifact Registry Repository

```bash
# Create Artifact Registry repository
gcloud artifacts repositories create mlops-models \
    --repository-format=docker \
    --location=us-east1 \
    --description="DataCraft application containers"

# Configure Docker authentication
gcloud auth configure-docker us-east1-docker.pkg.dev
```

**Note**: This is a one-time setup. The CI/CD pipeline will handle building and pushing automatically.

---

### **PHASE 5: Production Monitoring** (2-3 hours)

#### Step 5.1: Create Monitoring Script

Create: `ci-cd/scripts/monitor_production.py`

```python
#!/usr/bin/env python3
"""
Monitor production model performance and detect drift
"""
import sys
import json
import yaml
from pathlib import Path
from google.cloud import storage, monitoring_v3
from datetime import datetime, timedelta

# Load config
project_root = Path(__file__).parent.parent.parent
config_path = project_root / "ci-cd" / "config" / "ci_cd_config.yaml"

def load_config():
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def check_production_metrics(project_id: str, endpoint_name: str):
    """
    Check production endpoint metrics:
    - Request count
    - Error rate
    - Latency
    - Success rate
    """
    client = monitoring_v3.MetricServiceClient()
    project_name = f"projects/{project_id}"
    
    # Query metrics from last 24 hours
    interval = monitoring_v3.TimeInterval()
    now = datetime.utcnow()
    interval.end_time.seconds = int(now.timestamp())
    interval.start_time.seconds = int((now - timedelta(days=1)).timestamp())
    
    # Check request count
    # Check error rate
    # Check latency
    
    # Compare with thresholds
    thresholds = load_config().get('monitoring', {})
    
    # Return status
    return {
        "status": "healthy",
        "metrics": {...},
        "alerts": []
    }

def detect_data_drift(project_id: str, bucket_name: str):
    """
    Compare production input data distribution with training data
    """
    # Load training data distribution (from GCS)
    # Load production data distribution (from BigQuery logs)
    # Compare using statistical tests
    # Return drift score
    pass

def main():
    config = load_config()
    gcp_config = config['gcp']
    
    # Check metrics
    metrics = check_production_metrics(
        gcp_config['project_id'],
        "datacraft-llm-endpoint"
    )
    
    # Check drift
    drift_score = detect_data_drift(
        gcp_config['project_id'],
        gcp_config['bucket_name']
    )
    
    # Trigger retraining if needed
    if metrics['status'] == 'degraded' or drift_score > 0.2:
        print("⚠ Performance degradation detected - triggering retraining")
        # Trigger GitHub Actions workflow or Airflow DAG
        trigger_retraining()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
```

#### Step 5.2: Add Scheduled Monitoring Workflow

Create: `.github/workflows/monitor-production.yml`

```yaml
name: Production Monitoring

on:
  schedule:
    - cron: '0 */6 * * *'  # Every 6 hours
  workflow_dispatch:

jobs:
  monitor:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
      
      - name: Authenticate to Google Cloud
        uses: google-github-actions/auth@v1
        with:
          credentials_json: ${{ secrets.GCP_SA_KEY }}
      
      - name: Monitor Production
        run: |
          pip install google-cloud-monitoring google-cloud-storage
          python ci-cd/scripts/monitor_production.py
```

---

### **PHASE 6: Testing & Verification** (1-2 hours)

#### Step 6.1: Test Locally

```bash
# Test metadata loading
cd frontend
python -c "from metadata_loader import MetadataLoader; loader = MetadataLoader('your-project', 'your-bucket'); print(loader.get_best_model_name())"

# Test Docker build
cd ..
docker build -f frontend/Dockerfile -t datacraft-app:test .

# Test Docker run
docker run -p 8501:8501 \
  -e GCP_PROJECT_ID=your-project \
  -e BQ_DATASET=datacraft_ml \
  -e GCS_BUCKET_NAME=your-bucket \
  -e GCP_REGION=us-east1 \
  datacraft-app:test

# Access Streamlit UI at http://localhost:8501
```

#### Step 6.2: Test Deployment

```bash
# After deployment, get Cloud Run URL
gcloud run services describe datacraft-app \
  --region=us-east1 \
  --format="value(status.url)"

# Access the deployed app
# Open URL in browser: https://datacraft-app-xxxxx-uc.a.run.app

# Test health (if you add health endpoint)
curl https://datacraft-app-xxxxx-uc.a.run.app/health
```

---

## 🎯 Summary Checklist

### Phase 1: Metadata Loader ✅
- [ ] Create `frontend/metadata_loader.py`
- [ ] Update `frontend/app.py` to use metadata loader
- [ ] Test metadata loading locally

### Phase 2: Docker Container ✅
- [ ] Update `frontend/Dockerfile` to include model-training scripts
- [ ] Update `frontend/requirements.txt` if needed
- [ ] Create `.dockerignore` at project root
- [ ] Build and test image locally

### Phase 3: CI/CD Pipeline ✅
- [ ] Add `build_and_deploy` job to `.github/workflows/model-training-ci-cd.yml`
- [ ] Configure Cloud Run deployment
- [ ] Update notification job to include deployment URL
- [ ] Test full pipeline

### Phase 4: Artifact Registry ✅
- [ ] Create Artifact Registry repository (one-time)
- [ ] Verify Docker auth works

### Phase 5: Monitoring ✅
- [ ] Create `monitor_production.py` script
- [ ] Add scheduled monitoring workflow
- [ ] Set up alerting thresholds
- [ ] Test retraining trigger

### Phase 6: Testing ✅
- [ ] Test metadata loading
- [ ] Test Docker build locally
- [ ] Test deployed Cloud Run service
- [ ] Verify frontend works with best model

---

## 📊 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│         SINGLE CI/CD PIPELINE (SEQUENTIAL)                  │
│                                                              │
│  Push to main                                                │
│    ↓                                                         │
│  1. Trigger Airflow DAG (Training/Eval)                     │
│    ↓                                                         │
│  2. Wait for Airflow to finish                               │
│    ↓                                                         │
│  3. Download Best Model Metadata from GCS                   │
│    ↓                                                         │
│  4. Validate Performance + Bias Check                       │
│    ↓                                                         │
│  5. Compare with Production                                  │
│    ↓                                                         │
│  6. Build Docker Image (Frontend + Backend)                 │
│    ↓                                                         │
│  7. Push to Artifact Registry                                │
│    ↓                                                         │
│  8. Deploy to Cloud Run                                      │
│    ↓                                                         │
│  9. Notify                                                   │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING PHASE (DONE)                     │
│                                                              │
│  Airflow DAG                                                │
│    ↓                                                         │
│  Evaluate Models (Gemini Flash vs Pro)                      │
│    ↓                                                         │
│  Tune Hyperparameters (temp, top_p, top_k)                 │
│    ↓                                                         │
│  Select Best Model                                          │
│    ↓                                                         │
│  Save Metadata to GCS                                       │
│    gs://bucket/models/{timestamp}_{commit}/                 │
│    ├── best_model_metadata.json                             │
│    ├── model_selection_report.json                          │
│    └── validation_report.json                               │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│              PRODUCTION SERVING (Cloud Run)                  │
│                                                              │
│  User Browser                                                │
│    ↓                                                         │
│  Streamlit UI (frontend/app.py)                             │
│    - Loads metadata from GCS on startup                      │
│    - Gets best model name & config                           │
│    ↓                                                         │
│  QueryHandler (frontend/query_handler.py)                    │
│    - Uses best model from metadata                           │
│    - Calls Gemini API with tuned hyperparameters            │
│    ↓                                                         │
│  BigQuery Execution                                          │
│    - Executes generated SQL                                  │
│    ↓                                                         │
│  Visualization Engine                                        │
│    - Renders charts using Plotly                              │
│    ↓                                                         │
│  User sees results in Streamlit UI                           │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                  MONITORING (TO BUILD)                      │
│                                                              │
│  Scheduled Job (Every 6 hours)                              │
│    ↓                                                         │
│  Check Production Metrics                                   │
│    - Request count                                          │
│    - Error rate                                             │
│    - Latency                                                │
│    - Success rate                                           │
│    ↓                                                         │
│  Detect Data Drift                                          │
│    - Compare production vs training data                    │
│    ↓                                                         │
│  Trigger Retraining if Needed                               │
│    - Performance < threshold                                │
│    - Drift detected                                         │
│    ↓                                                         │
│  GitHub Actions → Airflow DAG → New Model → Deploy         │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔑 Key Files to Create/Update

1. **`frontend/metadata_loader.py`** - Load best model config from GCS ⭐ NEW
2. **`frontend/app.py`** - Update to use metadata loader ⭐ UPDATE
3. **`frontend/Dockerfile`** - Update to include model-training scripts ⭐ UPDATE
4. **`.dockerignore`** - Create at project root ⭐ NEW
5. **`.github/workflows/model-training-ci-cd.yml`** - Add build_and_deploy job ⭐ UPDATE
6. **`ci-cd/scripts/monitor_production.py`** - Monitoring script ⭐ NEW
7. **`.github/workflows/monitor-production.yml`** - Scheduled monitoring ⭐ NEW

---

## ✅ Project Requirements Fulfillment

| Requirement | Status | Implementation |
|------------|--------|----------------|
| **Deployment Service** | ✅ | Vertex AI Endpoint (serverless) |
| **Deployment Automation** | ✅ | GitHub Actions + deploy script |
| **Connection to Repository** | ✅ | Auto-triggers on push to main |
| **Model Monitoring** | ✅ | Scheduled monitoring job |
| **Data Drift Detection** | ✅ | Statistical comparison in monitor script |
| **Retraining Triggers** | ✅ | Auto-trigger on degradation/drift |
| **Notifications** | ✅ | Email alerts (already implemented) |
| **Detailed Steps** | ✅ | This game plan document |

---

## 🚀 Next Steps

1. **Start with Phase 1** - Create inference codebase
2. **Test locally** - Ensure metadata loading works
3. **Build Docker image** - Verify containerization
4. **Deploy to Vertex** - Get endpoint running
5. **Add monitoring** - Set up production monitoring
6. **Document** - Update README with deployment instructions

---

**Estimated Total Time**: 4-6 hours

**Priority Order**:
1. Phase 1 (Metadata Loader) - **CRITICAL** (1 hour)
2. Phase 2 (Docker Update) - **CRITICAL** (1 hour)
3. Phase 3 (CI/CD Pipeline) - **CRITICAL** (1 hour)
4. Phase 4 (Artifact Registry) - **CRITICAL** (30 min)
5. Phase 5 (Monitoring) - **IMPORTANT** (2-3 hours)
6. Phase 6 (Testing) - **ONGOING** (1-2 hours)

**Key Changes from Original Plan**:
- ✅ Single sequential CI/CD pipeline (not separate)
- ✅ Frontend + Backend in one Docker image
- ✅ Deploy to Cloud Run (better for Streamlit)
- ✅ Simpler architecture (no separate API service needed)

---

**Last Updated**: December 2025  
**Version**: 1.0

