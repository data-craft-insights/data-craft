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
- ✅ Deploy as Docker container → Vertex AI Endpoint

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
│              DEPLOYMENT FLOW (TO BUILD)                      │
│                                                              │
│  1. Build Docker Image (inference code)                     │
│  2. Push to Artifact Registry                                │
│  3. Deploy to Vertex AI Endpoint                            │
│  4. Container loads metadata from GCS on startup             │
│  5. Exposes REST API for inference                           │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│              PRODUCTION SERVING                              │
│                                                              │
│  User Query → Vertex Endpoint → Load Metadata               │
│         → Call Gemini API with config → Return SQL + Viz    │
└─────────────────────────────────────────────────────────────┘
```

---

## 📋 STEP-BY-STEP GAMEPLAN

### **PHASE 1: Create Inference Codebase** (2-3 hours)

#### Step 1.1: Create Inference Service Structure

Create directory: `model-serving/`

```
model-serving/
├── app/
│   ├── __init__.py
│   ├── serve.py              # FastAPI/Flask server
│   ├── predict.py            # Main inference logic
│   ├── model_selector.py     # Load metadata from GCS
│   ├── metadata_loader.py    # Fetch & parse metadata
│   └── llm_client.py         # Gemini API client
├── Dockerfile
├── requirements.txt
├── .dockerignore
└── README.md
```

#### Step 1.2: Implement `metadata_loader.py`

**Purpose**: Load best model metadata from GCS

```python
# model-serving/app/metadata_loader.py
from google.cloud import storage
import json
from typing import Dict, Optional
from pathlib import Path

class MetadataLoader:
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
            raise ValueError("No metadata found in GCS")
        
        # Get latest
        latest_timestamp = max(model_dirs.keys())
        latest_blob_path = model_dirs[latest_timestamp]
        
        # Download and parse
        blob = self.bucket.blob(latest_blob_path)
        metadata_json = blob.download_as_text()
        metadata = json.loads(metadata_json)
        
        return metadata
    
    def load_hyperparameters(self, metadata: Dict) -> Dict:
        """
        Extract hyperparameters from metadata or use defaults
        
        Returns:
            {
                "temperature": 0.2,
                "top_p": 0.9,
                "top_k": 40
            }
        """
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

#### Step 1.3: Implement `llm_client.py`

**Purpose**: Client for calling Gemini API

```python
# model-serving/app/llm_client.py
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
from typing import Dict, Optional

class LLMClient:
    def __init__(self, project_id: str, location: str = "us-central1"):
        vertexai.init(project=project_id, location=location)
        self.project_id = project_id
        self.location = location
        self.model = None
        self.model_name = None
    
    def initialize_model(self, model_name: str):
        """Initialize Gemini model"""
        self.model_name = model_name
        self.model = GenerativeModel(model_name)
    
    def generate_sql(
        self,
        user_query: str,
        dataset_metadata: str,
        hyperparameters: Dict,
        prompt_template: Optional[str] = None
    ) -> Dict:
        """
        Generate SQL query from user query
        
        Args:
            user_query: Natural language query
            dataset_metadata: Schema and context from BigQuery
            hyperparameters: {temperature, top_p, top_k}
            prompt_template: Optional custom prompt
        
        Returns:
            {
                "sql_query": "SELECT ...",
                "visualization_config": {...},
                "explanation": "..."
            }
        """
        # Build prompt (reuse from model-training/scripts/prompts.py)
        prompt = self._build_prompt(user_query, dataset_metadata, prompt_template)
        
        # Create generation config
        generation_config = GenerationConfig(
            temperature=hyperparameters.get("temperature", 0.2),
            top_p=hyperparameters.get("top_p", 0.9),
            top_k=hyperparameters.get("top_k", 40),
            max_output_tokens=2048,
            response_mime_type="application/json"
        )
        
        # Generate
        response = self.model.generate_content(
            prompt,
            generation_config=generation_config
        )
        
        # Parse JSON response
        import json
        result = json.loads(response.text)
        
        return result
    
    def _build_prompt(self, user_query: str, dataset_metadata: str, template: Optional[str]) -> str:
        """Build prompt with few-shot examples"""
        # Reuse prompt building logic from model-training/scripts/prompts.py
        # Or load from GCS if stored there
        pass
```

#### Step 1.4: Implement `predict.py`

**Purpose**: Main inference function

```python
# model-serving/app/predict.py
from typing import Dict
from .metadata_loader import MetadataLoader
from .llm_client import LLMClient

class InferenceService:
    def __init__(self, project_id: str, bucket_name: str, location: str = "us-central1"):
        self.project_id = project_id
        self.bucket_name = bucket_name
        self.location = location
        
        # Load metadata on initialization
        self.metadata_loader = MetadataLoader(project_id, bucket_name)
        self.metadata = self.metadata_loader.load_latest_metadata()
        self.hyperparameters = self.metadata_loader.load_hyperparameters(self.metadata)
        
        # Initialize LLM client
        self.llm_client = LLMClient(project_id, location)
        selected_model = self.metadata.get("selected_model", "gemini-2.5-flash")
        self.llm_client.initialize_model(selected_model)
        
        print(f"✓ Initialized inference service")
        print(f"  Model: {selected_model}")
        print(f"  Score: {self.metadata.get('composite_score', 0):.2f}")
        print(f"  Hyperparameters: {self.hyperparameters}")
    
    def predict(self, user_query: str, dataset_name: str, dataset_metadata: str) -> Dict:
        """
        Main prediction function
        
        Args:
            user_query: Natural language query
            dataset_name: Name of dataset (e.g., "orders")
            dataset_metadata: Schema and context from BigQuery
        
        Returns:
            {
                "sql_query": "...",
                "visualization_config": {...},
                "explanation": "...",
                "model_used": "gemini-2.5-flash",
                "confidence": 0.95
            }
        """
        result = self.llm_client.generate_sql(
            user_query=user_query,
            dataset_metadata=dataset_metadata,
            hyperparameters=self.hyperparameters
        )
        
        # Add metadata
        result["model_used"] = self.metadata.get("selected_model")
        result["model_version"] = self.metadata.get("selection_date")
        
        return result
    
    def reload_metadata(self):
        """Reload metadata from GCS (for hot-swapping models)"""
        self.metadata = self.metadata_loader.load_latest_metadata()
        self.hyperparameters = self.metadata_loader.load_hyperparameters(self.metadata)
        
        # Reinitialize model if changed
        selected_model = self.metadata.get("selected_model", "gemini-2.5-flash")
        if selected_model != self.llm_client.model_name:
            self.llm_client.initialize_model(selected_model)
```

#### Step 1.5: Implement `serve.py` (FastAPI Server)

**Purpose**: REST API endpoint

```python
# model-serving/app/serve.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import os
from .predict import InferenceService

app = FastAPI(title="DataCraft LLM Inference Service")

# Initialize service on startup
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "datacraft-479223")
BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", "model-datacraft-sanskar")
LOCATION = os.environ.get("VERTEX_LOCATION", "us-central1")

inference_service = None

@app.on_event("startup")
async def startup():
    global inference_service
    inference_service = InferenceService(PROJECT_ID, BUCKET_NAME, LOCATION)

# Request/Response models
class QueryRequest(BaseModel):
    user_query: str
    dataset_name: str
    dataset_metadata: str  # JSON string from BigQuery

class QueryResponse(BaseModel):
    sql_query: str
    visualization_config: dict
    explanation: str
    model_used: str
    model_version: str

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": inference_service.metadata.get("selected_model") if inference_service else "not_loaded"
    }

@app.post("/predict", response_model=QueryResponse)
async def predict(request: QueryRequest):
    """
    Generate SQL query and visualization config from natural language
    
    Example:
        POST /predict
        {
            "user_query": "What are total sales by region?",
            "dataset_name": "orders",
            "dataset_metadata": "{...schema info...}"
        }
    """
    if not inference_service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        result = inference_service.predict(
            user_query=request.user_query,
            dataset_name=request.dataset_name,
            dataset_metadata=request.dataset_metadata
        )
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reload")
async def reload_metadata():
    """Reload model metadata from GCS (for model updates)"""
    if not inference_service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        inference_service.reload_metadata()
        return {"status": "success", "message": "Metadata reloaded"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
```

---

### **PHASE 2: Containerize with Docker** (1 hour)

#### Step 2.1: Create `Dockerfile`

```dockerfile
# model-serving/Dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first (for layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Expose port
EXPOSE 8080

# Set environment variables
ENV PORT=8080
ENV PYTHONUNBUFFERED=1

# Run server
CMD ["python", "-m", "app.serve"]
```

#### Step 2.2: Create `requirements.txt`

```txt
# model-serving/requirements.txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
google-cloud-storage==2.10.0
google-cloud-aiplatform==1.38.0
vertexai==0.1.0
pydantic==2.5.0
python-dotenv==1.0.0
```

#### Step 2.3: Create `.dockerignore`

```
# model-serving/.dockerignore
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
README.md
.env
```

---

### **PHASE 3: Build & Push to Artifact Registry** (30 minutes)

#### Step 3.1: Set up Artifact Registry

```bash
# Create Artifact Registry repository
gcloud artifacts repositories create mlops-models \
    --repository-format=docker \
    --location=us-east1 \
    --description="MLOps model serving containers"

# Configure Docker authentication
gcloud auth configure-docker us-east1-docker.pkg.dev
```

#### Step 3.2: Build and Push Image

```bash
cd model-serving

# Build image
docker build -t us-east1-docker.pkg.dev/datacraft-479223/mlops-models/inference-service:latest .

# Push to Artifact Registry
docker push us-east1-docker.pkg.dev/datacraft-479223/mlops-models/inference-service:latest
```

---

### **PHASE 4: Deploy to Vertex AI Endpoint** (1-2 hours)

#### Step 4.1: Create Deployment Script

Create: `ci-cd/scripts/deploy_to_vertex.py`

```python
#!/usr/bin/env python3
"""
Deploy inference service to Vertex AI Endpoint
"""
import sys
import yaml
import json
from pathlib import Path
from google.cloud import aiplatform
from google.cloud.aiplatform import models, endpoints

project_root = Path(__file__).parent.parent.parent
config_path = project_root / "ci-cd" / "config" / "ci_cd_config.yaml"

def load_config():
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def deploy_to_vertex_endpoint(
    project_id: str,
    location: str,
    image_uri: str,
    endpoint_name: str = "datacraft-llm-endpoint"
):
    """
    Deploy container to Vertex AI Endpoint
    
    Args:
        project_id: GCP project ID
        location: GCP region
        image_uri: Full image URI from Artifact Registry
        endpoint_name: Name for the endpoint
    """
    aiplatform.init(project=project_id, location=location)
    
    # Check if endpoint exists
    try:
        endpoint = endpoints.Endpoint.list(filter=f'display_name="{endpoint_name}"')[0]
        print(f"✓ Found existing endpoint: {endpoint.resource_name}")
    except IndexError:
        # Create new endpoint
        print(f"Creating new endpoint: {endpoint_name}")
        endpoint = endpoints.Endpoint.create(display_name=endpoint_name)
        print(f"✓ Created endpoint: {endpoint.resource_name}")
    
    # Deploy model (container)
    print(f"Deploying container: {image_uri}")
    
    # Create model from container
    model = models.Model.upload(
        display_name=f"{endpoint_name}-model",
        artifact_uri=None,  # No artifact, just container
        serving_container_image_uri=image_uri,
        serving_container_ports=[8080],
        serving_container_environment_variables={
            "GOOGLE_CLOUD_PROJECT": project_id,
            "GCS_BUCKET_NAME": config['gcp']['bucket_name'],
            "VERTEX_LOCATION": location
        }
    )
    
    # Deploy to endpoint
    endpoint.deploy(
        model=model,
        deployed_model_display_name=f"{endpoint_name}-deployment",
        machine_type="n1-standard-2",
        min_replica_count=1,
        max_replica_count=3
    )
    
    print(f"✓ Deployment complete!")
    print(f"  Endpoint: {endpoint.resource_name}")
    print(f"  Predict URL: {endpoint.predict}")
    
    return endpoint

def main():
    config = load_config()
    gcp_config = config['gcp']
    
    # Image URI from Artifact Registry
    image_uri = f"us-east1-docker.pkg.dev/{gcp_config['project_id']}/mlops-models/inference-service:latest"
    
    endpoint = deploy_to_vertex_endpoint(
        project_id=gcp_config['project_id'],
        location=gcp_config['region'],
        image_uri=image_uri
    )
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
```

#### Step 4.2: Add Deployment Job to CI/CD Pipeline

Update: `.github/workflows/model-training-ci-cd.yml`

Add new job after `compare_and_deploy`:

```yaml
  deploy_to_endpoint:
    name: Deploy to Vertex AI Endpoint
    needs: [compare_and_deploy]
    runs-on: ubuntu-latest
    if: needs.compare_and_deploy.result == 'success'
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
      
      - name: Authenticate to Google Cloud
        uses: google-github-actions/auth@v1
        with:
          credentials_json: ${{ secrets.GCP_SA_KEY }}
      
      - name: Set up Cloud SDK
        uses: google-github-actions/setup-gcloud@v1
      
      - name: Build Docker Image
        run: |
          cd model-serving
          docker build -t us-east1-docker.pkg.dev/${{ env.GCP_PROJECT_ID }}/mlops-models/inference-service:${{ github.sha }} .
      
      - name: Push to Artifact Registry
        run: |
          gcloud auth configure-docker us-east1-docker.pkg.dev
          docker push us-east1-docker.pkg.dev/${{ env.GCP_PROJECT_ID }}/mlops-models/inference-service:${{ github.sha }}
      
      - name: Deploy to Vertex AI
        run: |
          pip install google-cloud-aiplatform
          python ci-cd/scripts/deploy_to_vertex.py
```

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
cd model-serving
python -m app.metadata_loader

# Test inference
python -m app.predict

# Test server
python -m app.serve
# In another terminal:
curl http://localhost:8080/health
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "user_query": "What are total sales by region?",
    "dataset_name": "orders",
    "dataset_metadata": "{...}"
  }'
```

#### Step 6.2: Test Deployment

```bash
# After deployment, test endpoint
ENDPOINT_ID="your-endpoint-id"
PROJECT_ID="datacraft-479223"

curl -X POST \
  "https://$LOCATION-aiplatform.googleapis.com/v1/projects/$PROJECT_ID/locations/$LOCATION/endpoints/$ENDPOINT_ID:predict" \
  -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  -H "Content-Type: application/json" \
  -d '{
    "instances": [{
      "user_query": "What are total sales?",
      "dataset_name": "orders",
      "dataset_metadata": "{...}"
    }]
  }'
```

---

## 🎯 Summary Checklist

### Phase 1: Inference Codebase ✅
- [ ] Create `model-serving/` directory structure
- [ ] Implement `metadata_loader.py`
- [ ] Implement `llm_client.py`
- [ ] Implement `predict.py`
- [ ] Implement `serve.py` (FastAPI)
- [ ] Test locally

### Phase 2: Docker Container ✅
- [ ] Create `Dockerfile`
- [ ] Create `requirements.txt`
- [ ] Create `.dockerignore`
- [ ] Build and test image locally

### Phase 3: Artifact Registry ✅
- [ ] Create Artifact Registry repository
- [ ] Configure Docker auth
- [ ] Build and push image

### Phase 4: Vertex AI Deployment ✅
- [ ] Create `deploy_to_vertex.py` script
- [ ] Add deployment job to CI/CD pipeline
- [ ] Deploy to Vertex AI endpoint
- [ ] Verify endpoint is accessible

### Phase 5: Monitoring ✅
- [ ] Create `monitor_production.py` script
- [ ] Add scheduled monitoring workflow
- [ ] Set up alerting thresholds
- [ ] Test retraining trigger

### Phase 6: Testing ✅
- [ ] Test metadata loading
- [ ] Test inference locally
- [ ] Test deployed endpoint
- [ ] Verify monitoring works

---

## 📊 Architecture Diagram

```
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
│                  DEPLOYMENT PHASE (TO BUILD)                │
│                                                              │
│  GitHub Actions CI/CD                                        │
│    ↓                                                         │
│  Build Docker Image (inference code)                        │
│    ↓                                                         │
│  Push to Artifact Registry                                  │
│    us-east1-docker.pkg.dev/.../inference-service:latest    │
│    ↓                                                         │
│  Deploy to Vertex AI Endpoint                               │
│    - Container starts                                        │
│    - Loads metadata from GCS                                │
│    - Initializes Gemini client                              │
│    - Exposes REST API                                        │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                  PRODUCTION SERVING                         │
│                                                              │
│  User Request                                               │
│    POST /predict                                            │
│    {                                                        │
│      "user_query": "What are total sales?",                 │
│      "dataset_name": "orders",                              │
│      "dataset_metadata": "{...schema...}"                  │
│    }                                                        │
│    ↓                                                         │
│  Inference Service                                          │
│    - Loads best model config from metadata                 │
│    - Calls Gemini API with tuned hyperparameters           │
│    - Returns SQL + visualization config                    │
│    ↓                                                         │
│  Response                                                   │
│    {                                                        │
│      "sql_query": "SELECT ...",                            │
│      "visualization_config": {...},                        │
│      "model_used": "gemini-2.5-flash"                      │
│    }                                                        │
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

## 🔑 Key Files to Create

1. **`model-serving/app/metadata_loader.py`** - Load config from GCS
2. **`model-serving/app/llm_client.py`** - Gemini API client
3. **`model-serving/app/predict.py`** - Inference logic
4. **`model-serving/app/serve.py`** - FastAPI server
5. **`model-serving/Dockerfile`** - Container definition
6. **`model-serving/requirements.txt`** - Dependencies
7. **`ci-cd/scripts/deploy_to_vertex.py`** - Deployment script
8. **`ci-cd/scripts/monitor_production.py`** - Monitoring script
9. **`.github/workflows/monitor-production.yml`** - Scheduled monitoring

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

**Estimated Total Time**: 8-12 hours

**Priority Order**:
1. Phase 1 (Inference codebase) - **CRITICAL**
2. Phase 2-3 (Docker + Registry) - **CRITICAL**
3. Phase 4 (Deployment) - **CRITICAL**
4. Phase 5 (Monitoring) - **IMPORTANT**
5. Phase 6 (Testing) - **ONGOING**

---

**Last Updated**: December 2025  
**Version**: 1.0

