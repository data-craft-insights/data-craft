# Complete Deployment Instructions

This guide walks you through running the entire deployment workflow from start to finish.

---

## Prerequisites Checklist

- [ ] GCP project created and configured (see `GCP_SETUP_GUIDE.md`)
- [ ] All APIs enabled
- [ ] Service account created with keys
- [ ] GCS bucket created
- [ ] BigQuery dataset created
- [ ] Artifact Registry repository created
- [ ] GitHub secrets configured
- [ ] Airflow running and accessible
- [ ] Local environment set up (Docker, Python 3.9+)

---

## Part 1: Local Testing (Before CI/CD)

### Step 1: Test Metadata Loader

```bash
cd frontend

# Set environment variables
export GOOGLE_APPLICATION_CREDENTIALS="../gcp/service-account.json"
export GCP_PROJECT_ID="datacraft-479223"  # Your project ID
export GCS_BUCKET_NAME="model-datacraft-sanskar"  # Your bucket name

# Test metadata loader
python -c "
from metadata_loader import MetadataLoader
loader = MetadataLoader('$GCP_PROJECT_ID', '$GCS_BUCKET_NAME')
print('Best model:', loader.get_best_model_name())
print('Model info:', loader.get_model_info())
"
```

**Expected Output**:
```
Best model: gemini-2.5-flash
Model info: {'selected_model': 'gemini-2.5-flash', 'composite_score': 89.5, ...}
```

### Step 2: Test Docker Build Locally

```bash
# From project root
cd /Users/sanskar/Personal/mlops-project

# Build Docker image
docker build -f frontend/Dockerfile -t datacraft-app:local .

# Test run (if you have metadata in GCS)
docker run -p 8501:8501 \
  -e GCP_PROJECT_ID=datacraft-479223 \
  -e BQ_DATASET=datacraft_ml \
  -e GCS_BUCKET_NAME=model-datacraft-sanskar \
  -e GCP_REGION=us-east1 \
  -v $(pwd)/gcp/service-account.json:/app/gcp/service-account.json:ro \
  -e GOOGLE_APPLICATION_CREDENTIALS=/app/gcp/service-account.json \
  datacraft-app:local
```

**Access**: Open http://localhost:8501 in browser

---

## Part 2: First Model Training Run

### Step 1: Trigger Airflow DAG

1. **Start Airflow** (if not running):
   ```bash
   docker-compose up -d
   ```

2. **Access Airflow UI**: http://localhost:8080
   - Username: `admin`
   - Password: `admin`

3. **Trigger DAG**: `model_pipeline_with_evaluation`
   - Click on DAG name
   - Click "Trigger DAG" button
   - Wait for completion (5-10 minutes)

4. **Verify Outputs**:
   ```bash
   # Check local outputs
   ls -la outputs/best-model-responses/
   
   # Check GCS uploads
   gsutil ls gs://model-datacraft-sanskar/best_model_responses/
   ```

### Step 2: Verify Metadata is Saved

```bash
# Check GCS for metadata
gsutil ls -r gs://model-datacraft-sanskar/models/ || echo "No models/ directory yet"
gsutil ls -r gs://model-datacraft-sanskar/best_model_responses/ | grep metadata
```

---

## Part 3: Run CI/CD Pipeline

### Option A: Automatic Trigger (Recommended)

1. **Push to main branch**:
   ```bash
   git add .
   git commit -m "Add deployment pipeline"
   git push origin main
   ```

2. **Monitor in GitHub Actions**:
   - Go to: `https://github.com/YOUR_USERNAME/YOUR_REPO/actions`
   - Click on the running workflow
   - Watch each job complete

### Option B: Manual Trigger

1. **Go to GitHub Actions**:
   - Navigate to your repository
   - Click "Actions" tab
   - Select "Model Training CI/CD Pipeline"
   - Click "Run workflow"
   - Select mode: `prod`
   - Click "Run workflow"

### Pipeline Flow

The pipeline will execute in this order:

```
1. trigger_dag          → Triggers Airflow DAG
2. download_outputs     → Downloads model artifacts from GCS
3. validate             → Validates model performance
4. check_bias           → Checks model bias
5. compare_and_deploy   → Compares with production, pushes to registry
6. build_and_deploy     → Builds Docker image, deploys to Cloud Run
7. notify               → Sends email notification
```

**Expected Duration**: 15-20 minutes

---

## Part 4: Verify Deployment

### Step 1: Check Cloud Run Service

```bash
# Get service URL
gcloud run services describe datacraft-app \
  --region=us-east1 \
  --format="value(status.url)" \
  --project=datacraft-479223

# Or list all services
gcloud run services list --region=us-east1 --project=datacraft-479223
```

### Step 2: Access Deployed Application

1. **Get URL from email notification** (sent after deployment)
2. **Or get from Cloud Console**:
   - Go to: https://console.cloud.google.com/run
   - Click on `datacraft-app`
   - Copy the URL

3. **Open in browser**: `https://datacraft-app-xxxxx-uc.a.run.app`

### Step 3: Test Application

1. **Select a dataset** from sidebar
2. **Enter a query**: "What are total sales by region?"
3. **Click "Generate"**
4. **Verify**:
   - SQL query is generated
   - Results are displayed
   - Visualization is rendered

---

## Part 5: Monitor Production

### Check Logs

```bash
# View Cloud Run logs
gcloud run services logs read datacraft-app \
  --region=us-east1 \
  --project=datacraft-479223 \
  --limit=50
```

### Check Metrics

1. **Go to Cloud Console**: https://console.cloud.google.com/run
2. **Click on `datacraft-app`**
3. **View metrics**:
   - Request count
   - Latency
   - Error rate
   - CPU/Memory usage

---

## Troubleshooting

### Issue: Pipeline fails at "Trigger DAG"

**Symptoms**: Cannot connect to Airflow

**Solutions**:
1. Verify Airflow is running: `docker-compose ps`
2. Check Airflow URL in GitHub secrets
3. For production, use Cloud Composer or expose Airflow publicly

### Issue: "No metadata found in GCS"

**Symptoms**: Metadata loader fails

**Solutions**:
1. Verify DAG has run and completed
2. Check GCS bucket: `gsutil ls gs://your-bucket/models/`
3. Verify bucket name in config matches actual bucket

### Issue: Docker build fails

**Symptoms**: Build errors in CI/CD

**Solutions**:
1. Test build locally first: `docker build -f frontend/Dockerfile .`
2. Check Dockerfile paths are correct
3. Verify all required files exist

### Issue: Cloud Run deployment fails

**Symptoms**: Deployment job fails

**Solutions**:
1. Check service account has `run.admin` role
2. Verify Artifact Registry image exists
3. Check Cloud Run API is enabled
4. Review logs: `gcloud run services describe datacraft-app --region=us-east1`

### Issue: Application doesn't load metadata

**Symptoms**: Frontend shows default model

**Solutions**:
1. Check Cloud Run environment variables are set
2. Verify service account has `storage.objectViewer` role
3. Check application logs for errors
4. Verify metadata exists in GCS

---

## Updating the Application

### After Model Retraining

When a new model is trained:

1. **Pipeline automatically**:
   - Trains new model
   - Validates performance
   - Pushes to registry
   - Builds new Docker image
   - Deploys to Cloud Run

2. **Frontend automatically**:
   - Loads latest metadata on startup
   - Uses best model configuration

### Manual Update

To manually trigger update:

```bash
# Trigger workflow manually via GitHub Actions UI
# Or push changes to main branch
```

---

## Rollback Procedure

If deployment fails or model performs poorly:

### Option 1: Rollback via CI/CD

The pipeline has rollback protection. If new model is worse, it won't deploy.

### Option 2: Manual Rollback

```bash
# Deploy previous version
gcloud run deploy datacraft-app \
  --image us-east1-docker.pkg.dev/$PROJECT_ID/mlops-models/datacraft-app:PREVIOUS_COMMIT_SHA \
  --region us-east1 \
  --project $PROJECT_ID
```

### Option 3: Revert Code

```bash
# Revert to previous commit
git revert HEAD
git push origin main
```

---

## Next Steps

1. **Set up monitoring** (see `DEPLOYMENT_GAMEPLAN.md` Phase 5)
2. **Configure alerts** for production issues
3. **Set up scheduled retraining** (weekly/monthly)
4. **Add A/B testing** for gradual rollouts
5. **Implement data drift detection**

---

## Quick Reference Commands

```bash
# View pipeline status
gh run list  # If GitHub CLI installed

# Check Cloud Run service
gcloud run services describe datacraft-app --region=us-east1

# View logs
gcloud run services logs read datacraft-app --region=us-east1 --limit=100

# Check GCS metadata
gsutil ls gs://model-datacraft-sanskar/models/

# List Artifact Registry images
gcloud artifacts docker images list us-east1-docker.pkg.dev/datacraft-479223/mlops-models

# Test locally
docker build -f frontend/Dockerfile -t datacraft-app:test .
docker run -p 8501:8501 datacraft-app:test
```

---

**Last Updated**: December 2025

