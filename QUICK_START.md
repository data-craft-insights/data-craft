# Quick Start Guide - Complete Deployment

This is your **quick reference** for running the entire deployment workflow.

---

## 🚀 Complete Workflow Overview

```
1. Setup GCP (One-time)          → See GCP_SETUP_GUIDE.md
2. Configure GitHub Secrets       → See below
3. Run First Training             → Trigger Airflow DAG
4. Push to GitHub                 → Triggers CI/CD automatically
5. Access Deployed App            → Get URL from email/Cloud Console
```

---

## Step-by-Step Execution

### Step 1: GCP Setup (One-Time, 30 minutes)

**Follow**: `GCP_SETUP_GUIDE.md`

**Quick commands**:
```bash
# Run setup script (update PROJECT_ID and BUCKET_NAME first)
chmod +x setup_gcp.sh
./setup_gcp.sh

# Or follow manual steps in GCP_SETUP_GUIDE.md
```

**What you'll get**:
- ✅ Service account key: `gcp-key.json`
- ✅ GCS bucket created
- ✅ BigQuery dataset created
- ✅ Artifact Registry created

---

### Step 2: Configure GitHub Secrets (5 minutes)

Go to: **GitHub Repo → Settings → Secrets and variables → Actions**

**Add these secrets**:

| Secret Name | Value | How to Get |
|------------|-------|------------|
| `GCP_SA_KEY` | Contents of `gcp-key.json` | `cat gcp-key.json` (copy entire JSON) |
| `GCS_BUCKET_NAME` | Your bucket name | e.g., `model-datacraft-sanskar` |
| `EMAIL_SMTP_PASSWORD` | Gmail App Password | Google Account → Security → App Passwords |
| `AIRFLOW_URL` | Your Airflow URL | e.g., `http://localhost:8080` or Cloud Composer URL |
| `AIRFLOW_USERNAME` | Airflow username | e.g., `admin` |
| `AIRFLOW_PASSWORD` | Airflow password | e.g., `admin` |

**Optional**:
- `EMAIL_SMTP_USER` - Your email (defaults to config file)

---

### Step 3: Update Configuration Files (2 minutes)

**File**: `ci-cd/config/ci_cd_config.yaml`

```yaml
gcp:
  project_id: "datacraft-479223"  # YOUR PROJECT ID
  region: "us-east1"              # YOUR REGION
  dataset_id: "datacraft_ml"     # YOUR DATASET
  bucket_name: "model-datacraft-sanskar"  # YOUR BUCKET

notifications:
  email:
    from_email: "your-email@gmail.com"  # YOUR EMAIL
    to_email: "your-email@gmail.com"    # YOUR EMAIL
```

**File**: `.github/workflows/model-training-ci-cd.yml`

```yaml
env:
  PYTHON_VERSION: '3.9'
  GCP_PROJECT_ID: 'datacraft-479223'  # YOUR PROJECT ID
```

---

### Step 4: Run First Training (10 minutes)

**Option A: Via Airflow UI** (Recommended for first time)

1. Start Airflow:
   ```bash
   docker-compose up -d
   ```

2. Access: http://localhost:8080
   - Username: `admin`
   - Password: `admin`

3. Trigger DAG: `model_pipeline_with_evaluation`
   - Wait for completion (5-10 minutes)

4. Verify outputs:
   ```bash
   ls outputs/best-model-responses/
   gsutil ls gs://your-bucket/best_model_responses/
   ```

**Option B: Via CI/CD** (After first run)

Just push to main - pipeline will trigger automatically.

---

### Step 5: Trigger CI/CD Pipeline (15-20 minutes)

**Option A: Automatic** (Recommended)

```bash
git add .
git commit -m "Deploy model serving application"
git push origin main
```

**Option B: Manual**

1. Go to: GitHub → Actions → "Model Training CI/CD Pipeline"
2. Click "Run workflow"
3. Select mode: `prod`
4. Click "Run workflow"

**Monitor Progress**:
- Go to: GitHub → Actions
- Click on running workflow
- Watch each job complete:
  1. ✅ Trigger DAG
  2. ✅ Download Outputs
  3. ✅ Validate
  4. ✅ Check Bias
  5. ✅ Compare & Deploy
  6. ✅ Build & Deploy ← **New!**
  7. ✅ Notify

---

### Step 6: Access Deployed Application

**Get URL**:

1. **From Email**: Check your email for deployment notification
2. **From Cloud Console**: 
   - Go to: https://console.cloud.google.com/run
   - Click on `datacraft-app`
   - Copy the URL
3. **From Command Line**:
   ```bash
   gcloud run services describe datacraft-app \
     --region=us-east1 \
     --format="value(status.url)" \
     --project=datacraft-479223
   ```

**Access**: Open URL in browser (e.g., `https://datacraft-app-xxxxx-uc.a.run.app`)

**Test**:
1. Select a dataset from sidebar
2. Enter query: "What are total sales by region?"
3. Click "Generate"
4. Verify SQL, results, and visualization appear

---

## 🔍 Verification Checklist

After deployment, verify:

- [ ] Cloud Run service is running
- [ ] Application URL is accessible
- [ ] Frontend loads correctly
- [ ] Metadata loads from GCS (check browser console)
- [ ] Query generation works
- [ ] Results are displayed
- [ ] Visualization renders

---

## 📋 Files Created/Updated

### New Files
- ✅ `frontend/metadata_loader.py` - Loads best model from GCS
- ✅ `.dockerignore` - Excludes unnecessary files from Docker build
- ✅ `GCP_SETUP_GUIDE.md` - Complete GCP setup instructions
- ✅ `DEPLOYMENT_INSTRUCTIONS.md` - Detailed deployment guide
- ✅ `QUICK_START.md` - This file

### Updated Files
- ✅ `frontend/app.py` - Uses metadata loader
- ✅ `frontend/Dockerfile` - Includes model-training scripts
- ✅ `.github/workflows/model-training-ci-cd.yml` - Added build_and_deploy job
- ✅ `ci-cd/scripts/send_notification.py` - Includes deployment URL

---

## 🐛 Common Issues & Solutions

### Issue: "No metadata found in GCS"

**Solution**: Run Airflow DAG first to generate metadata
```bash
# Trigger DAG in Airflow UI
# Wait for completion
# Verify: gsutil ls gs://your-bucket/models/
```

### Issue: Docker build fails

**Solution**: Test locally first
```bash
docker build -f frontend/Dockerfile -t test .
```

### Issue: Cloud Run deployment fails

**Solution**: Check service account permissions
```bash
gcloud projects get-iam-policy datacraft-479223 \
  --flatten="bindings[].members" \
  --filter="bindings.members:serviceAccount:mlops-ci-cd@*"
```

### Issue: Application can't access GCS

**Solution**: Verify service account has storage permissions
```bash
# Check role
gcloud projects get-iam-policy $PROJECT_ID \
  --flatten="bindings[].members" \
  --filter="bindings.members:serviceAccount:mlops-ci-cd@*" \
  --format="table(bindings.role)"
```

---

## 📞 Next Steps

1. **Monitor Production**: Set up Cloud Monitoring dashboards
2. **Add Alerts**: Configure alerts for errors/performance
3. **Schedule Retraining**: Set up weekly/monthly retraining
4. **Scale**: Adjust Cloud Run resources as needed

---

## 📚 Documentation Reference

- **GCP Setup**: `GCP_SETUP_GUIDE.md`
- **Deployment Details**: `DEPLOYMENT_INSTRUCTIONS.md`
- **Architecture**: `DEPLOYMENT_GAMEPLAN.md`
- **Model Deployment**: `MODEL_DEPLOYMENT_README.md`

---

## ⚡ Quick Commands Reference

```bash
# Test metadata loader
cd frontend && python -c "from metadata_loader import MetadataLoader; print(MetadataLoader('PROJECT', 'BUCKET').get_best_model_name())"

# Build Docker locally
docker build -f frontend/Dockerfile -t datacraft-app:test .

# Run locally
docker run -p 8501:8501 -e GCP_PROJECT_ID=xxx -e GCS_BUCKET_NAME=xxx datacraft-app:test

# Check Cloud Run
gcloud run services list --region=us-east1

# View logs
gcloud run services logs read datacraft-app --region=us-east1

# Check GCS metadata
gsutil ls gs://your-bucket/models/
```

---

**Ready to deploy?** Start with `GCP_SETUP_GUIDE.md` → Then follow steps above! 🚀

