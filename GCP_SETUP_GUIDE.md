# GCP Setup Guide for Model Deployment

This guide walks you through all the GCP setup steps needed for the complete CI/CD deployment pipeline.

## Prerequisites

- GCP account with billing enabled
- `gcloud` CLI installed and configured
- GitHub repository with Actions enabled

---

## Step 1: Create/Select GCP Project (5 minutes)

```bash
# Set your project ID (use existing or create new)
export PROJECT_ID="datacraft-data-pipeline"  # Replace with your project ID
export REGION="us-east1"

# Set project
gcloud config set project $PROJECT_ID

# If creating new project:
# gcloud projects create $PROJECT_ID --name="DataCraft MLOps"
```

---

## Step 2: Enable Required APIs (5 minutes)

```bash
# Enable all required APIs
gcloud services enable \
  aiplatform.googleapis.com \
  storage.googleapis.com \
  bigquery.googleapis.com \
  cloudresourcemanager.googleapis.com \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  --project=$PROJECT_ID

# Verify APIs are enabled
gcloud services list --enabled --project=$PROJECT_ID
```

---

## Step 3: Create Service Account (5 minutes)

```bash
# Create service account for CI/CD
gcloud iam service-accounts create mlops-ci-cd \
  --display-name="MLOps CI/CD Service Account" \
  --description="Service account for CI/CD pipeline and Cloud Run deployment" \
  --project=$PROJECT_ID

# Grant required roles
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:mlops-ci-cd@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/aiplatform.user"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:mlops-ci-cd@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/storage.admin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:mlops-ci-cd@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/bigquery.admin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:mlops-ci-cd@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/run.admin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:mlops-ci-cd@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/artifactregistry.writer"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:mlops-ci-cd@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/iam.serviceAccountUser"

# Create and download key
gcloud iam service-accounts keys create gcp-key.json \
  --iam-account=mlops-ci-cd@$PROJECT_ID.iam.gserviceaccount.com \
  --project=$PROJECT_ID

echo "✓ Service account key saved to: gcp-key.json"
echo "⚠️  IMPORTANT: Add this to GitHub Secrets as GCP_SA_KEY"
```

---

## Step 4: Create GCS Bucket (2 minutes)

```bash
# Create bucket for model artifacts
export BUCKET_NAME="model-datacraft-sanskar"  # Replace with your bucket name

# Create bucket
gsutil mb -p $PROJECT_ID -c STANDARD -l $REGION gs://$BUCKET_NAME

# Verify bucket exists
gsutil ls -p $PROJECT_ID
```

**Note**: Bucket name must be globally unique. If it exists, choose a different name.

---

## Step 5: Create BigQuery Dataset (2 minutes)

```bash
# Create BigQuery dataset
bq mk --dataset \
  --location=US \
  --description="DataCraft ML dataset" \
  $PROJECT_ID:datacraft_ml

# Verify dataset
bq ls --project_id=$PROJECT_ID
```

---

## Step 6: Create Artifact Registry (3 minutes)

```bash
# Create Artifact Registry repository for Docker images
gcloud artifacts repositories create mlops-models \
  --repository-format=docker \
  --location=$REGION \
  --description="DataCraft application containers" \
  --project=$PROJECT_ID

# Configure Docker authentication
gcloud auth configure-docker $REGION-docker.pkg.dev

# Verify repository
gcloud artifacts repositories list --location=$REGION --project=$PROJECT_ID
```

---

## Step 7: Configure Cloud Run (Automatic)

Cloud Run will be automatically configured during deployment. However, you can verify settings:

```bash
# Check Cloud Run API is enabled
gcloud services list --enabled --filter="name:run.googleapis.com" --project=$PROJECT_ID

# List existing services (should be empty initially)
gcloud run services list --region=$REGION --project=$PROJECT_ID
```

---

## Step 8: Verify Setup (5 minutes)

Run this verification script:

```bash
#!/bin/bash
# verify_gcp_setup.sh

PROJECT_ID="datacraft-data-pipeline"  # Replace with your project ID
REGION="us-east1"
BUCKET_NAME="model-datacraft-sanskar"  # Replace with your bucket name

echo "Verifying GCP Setup..."
echo "======================"

# Check project
echo -n "Project: "
gcloud config get-value project

# Check APIs
echo -n "APIs enabled: "
gcloud services list --enabled --filter="name:aiplatform.googleapis.com OR name:storage.googleapis.com OR name:run.googleapis.com" --format="value(name)" | wc -l

# Check service account
echo -n "Service account exists: "
gcloud iam service-accounts list --filter="email:mlops-ci-cd@$PROJECT_ID.iam.gserviceaccount.com" --format="value(email)" | wc -l

# Check bucket
echo -n "GCS bucket exists: "
gsutil ls gs://$BUCKET_NAME > /dev/null 2>&1 && echo "✓" || echo "✗"

# Check BigQuery dataset
echo -n "BigQuery dataset exists: "
bq ls --project_id=$PROJECT_ID datacraft_ml > /dev/null 2>&1 && echo "✓" || echo "✗"

# Check Artifact Registry
echo -n "Artifact Registry exists: "
gcloud artifacts repositories list --location=$REGION --filter="name:mlops-models" --format="value(name)" | wc -l

echo "======================"
echo "Verification complete!"
```

---

## Step 9: Configure GitHub Secrets

Go to your GitHub repository: **Settings → Secrets and variables → Actions**

Add these secrets:

### Required Secrets

1. **`GCP_SA_KEY`**
   - Value: Entire contents of `gcp-key.json` (from Step 3)
   - How to get: `cat gcp-key.json | pbcopy` (Mac) or `cat gcp-key.json` (copy output)

2. **`GCS_BUCKET_NAME`**
   - Value: Your bucket name (e.g., `model-datacraft-sanskar`)

3. **`EMAIL_SMTP_PASSWORD`**
   - Value: Gmail App Password (16 characters)
   - How to get:
     1. Go to [Google Account → Security](https://myaccount.google.com/security)
     2. Enable **2-Step Verification** (if not enabled)
     3. Go to **App Passwords**
     4. Generate password for "Mail"
     5. Copy the 16-character password

4. **`AIRFLOW_URL`**
   - Value: Your Airflow URL (e.g., `http://your-airflow-instance:8080`)
   - For local testing: Use ngrok or your public IP

5. **`AIRFLOW_USERNAME`**
   - Value: Your Airflow username (e.g., `admin`)

6. **`AIRFLOW_PASSWORD`**
   - Value: Your Airflow password

### Optional Secrets

7. **`EMAIL_SMTP_USER`**
   - Value: Your email address (optional, defaults to config file)

---

## Step 10: Update Configuration Files

Update `ci-cd/config/ci_cd_config.yaml`:

```yaml
gcp:
  project_id: "datacraft-data-pipeline"  # Your project ID
  region: "us-east1"              # Your region
  dataset_id: "datacraft_ml"     # Your BigQuery dataset
  bucket_name: "model-datacraft-sanskar"  # Your bucket name

notifications:
  email:
    smtp_server: "smtp.gmail.com"
    smtp_port: 587
    from_email: "your-email@gmail.com"  # Your email
    to_email: "your-email@gmail.com"    # Your email
    use_tls: true

model_registry:
  base_path: "models"

rollback:
  enabled: true
  min_improvement_threshold: 0.0
```

---

## Step 11: Update GitHub Workflow Environment

Update `.github/workflows/model-training-ci-cd.yml`:

```yaml
env:
  PYTHON_VERSION: '3.9'
  GCP_PROJECT_ID: 'datacraft-data-pipeline'  # Your project ID
```

---

## Quick Setup Script

Save this as `setup_gcp.sh` and run it:

```bash
#!/bin/bash
set -e

PROJECT_ID="datacraft-data-pipeline"  # CHANGE THIS
REGION="us-east1"
BUCKET_NAME="model-datacraft-sanskar"  # CHANGE THIS (must be globally unique)

echo "Setting up GCP for DataCraft MLOps..."
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo "Bucket: $BUCKET_NAME"
echo ""

# Set project
gcloud config set project $PROJECT_ID

# Enable APIs
echo "Enabling APIs..."
gcloud services enable \
  aiplatform.googleapis.com \
  storage.googleapis.com \
  bigquery.googleapis.com \
  cloudresourcemanager.googleapis.com \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  --project=$PROJECT_ID

# Create service account
echo "Creating service account..."
gcloud iam service-accounts create mlops-ci-cd \
  --display-name="MLOps CI/CD Service Account" \
  --project=$PROJECT_ID || echo "Service account may already exist"

# Grant roles
echo "Granting roles..."
for role in aiplatform.user storage.admin bigquery.admin run.admin artifactregistry.writer iam.serviceAccountUser; do
  gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:mlops-ci-cd@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/$role" || echo "Role $role may already be granted"
done

# Create key
echo "Creating service account key..."
gcloud iam service-accounts keys create gcp-key.json \
  --iam-account=mlops-ci-cd@$PROJECT_ID.iam.gserviceaccount.com \
  --project=$PROJECT_ID

# Create bucket
echo "Creating GCS bucket..."
gsutil mb -p $PROJECT_ID -c STANDARD -l $REGION gs://$BUCKET_NAME || echo "Bucket may already exist"

# Create BigQuery dataset
echo "Creating BigQuery dataset..."
bq mk --dataset --location=US $PROJECT_ID:datacraft_ml || echo "Dataset may already exist"

# Create Artifact Registry
echo "Creating Artifact Registry..."
gcloud artifacts repositories create mlops-models \
  --repository-format=docker \
  --location=$REGION \
  --description="DataCraft application containers" \
  --project=$PROJECT_ID || echo "Repository may already exist"

# Configure Docker
echo "Configuring Docker authentication..."
gcloud auth configure-docker $REGION-docker.pkg.dev

echo ""
echo "✓ GCP setup complete!"
echo ""
echo "Next steps:"
echo "1. Add gcp-key.json contents to GitHub Secret: GCP_SA_KEY"
echo "2. Add $BUCKET_NAME to GitHub Secret: GCS_BUCKET_NAME"
echo "3. Update ci-cd/config/ci_cd_config.yaml with your settings"
echo "4. Update .github/workflows/model-training-ci-cd.yml with your PROJECT_ID"
```

**To use the script**:
```bash
# Edit the script and update PROJECT_ID and BUCKET_NAME
nano setup_gcp.sh

# Make executable
chmod +x setup_gcp.sh

# Run it
./setup_gcp.sh
```

---

## Troubleshooting

### Issue: "Permission denied" errors

**Solution**: Verify service account has all required roles:
```bash
gcloud projects get-iam-policy $PROJECT_ID \
  --flatten="bindings[].members" \
  --filter="bindings.members:serviceAccount:mlops-ci-cd@$PROJECT_ID.iam.gserviceaccount.com"
```

### Issue: "Bucket already exists"

**Solution**: Use a different bucket name (must be globally unique):
```bash
export BUCKET_NAME="model-datacraft-$(date +%s)"
gsutil mb -p $PROJECT_ID -c STANDARD -l $REGION gs://$BUCKET_NAME
```

### Issue: "API not enabled"

**Solution**: Enable the specific API:
```bash
gcloud services enable <api-name>.googleapis.com --project=$PROJECT_ID
```

### Issue: "Service account already exists"

**Solution**: This is fine - the script will continue. If you need to recreate:
```bash
# Delete existing (if needed)
gcloud iam service-accounts delete mlops-ci-cd@$PROJECT_ID.iam.gserviceaccount.com

# Then create again
gcloud iam service-accounts create mlops-ci-cd \
  --display-name="MLOps CI/CD Service Account" \
  --project=$PROJECT_ID
```

---

## Cost Estimation

Approximate monthly costs (varies by usage):

- **Cloud Run**: ~$0.40 per million requests + compute time
- **Artifact Registry**: ~$0.10 per GB stored
- **Cloud Storage**: ~$0.020 per GB stored
- **BigQuery**: ~$5 per TB queried
- **Vertex AI**: Pay per API call (Gemini pricing)

**Estimated**: $10-50/month for moderate usage

---

## Security Best Practices

1. ✅ **Never commit** `gcp-key.json` to git
2. ✅ **Rotate keys** periodically (every 90 days)
3. ✅ **Use least privilege** - only grant necessary roles
4. ✅ **Enable audit logs** for production
5. ✅ **Use VPC** for production deployments

---

## Verification Checklist

After completing setup, verify:

- [ ] All APIs enabled
- [ ] Service account created with key downloaded
- [ ] GCS bucket created and accessible
- [ ] BigQuery dataset created
- [ ] Artifact Registry repository created
- [ ] Docker authentication configured
- [ ] GitHub secrets configured
- [ ] Configuration files updated

---

## Next Steps

After GCP setup is complete:

1. **Follow**: `DEPLOYMENT_INSTRUCTIONS.md` for deployment steps
2. **Or**: `QUICK_START.md` for quick reference
3. **Test**: Run first Airflow DAG to generate model metadata
4. **Deploy**: Push to GitHub to trigger CI/CD pipeline

---

**Last Updated**: December 2025  
**Project ID**: datacraft-data-pipeline

