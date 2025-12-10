# Cloud Run Monitoring Setup

This Terraform configuration sets up comprehensive monitoring for your Cloud Run service.

## Features

✅ **Uptime Check** - Monitors service availability  
✅ **Alert Policies** - Service down, high error rate, high latency  
✅ **Monitoring Dashboard** - Real-time metrics visualization  
✅ **Email Notifications** - Automatic alerts via email  

## Prerequisites

1. **Terraform installed** (v1.6.0+)
   ```bash
   # macOS
   brew install terraform
   
   # Or download from https://www.terraform.io/downloads
   ```

2. **GCP Authentication**
   ```bash
   gcloud auth application-default login
   ```

3. **Required GCP APIs enabled:**
   ```bash
   gcloud services enable \
     monitoring.googleapis.com \
     cloudresourcemanager.googleapis.com
   ```

## Quick Start

### Option 1: Manual Setup (One-time)

1. **Navigate to monitoring directory:**
   ```bash
   cd monitoring
   ```

2. **Create terraform.tfvars file:**
   ```bash
   cat > terraform.tfvars << EOF
   cloud_run_url = "datacraft-app-xxxxx.us-east1.run.app"
   notification_email = "your-email@example.com"
   EOF
   ```
   > Replace with your actual Cloud Run URL and email

3. **Initialize Terraform:**
   ```bash
   terraform init
   ```

4. **Review changes:**
   ```bash
   terraform plan
   ```

5. **Apply configuration:**
   ```bash
   terraform apply
   ```

### Option 2: Automated via CI/CD

The monitoring infrastructure is automatically deployed via GitHub Actions after each Cloud Run deployment.

**Configuration:**
- Cloud Run URL is automatically detected from deployment
- Email is set via `MONITORING_EMAIL` GitHub secret (defaults to config file email)

**To set custom email:**
1. Go to GitHub repo → Settings → Secrets and variables → Actions
2. Add secret: `MONITORING_EMAIL` = `your-email@example.com`

## What Gets Created

### 1. Notification Channel
- Email channel for alerts
- Location: GCP Console → Monitoring → Notifications

### 2. Uptime Check
- Checks service every 60 seconds
- Timeout: 10 seconds
- Location: GCP Console → Monitoring → Uptime Checks

### 3. Alert Policies

#### Service Down Alert
- Triggers when uptime check fails
- Duration: 60 seconds
- Auto-closes after 30 minutes if resolved

#### High Error Rate Alert
- Triggers when error rate > 5%
- Duration: 5 minutes
- Monitors Cloud Run error count

#### High Latency Alert
- Triggers when P95 latency > 5 seconds
- Duration: 5 minutes
- Monitors request latencies

### 4. Monitoring Dashboard

**Metrics included:**
- Request Count (rate)
- Latency (p50, p95, p99)
- Error Count
- Error Rate (%)
- Instance Count
- CPU Utilization

**Access:**
- GCP Console → Monitoring → Dashboards
- Or use the output URL from `terraform output`

## Viewing Metrics

### Via Dashboard
```bash
# Get dashboard URL
terraform output dashboard_url
```

### Via CLI
```bash
# View uptime check
gcloud monitoring uptime-checks list

# View alert policies
gcloud alpha monitoring policies list

# View dashboard
gcloud monitoring dashboards list
```

## Customization

### Change Alert Thresholds

Edit `main.tf`:

```hcl
# Error rate threshold (default: 0.05 = 5%)
threshold_value = 0.05

# Latency threshold in milliseconds (default: 5000 = 5s)
threshold_value = 5000
```

### Add More Notification Channels

Edit `main.tf` to add Slack, PagerDuty, etc.:

```hcl
resource "google_monitoring_notification_channel" "slack_channel" {
  display_name = "Slack Alerts"
  type         = "slack"
  labels = {
    channel_name = "#alerts"
    auth_token   = var.slack_token
  }
}
```

### Modify Dashboard

Edit the `google_monitoring_dashboard` resource in `main.tf` to add/remove widgets.

## Troubleshooting

### Terraform Apply Fails

**Error: Permission denied**
```bash
# Ensure service account has Monitoring Admin role
gcloud projects add-iam-policy-binding datacraft-data-pipeline \
  --member="serviceAccount:mlops-ci-cd@datacraft-data-pipeline.iam.gserviceaccount.com" \
  --role="roles/monitoring.admin"
```

**Error: API not enabled**
```bash
gcloud services enable monitoring.googleapis.com
```

### Alerts Not Firing

1. Check notification channel is verified (check email)
2. Verify alert policy is enabled in GCP Console
3. Check alert conditions match your service metrics

### Dashboard Not Showing Data

1. Ensure Cloud Run service is receiving traffic
2. Wait 5-10 minutes for metrics to populate
3. Check time range in dashboard (default: last 6 hours)

## Cleanup

To remove all monitoring resources:

```bash
cd monitoring
terraform destroy
```

## Files

- `main.tf` - Main Terraform configuration
- `variables.tf` - Input variables
- `outputs.tf` - Output values (dashboard URL, etc.)
- `.gitignore` - Terraform state files (not committed)

## Additional Resources

- [GCP Monitoring Documentation](https://cloud.google.com/monitoring/docs)
- [Terraform Google Provider](https://registry.terraform.io/providers/hashicorp/google/latest/docs)
- [Cloud Run Metrics](https://cloud.google.com/run/docs/monitoring/metrics)

