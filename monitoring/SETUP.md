# Monitoring Setup - Additional Steps Required

## ✅ What's Already Done

1. ✅ Terraform configuration created in `monitoring/` directory
2. ✅ GitHub Actions workflow updated to auto-deploy monitoring
3. ✅ Alert policies configured (uptime, error rate, latency)
4. ✅ Dashboard configured with key metrics

## 🔧 Additional Steps You Need to Do

### Step 1: Enable Required GCP APIs

Run these commands to enable the necessary APIs:

```bash
gcloud services enable \
  monitoring.googleapis.com \
  cloudresourcemanager.googleapis.com \
  --project=datacraft-data-pipeline
```

### Step 2: Grant Monitoring Permissions to Service Account

Your CI/CD service account needs Monitoring Admin role:

```bash
gcloud projects add-iam-policy-binding datacraft-data-pipeline \
  --member="serviceAccount:mlops-ci-cd@datacraft-data-pipeline.iam.gserviceaccount.com" \
  --role="roles/monitoring.admin" \
  --condition=None
```

**Alternative:** If you prefer minimal permissions, use:
- `roles/monitoring.metricWriter` (for writing metrics)
- `roles/monitoring.alertPolicyEditor` (for alert policies)
- `roles/monitoring.notificationChannelEditor` (for notification channels)
- `roles/monitoring.dashboardEditor` (for dashboards)
- `roles/monitoring.uptimeCheckEditor` (for uptime checks)

### Step 3: (Optional) Set Custom Email for Alerts

If you want a different email than the default (`sharmasanskar004@gmail.com`):

1. Go to GitHub repo → **Settings** → **Secrets and variables** → **Actions**
2. Click **New repository secret**
3. Name: `MONITORING_EMAIL`
4. Value: `your-preferred-email@example.com`
5. Click **Add secret**

### Step 4: Verify Email Notification Channel

After the first deployment:

1. Go to [GCP Console → Monitoring → Notifications](https://console.cloud.google.com/monitoring/notifications)
2. Find "Datacraft Ops Email" channel
3. Click on it and verify your email
4. Check your email inbox for verification message
5. Click the verification link

**⚠️ Important:** Alerts won't work until the email is verified!

### Step 5: Test the Setup

After deployment, verify everything works:

```bash
# Check if uptime check was created
gcloud monitoring uptime-checks list

# Check if alert policies were created
gcloud alpha monitoring policies list

# Check if dashboard was created
gcloud monitoring dashboards list
```

## 🚀 How It Works

### Automatic Deployment

When you push to `main` branch and the Cloud Run deployment succeeds:

1. GitHub Actions automatically runs Terraform
2. Terraform creates/updates:
   - Uptime check (if not exists)
   - Alert policies (if not exists)
   - Dashboard (if not exists)
   - Notification channel (if not exists)

### Manual Deployment (Optional)

If you want to set up monitoring manually first:

```bash
cd monitoring

# Create terraform.tfvars
cat > terraform.tfvars << EOF
cloud_run_url = "YOUR_CLOUD_RUN_URL_HERE"
notification_email = "your-email@example.com"
EOF

# Initialize and apply
terraform init
terraform plan
terraform apply
```

## 📊 Accessing Your Dashboard

After deployment, get the dashboard URL:

```bash
cd monitoring
terraform output dashboard_url
```

Or find it in:
- GCP Console → Monitoring → Dashboards → "Cloud Run - Datacraft"

## 🔔 Alert Configuration

### Current Alert Thresholds

| Alert | Threshold | Duration |
|-------|-----------|----------|
| Service Down | Uptime check fails | 60s |
| High Error Rate | > 5% | 5 minutes |
| High Latency | P95 > 5 seconds | 5 minutes |

### Customizing Thresholds

Edit `monitoring/main.tf` and modify the `condition_threshold` blocks:

```hcl
# Example: Change error rate threshold to 10%
threshold_value = 0.10  # 10%

# Example: Change latency threshold to 3 seconds
threshold_value = 3000  # 3000ms = 3s
```

Then commit and push - GitHub Actions will update the policies automatically.

## 🐛 Troubleshooting

### Issue: Terraform fails with "Permission denied"

**Solution:** Run Step 2 above to grant Monitoring Admin role.

### Issue: Email alerts not working

**Solution:** 
1. Check Step 4 - verify email in GCP Console
2. Check spam folder for verification email
3. Ensure `MONITORING_EMAIL` secret is set correctly

### Issue: Dashboard shows "No data"

**Solution:**
1. Wait 5-10 minutes after deployment
2. Generate some traffic to your Cloud Run service
3. Check time range in dashboard (try "Last 24 hours")

### Issue: Alerts firing too often

**Solution:**
1. Increase `duration` in alert policy (e.g., from "300s" to "600s")
2. Adjust `threshold_value` to be less sensitive
3. Edit `monitoring/main.tf` and push changes

## 📝 Next Steps

1. ✅ Run Step 1 (Enable APIs)
2. ✅ Run Step 2 (Grant permissions)
3. ✅ (Optional) Run Step 3 (Set custom email)
4. ✅ Push to main branch to trigger deployment
5. ✅ Verify email channel (Step 4)
6. ✅ Check dashboard is working
7. ✅ Test alerts by stopping the service briefly

## 🎯 What You'll Get

After setup, you'll have:

- ✅ **Real-time dashboard** with 6 key metrics
- ✅ **Email alerts** when service goes down
- ✅ **Error rate monitoring** (alerts if > 5%)
- ✅ **Latency monitoring** (alerts if P95 > 5s)
- ✅ **Uptime tracking** (checks every 60s)

All automatically updated on each deployment! 🚀

