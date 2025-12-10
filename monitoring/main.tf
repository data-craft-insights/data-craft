terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
  }
}

provider "google" {
  project = "datacraft-data-pipeline"
  region  = "us-east1"
}

########################
# Notification channel
########################
resource "google_monitoring_notification_channel" "email_channel" {
  display_name = "Datacraft Ops Email"
  type         = "email"
  labels = {
    email_address = var.notification_email
  }
}

########################
# Uptime Check
########################
resource "google_monitoring_uptime_check_config" "cloudrun_uptime" {
  display_name = "Datacraft Cloud Run Uptime"
  timeout      = "10s"
  period       = "60s"
  
  http_check {
    path         = "/"
    port         = "443"
    use_ssl      = true
    validate_ssl = true
  }
  
  monitored_resource {
    type = "uptime_url"
    labels = {
      project_id = "datacraft-data-pipeline"
      host       = var.cloud_run_url
    }
  }
}

########################
# Alert: Service Down
########################
resource "google_monitoring_alert_policy" "uptime_alert" {
  display_name = "Cloud Run Service DOWN"
  combiner     = "OR"
  
  conditions {
    display_name = "Uptime check failed"
    condition_threshold {
      filter = <<-EOT
        metric.type="monitoring.googleapis.com/uptime_check/check_passed"
        AND resource.type="uptime_url"
        AND resource.labels.host="${var.cloud_run_url}"
      EOT
      comparison      = "COMPARISON_LT"
      threshold_value = 1
      duration        = "60s"
      
      aggregations {
        alignment_period     = "60s"
        per_series_aligner   = "ALIGN_NEXT_OLDER"
        cross_series_reducer = "REDUCE_COUNT_FALSE"
      }
    }
  }
  
  notification_channels = [
    google_monitoring_notification_channel.email_channel.name
  ]
  
  alert_strategy {
    auto_close = "1800s" # Auto-close after 30 minutes if resolved
  }
}

########################
# Alert: High Error Rate
########################
resource "google_monitoring_alert_policy" "error_rate_alert" {
  display_name = "Cloud Run High Error Rate"
  combiner     = "OR"
  
  conditions {
    display_name = "Error rate > 5%"
    condition_threshold {
      filter = <<-EOT
        resource.type="cloud_run_revision"
        AND resource.labels.service_name="datacraft-app"
        AND metric.type="run.googleapis.com/request_count"
        AND metric.labels.response_code_class="5xx"
      EOT
      comparison      = "COMPARISON_GT"
      threshold_value = 0.05
      duration        = "300s" # 5 minutes
      
      aggregations {
        alignment_period     = "60s"
        per_series_aligner   = "ALIGN_RATE"
        cross_series_reducer = "REDUCE_SUM"
        group_by_fields      = ["resource.label.service_name"]
      }
      
      denominator_filter = <<-EOT
        resource.type="cloud_run_revision"
        AND resource.labels.service_name="datacraft-app"
        AND metric.type="run.googleapis.com/request_count"
      EOT
      
      denominator_aggregations {
        alignment_period     = "60s"
        per_series_aligner   = "ALIGN_RATE"
        cross_series_reducer = "REDUCE_SUM"
      }
    }
  }
  
  notification_channels = [
    google_monitoring_notification_channel.email_channel.name
  ]
  
  alert_strategy {
    auto_close = "1800s"
  }
}

########################
# Alert: High Latency
########################
resource "google_monitoring_alert_policy" "latency_alert" {
  display_name = "Cloud Run High Latency"
  combiner     = "OR"
  
  conditions {
    display_name = "P95 latency > 5s"
    condition_threshold {
      filter = <<-EOT
        resource.type="cloud_run_revision"
        AND resource.labels.service_name="datacraft-app"
        AND metric.type="run.googleapis.com/request_latencies"
      EOT
      comparison      = "COMPARISON_GT"
      threshold_value = 5000 # 5 seconds in milliseconds
      duration        = "300s"
      
      aggregations {
        alignment_period     = "60s"
        per_series_aligner   = "ALIGN_DELTA"
        cross_series_reducer = "REDUCE_PERCENTILE_95"
        group_by_fields      = ["resource.label.service_name"]
      }
    }
  }
  
  notification_channels = [
    google_monitoring_notification_channel.email_channel.name
  ]
  
  alert_strategy {
    auto_close = "1800s"
  }
}

########################
# Cloud Run Dashboard
########################
resource "google_monitoring_dashboard" "cloudrun_dashboard" {
  dashboard_json = jsonencode({
    displayName = "Cloud Run - Datacraft"
    gridLayout = {
      widgets = [
        {
          title = "Request Count"
          xyChart = {
            dataSets = [
              {
                timeSeriesFilter = {
                  filter = "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"datacraft-app\" AND metric.type=\"run.googleapis.com/request_count\""
                  aggregation = {
                    alignmentPeriod   = "60s"
                    perSeriesAligner  = "ALIGN_RATE"
                    crossSeriesReducer = "REDUCE_SUM"
                  }
                }
              }
            ]
          }
        },
        {
          title = "Latency (p50, p95, p99)"
          xyChart = {
            dataSets = [
              {
                timeSeriesFilter = {
                  filter = "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"datacraft-app\" AND metric.type=\"run.googleapis.com/request_latencies\""
                  aggregation = {
                    alignmentPeriod   = "60s"
                    perSeriesAligner  = "ALIGN_DELTA"
                    crossSeriesReducer = "REDUCE_PERCENTILE_50"
                  }
                }
                plotType = "LINE"
              },
              {
                timeSeriesFilter = {
                  filter = "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"datacraft-app\" AND metric.type=\"run.googleapis.com/request_latencies\""
                  aggregation = {
                    alignmentPeriod   = "60s"
                    perSeriesAligner  = "ALIGN_DELTA"
                    crossSeriesReducer = "REDUCE_PERCENTILE_95"
                  }
                }
                plotType = "LINE"
              },
              {
                timeSeriesFilter = {
                  filter = "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"datacraft-app\" AND metric.type=\"run.googleapis.com/request_latencies\""
                  aggregation = {
                    alignmentPeriod   = "60s"
                    perSeriesAligner  = "ALIGN_DELTA"
                    crossSeriesReducer = "REDUCE_PERCENTILE_99"
                  }
                }
                plotType = "LINE"
              }
            ]
          }
        },
        {
          title = "Error Count"
          xyChart = {
            dataSets = [
              {
                timeSeriesFilter = {
                  filter = "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"datacraft-app\" AND metric.type=\"run.googleapis.com/error_count\""
                  aggregation = {
                    alignmentPeriod   = "60s"
                    perSeriesAligner  = "ALIGN_DELTA"
                    crossSeriesReducer = "REDUCE_SUM"
                  }
                }
              }
            ]
          }
        },
        {
          title = "Error Rate (%)"
          xyChart = {
            dataSets = [
              {
                timeSeriesFilter = {
                  filter = "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"datacraft-app\" AND metric.type=\"run.googleapis.com/request_count\""
                  aggregation = {
                    alignmentPeriod   = "60s"
                    perSeriesAligner  = "ALIGN_RATE"
                    crossSeriesReducer = "REDUCE_SUM"
                  }
                }
              }
            ]
          }
        },
        {
          title = "Instance Count"
          xyChart = {
            dataSets = [
              {
                timeSeriesFilter = {
                  filter = "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"datacraft-app\" AND metric.type=\"run.googleapis.com/container/instance_count\""
                  aggregation = {
                    alignmentPeriod   = "60s"
                    perSeriesAligner  = "ALIGN_MEAN"
                    crossSeriesReducer = "REDUCE_SUM"
                  }
                }
              }
            ]
          }
        },
        {
          title = "CPU Utilization"
          xyChart = {
            dataSets = [
              {
                timeSeriesFilter = {
                  filter = "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"datacraft-app\" AND metric.type=\"run.googleapis.com/container/cpu/utilizations\""
                  aggregation = {
                    alignmentPeriod   = "60s"
                    perSeriesAligner  = "ALIGN_MEAN"
                    crossSeriesReducer = "REDUCE_MEAN"
                  }
                }
              }
            ]
          }
        }
      ]
    }
  })
}

