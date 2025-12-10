output "dashboard_url" {
  description = "URL to the monitoring dashboard"
  value       = "https://console.cloud.google.com/monitoring/dashboards/custom/${google_monitoring_dashboard.cloudrun_dashboard.id}?project=datacraft-data-pipeline"
}

output "notification_channel_id" {
  description = "Notification channel ID"
  value       = google_monitoring_notification_channel.email_channel.name
}

output "uptime_check_id" {
  description = "Uptime check ID"
  value       = google_monitoring_uptime_check_config.cloudrun_uptime.uptime_check_id
}

