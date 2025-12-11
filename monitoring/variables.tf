variable "cloud_run_url" {
  description = "Cloud Run service URL (e.g., datacraft-app-xxxxx.us-east1.run.app)"
  type        = string
}

variable "notification_email" {
  description = "Email address for monitoring alerts"
  type        = string
  default     = "sharmasanskar004@gmail.com"
}

