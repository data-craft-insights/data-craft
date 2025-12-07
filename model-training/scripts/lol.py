from google.cloud import bigquery
from google.oauth2 import service_account
import os

# 🔐 1. Point directly to your service account key
SA_PATH = "/Users/shiviesaksenaa/Desktop/data-craft-2/gcp/service-account.json"

if not os.path.exists(SA_PATH):
    raise FileNotFoundError(f"Service account file not found at: {SA_PATH}")

# 🔐 2. Create credentials from that file
credentials = service_account.Credentials.from_service_account_file(SA_PATH)

# 🔐 3. Create BigQuery client with explicit project + credentials
PROJECT_ID = "datacraft-478300"
client = bigquery.Client(project=PROJECT_ID, credentials=credentials)

# 🔍 4. Run a tiny test query
query = "SELECT 1 AS x"
df = client.query(query).to_dataframe()

print("BigQuery test OK. DataFrame:")
print(df)
