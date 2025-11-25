#!/usr/bin/env python3
"""
Trigger Airflow DAG and Wait for Completion
Triggers the model training DAG and polls until completion
"""

import sys
import os
import time
import requests
import yaml
from pathlib import Path
from typing import Optional

# Add paths
project_root = Path(__file__).parent.parent.parent
config_path = project_root / "ci-cd" / "config" / "ci_cd_config.yaml"

def load_config():
    """Load CI/CD configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_airflow_session(airflow_url: str, username: str, password: str) -> Optional[requests.Session]:
    """
    Create authenticated session with Airflow
    Tests authentication by hitting the health endpoint
    """
    session = requests.Session()
    
    # Ensure URL doesn't have trailing slash
    base_url = airflow_url.rstrip('/')
    
    print(f"Authenticating with Airflow at: {base_url}")
    
    # Test authentication with health endpoint
    print("  Testing authentication...")
    test_url = f"{base_url}/api/v1/health"
    try:
        response = session.get(test_url, auth=(username, password), timeout=10)
        if response.status_code == 200:
            print("✓ Authentication successful")
            return session
        else:
            print(f"✗ Authentication failed: {response.status_code}")
            return None
    except Exception as e:
        print(f"✗ Authentication test failed: {e}")
        return None

def trigger_dag(session: requests.Session, airflow_url: str, dag_id: str, username: str, password: str) -> Optional[str]:
    """Trigger DAG and return DAG run ID"""
    base_url = airflow_url.rstrip('/')
    url = f"{base_url}/api/v1/dags/{dag_id}/dagRuns"
    
    payload = {"conf": {}}
    
    try:
        print(f"Attempting to trigger DAG at: {url}")
        
        # Always use explicit auth parameter for reliability
        response = session.post(url, json=payload, auth=(username, password), timeout=10)
        
        if response.status_code == 401:
            print(f"✗ Authentication failed (401)")
            print(f"  Response: {response.text}")
            return None
        
        response.raise_for_status()
        
        data = response.json()
        dag_run_id = data.get('dag_run_id')
        print(f"✓ DAG triggered: {dag_run_id}")
        return dag_run_id
        
    except requests.exceptions.RequestException as e:
        print(f"✗ Failed to trigger DAG: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"  Status Code: {e.response.status_code}")
            print(f"  Response: {e.response.text}")
        return None

def get_dag_run_status(session: requests.Session, airflow_url: str, dag_id: str, dag_run_id: str, username: str, password: str) -> Optional[str]:
    """Get DAG run status"""
    base_url = airflow_url.rstrip('/')
    url = f"{base_url}/api/v1/dags/{dag_id}/dagRuns/{dag_run_id}"
    
    try:
        response = session.get(url, auth=(username, password), timeout=10)
        response.raise_for_status()
        
        data = response.json()
        state = data.get('state')
        return state
        
    except requests.exceptions.RequestException as e:
        print(f"⚠ Failed to get DAG status: {e}")
        return None

def wait_for_dag_completion(
    session: requests.Session,
    airflow_url: str,
    dag_id: str,
    dag_run_id: str,
    username: str,
    password: str,
    max_wait_minutes: int = 30,
    poll_interval_seconds: int = 30
) -> int:
    """Wait for DAG to complete, return 0 on success, 1 on failure"""
    print(f"\nWaiting for DAG to complete (max {max_wait_minutes} minutes)...")
    
    max_wait_seconds = max_wait_minutes * 60
    start_time = time.time()
    last_state = None
    
    while True:
        elapsed = time.time() - start_time
        
        if elapsed > max_wait_seconds:
            print(f"\n✗ Timeout: DAG did not complete within {max_wait_minutes} minutes")
            return 1
        
        state = get_dag_run_status(session, airflow_url, dag_id, dag_run_id, username, password)
        
        if state != last_state:
            print(f"  DAG state: {state} (elapsed: {int(elapsed/60)}m {int(elapsed%60)}s)")
            last_state = state
        
        if state == "success":
            print(f"\n✓ DAG completed successfully in {int(elapsed/60)}m {int(elapsed%60)}s")
            return 0
        elif state == "failed":
            print(f"\n✗ DAG failed")
            return 1
        
        time.sleep(poll_interval_seconds)

def main():
    """Main function"""
    print("=" * 70)
    print("TRIGGER AIRFLOW DAG")
    print("=" * 70)
    
    # Get configuration from environment variables
    airflow_url = os.environ.get('AIRFLOW_URL', 'http://localhost:8080')
    dag_id = os.environ.get('AIRFLOW_DAG_ID', 'model_pipeline_with_evaluation')
    username = os.environ.get('AIRFLOW_USERNAME', 'admin')
    password = os.environ.get('AIRFLOW_PASSWORD', 'admin')
    
    print(f"Airflow URL: {airflow_url}")
    print(f"DAG ID: {dag_id}")
    print(f"Username: {username}")
    print(f"Password: {'*' * len(password) if password else 'NOT SET'}")
    
    # Validate credentials are set
    if not username or not password:
        print("\n✗ Error: AIRFLOW_USERNAME and AIRFLOW_PASSWORD must be set")
        return 1
    
    if not airflow_url:
        print("\n✗ Error: AIRFLOW_URL must be set")
        return 1
    
    # Create authenticated session
    session = create_airflow_session(airflow_url, username, password)
    if not session:
        print("\n✗ Failed to authenticate with Airflow")
        print("\nTroubleshooting:")
        print("  1. Verify AIRFLOW_URL is correct and accessible")
        print("  2. Verify AIRFLOW_USERNAME and AIRFLOW_PASSWORD are correct")
        print("  3. Ensure Airflow REST API is enabled")
        return 1
    
    # Trigger DAG
    dag_run_id = trigger_dag(session, airflow_url, dag_id, username, password)
    
    if not dag_run_id:
        print("\n✗ Failed to trigger DAG")
        print("\nTroubleshooting:")
        print("  1. Check that the DAG exists and is enabled in Airflow")
        print("  2. Verify the DAG ID is correct: model_pipeline_with_evaluation")
        print("  3. Check Airflow logs for any DAG trigger errors")
        return 1
    
    # Wait for completion
    result = wait_for_dag_completion(session, airflow_url, dag_id, dag_run_id, username, password)
    
    return result

if __name__ == "__main__":
    sys.exit(main())