# frontend/metadata_loader.py
"""
Metadata Loader - Load best model metadata from GCS
Used by frontend to get current production model configuration
"""
from google.cloud import storage
import json
from typing import Dict, Optional
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetadataLoader:
    """
    Load best model metadata from GCS
    Used by frontend to get current production model configuration
    """
    def __init__(self, project_id: str, bucket_name: str):
        self.project_id = project_id
        self.bucket_name = bucket_name
        try:
            self.storage_client = storage.Client(project=project_id)
            self.bucket = self.storage_client.bucket(bucket_name)
            logger.info(f"Initialized MetadataLoader for bucket: {bucket_name}")
        except Exception as e:
            logger.error(f"Failed to initialize storage client: {e}")
            raise
    
    def load_latest_metadata(self, base_path: str = "models") -> Dict:
        """
        Load latest best_model_metadata.json from GCS
        
        Returns:
            Dictionary with:
            - selected_model: "gemini-2.5-flash"
            - composite_score: 89.5
            - performance_score: 92.0
            - bias_score: 25.0
            - success_rate: 90.0
            - avg_response_time: 1.2
            - deployment_ready: True
        """
        logger.info(f"Loading latest metadata from gs://{self.bucket_name}/{base_path}/")
        
        # List all model directories
        try:
            blobs = list(self.bucket.list_blobs(prefix=f"{base_path}/"))
        except Exception as e:
            logger.warning(f"Failed to list blobs from {base_path}/: {e}")
            blobs = []
        
        # Find latest timestamp
        model_dirs = {}
        for blob in blobs:
            if "best_model_metadata.json" in blob.name:
                # Extract timestamp from path: models/{timestamp}_{commit}/
                parts = blob.name.split("/")
                if len(parts) >= 2:
                    dir_name = parts[1]  # {timestamp}_{commit}
                    timestamp = dir_name.split("_")[0]
                    model_dirs[timestamp] = blob.name
        
        # Fallback: try best_model_responses path
        if not model_dirs:
            logger.info("No metadata in models/, trying best_model_responses/")
            try:
                blobs = list(self.bucket.list_blobs(prefix="best_model_responses/"))
                for blob in blobs:
                    if "best_model_metadata.json" in blob.name:
                        model_dirs["fallback"] = blob.name
                        logger.info(f"Found metadata in fallback path: {blob.name}")
                        break
            except Exception as e:
                logger.warning(f"Failed to check fallback path: {e}")
        
        if not model_dirs:
            raise ValueError(f"No metadata found in GCS bucket {self.bucket_name}")
        
        # Get latest
        if "fallback" in model_dirs:
            latest_blob_path = model_dirs["fallback"]
        else:
            latest_timestamp = max(model_dirs.keys())
            latest_blob_path = model_dirs[latest_timestamp]
        
        logger.info(f"Loading metadata from: {latest_blob_path}")
        
        # Download and parse
        try:
            blob = self.bucket.blob(latest_blob_path)
            metadata_json = blob.download_as_text()
            metadata = json.loads(metadata_json)
            logger.info(f"✓ Loaded metadata for model: {metadata.get('selected_model', 'unknown')}")
            return metadata
        except Exception as e:
            logger.error(f"Failed to download/parse metadata: {e}")
            raise
    
    def get_best_model_name(self) -> str:
        """Get best model name from latest metadata"""
        try:
            metadata = self.load_latest_metadata()
            model_name = metadata.get("selected_model", "gemini-2.5-flash")
            logger.info(f"Best model: {model_name}")
            return model_name
        except Exception as e:
            logger.warning(f"Failed to load metadata, using default: {e}")
            return "gemini-2.5-flash"
    
    def get_hyperparameters(self) -> Dict:
        """
        Extract hyperparameters from metadata or use defaults
        
        Returns:
            {
                "temperature": 0.2,
                "top_p": 0.9,
                "top_k": 40
            }
        """
        try:
            metadata = self.load_latest_metadata()
            
            # Check if hyperparameters are in metadata
            if "hyperparameters" in metadata:
                return metadata["hyperparameters"]
        except Exception as e:
            logger.warning(f"Failed to load metadata for hyperparameters: {e}")
        
        # Defaults (should match tuning defaults)
        logger.info("Using default hyperparameters")
        return {
            "temperature": 0.2,
            "top_p": 0.9,
            "top_k": 40
        }
    
    def get_model_info(self) -> Dict:
        """
        Get complete model information including scores
        
        Returns:
            {
                "selected_model": "gemini-2.5-flash",
                "composite_score": 89.5,
                "performance_score": 92.0,
                "bias_score": 25.0,
                "success_rate": 90.0,
                "avg_response_time": 1.2
            }
        """
        try:
            metadata = self.load_latest_metadata()
            return {
                "selected_model": metadata.get("selected_model", "gemini-2.5-flash"),
                "composite_score": metadata.get("composite_score", 0),
                "performance_score": metadata.get("performance_score", 0),
                "bias_score": metadata.get("bias_score", 0),
                "success_rate": metadata.get("success_rate", 0),
                "avg_response_time": metadata.get("avg_response_time", 0),
                "selection_date": metadata.get("selection_date", "unknown")
            }
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return {
                "selected_model": "gemini-2.5-flash",
                "composite_score": 0,
                "performance_score": 0,
                "bias_score": 0,
                "success_rate": 0,
                "avg_response_time": 0,
                "selection_date": "unknown"
            }

