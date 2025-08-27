"""
Dataset Service for managing document collections.
"""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import asyncio

from app.core.config import settings

logger = logging.getLogger(__name__)


class DatasetService:
    """Service for managing datasets and document collections."""
    
    def __init__(self):
        self.datasets_dir = Path(settings.datasets_directory)
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
    
    async def store_documents(
        self,
        dataset_name: str,
        documents: List[Dict[str, Any]]
    ) -> bool:
        """
        Store processed documents in a dataset.
        
        Args:
            dataset_name: Name of the dataset
            documents: List of processed documents
            
        Returns:
            Success status
        """
        try:
            dataset_path = self.datasets_dir / dataset_name
            dataset_path.mkdir(parents=True, exist_ok=True)
            
            # Store each document
            for doc in documents:
                filename = doc["metadata"].get("filename", "unknown.json")
                doc_path = dataset_path / f"{filename}.processed.json"
                
                with open(doc_path, 'w') as f:
                    json.dump(doc, f, indent=2)
            
            logger.info(f"Stored {len(documents)} documents in dataset {dataset_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store documents: {str(e)}")
            return False
    
    async def get_sample_chunks(
        self,
        dataset_name: str,
        sample_size: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get sample chunks from a dataset.
        
        Args:
            dataset_name: Name of the dataset
            sample_size: Number of chunks to sample
            
        Returns:
            List of sample chunks
        """
        try:
            dataset_path = self.datasets_dir / dataset_name
            if not dataset_path.exists():
                return []
            
            all_chunks = []
            
            # Read processed documents
            for doc_file in dataset_path.glob("*.processed.json"):
                with open(doc_file, 'r') as f:
                    doc = json.load(f)
                    chunks = doc.get("chunks", [])
                    all_chunks.extend(chunks)
                
                if len(all_chunks) >= sample_size:
                    break
            
            # Return sample
            return all_chunks[:sample_size]
            
        except Exception as e:
            logger.error(f"Failed to get sample chunks: {str(e)}")
            return []
    
    async def list_datasets(self) -> List[str]:
        """
        List available datasets.
        
        Returns:
            List of dataset names
        """
        try:
            datasets = [
                d.name for d in self.datasets_dir.iterdir()
                if d.is_dir()
            ]
            return sorted(datasets)
        except Exception as e:
            logger.error(f"Failed to list datasets: {str(e)}")
            return []
    
    async def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """
        Get information about a dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Dataset information
        """
        try:
            dataset_path = self.datasets_dir / dataset_name
            if not dataset_path.exists():
                return {"error": "Dataset not found"}
            
            # Count documents and chunks
            doc_count = 0
            chunk_count = 0
            total_size = 0
            
            for doc_file in dataset_path.glob("*.processed.json"):
                doc_count += 1
                with open(doc_file, 'r') as f:
                    doc = json.load(f)
                    chunks = doc.get("chunks", [])
                    chunk_count += len(chunks)
                    total_size += doc_file.stat().st_size
            
            return {
                "name": dataset_name,
                "documents": doc_count,
                "chunks": chunk_count,
                "size_bytes": total_size,
                "path": str(dataset_path)
            }
            
        except Exception as e:
            logger.error(f"Failed to get dataset info: {str(e)}")
            return {"error": str(e)}