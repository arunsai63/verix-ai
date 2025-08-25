"""
Dataset Management API Routes
"""

import logging
from typing import List
from pathlib import Path
import shutil

from fastapi import APIRouter, HTTPException

from app.core.config import settings
from app.schemas.responses import DatasetInfo
from app.services.vector_store import VectorStoreService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/datasets", tags=["datasets"])

# Initialize services
vector_store = VectorStoreService()


@router.get("", response_model=List[DatasetInfo])
async def list_datasets():
    """List all available datasets."""
    try:
        datasets = []
        datasets_dir = Path(settings.datasets_directory)
        
        if datasets_dir.exists():
            for dataset_path in datasets_dir.iterdir():
                if dataset_path.is_dir():
                    files = list(dataset_path.glob("*"))
                    datasets.append(DatasetInfo(
                        name=dataset_path.name,
                        document_count=len(files),
                        created_at=dataset_path.stat().st_ctime,
                        size_bytes=sum(f.stat().st_size for f in files if f.is_file())
                    ))
        
        return datasets
        
    except Exception as e:
        logger.error(f"Error listing datasets: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{dataset_name}")
async def delete_dataset(dataset_name: str):
    """Delete a dataset and all its documents."""
    try:
        success = vector_store.delete_dataset(dataset_name)
        
        dataset_dir = Path(settings.datasets_directory) / dataset_name
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)
        
        if success:
            return {"status": "success", "message": f"Dataset {dataset_name} deleted"}
        else:
            raise HTTPException(status_code=500, detail="Failed to delete dataset")
            
    except Exception as e:
        logger.error(f"Error deleting dataset: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{dataset_name}/stats")
async def get_dataset_stats(dataset_name: str):
    """Get statistics for a specific dataset."""
    try:
        stats = vector_store.get_dataset_stats(dataset_name)
        
        dataset_dir = Path(settings.datasets_directory) / dataset_name
        if dataset_dir.exists():
            files = list(dataset_dir.glob("*"))
            stats["file_count"] = len(files)
            stats["total_size_bytes"] = sum(f.stat().st_size for f in files if f.is_file())
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting dataset stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))