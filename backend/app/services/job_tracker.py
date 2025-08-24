"""
Job tracker for async document processing with progress tracking.
"""
import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class JobTracker:
    """Singleton job tracker for managing async document processing jobs."""
    
    _instance = None
    _jobs: Dict[str, Dict[str, Any]] = {}
    _lock = asyncio.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    async def create_job(
        self,
        dataset_name: str,
        total_files: int,
        file_names: List[str]
    ) -> str:
        """Create a new job and return its ID."""
        job_id = str(uuid.uuid4())
        
        async with self._lock:
            self._jobs[job_id] = {
                "id": job_id,
                "dataset_name": dataset_name,
                "status": JobStatus.PENDING.value,
                "total_files": total_files,
                "processed_files": 0,
                "successful_files": 0,
                "failed_files": 0,
                "file_names": file_names,
                "file_statuses": {name: "pending" for name in file_names},
                "progress": 0,
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
                "completed_at": None,
                "error": None,
                "chunks_created": 0,
                "processing_time": None,
                "current_file": None,
                "logs": []
            }
        
        logger.info(f"Created job {job_id} for dataset {dataset_name} with {total_files} files")
        return job_id
    
    async def update_job_status(
        self,
        job_id: str,
        status: JobStatus,
        error: Optional[str] = None
    ):
        """Update job status."""
        async with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id]["status"] = status.value
                self._jobs[job_id]["updated_at"] = datetime.utcnow().isoformat()
                
                if error:
                    self._jobs[job_id]["error"] = error
                
                if status == JobStatus.COMPLETED or status == JobStatus.FAILED:
                    self._jobs[job_id]["completed_at"] = datetime.utcnow().isoformat()
                    
                    # Calculate processing time
                    created_at = datetime.fromisoformat(self._jobs[job_id]["created_at"])
                    completed_at = datetime.utcnow()
                    processing_time = (completed_at - created_at).total_seconds()
                    self._jobs[job_id]["processing_time"] = processing_time
    
    async def update_file_progress(
        self,
        job_id: str,
        file_name: str,
        status: str,
        chunks: Optional[int] = None
    ):
        """Update progress for a specific file."""
        async with self._lock:
            if job_id in self._jobs:
                job = self._jobs[job_id]
                
                # Update file status
                if file_name in job["file_statuses"]:
                    job["file_statuses"][file_name] = status
                
                # Update current file being processed
                if status == "processing":
                    job["current_file"] = file_name
                elif job["current_file"] == file_name:
                    job["current_file"] = None
                
                # Update counters based on status
                if status == "completed":
                    job["successful_files"] += 1
                    job["processed_files"] += 1
                    if chunks:
                        job["chunks_created"] += chunks
                elif status == "failed":
                    job["failed_files"] += 1
                    job["processed_files"] += 1
                
                # Calculate overall progress
                if job["total_files"] > 0:
                    job["progress"] = int((job["processed_files"] / job["total_files"]) * 100)
                
                job["updated_at"] = datetime.utcnow().isoformat()
                
                # Add log entry
                log_entry = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "file": file_name,
                    "status": status,
                    "chunks": chunks
                }
                job["logs"].append(log_entry)
                
                logger.info(f"Job {job_id}: File {file_name} status updated to {status}")
    
    async def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a job."""
        async with self._lock:
            return self._jobs.get(job_id)
    
    async def get_all_jobs(self) -> List[Dict[str, Any]]:
        """Get all jobs."""
        async with self._lock:
            return list(self._jobs.values())
    
    async def cleanup_old_jobs(self, hours: int = 24):
        """Remove jobs older than specified hours."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        async with self._lock:
            jobs_to_remove = []
            for job_id, job in self._jobs.items():
                created_at = datetime.fromisoformat(job["created_at"])
                if created_at < cutoff_time:
                    jobs_to_remove.append(job_id)
            
            for job_id in jobs_to_remove:
                del self._jobs[job_id]
                logger.info(f"Cleaned up old job {job_id}")
    
    async def add_log(self, job_id: str, message: str, level: str = "info"):
        """Add a log message to a job."""
        async with self._lock:
            if job_id in self._jobs:
                log_entry = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "level": level,
                    "message": message
                }
                self._jobs[job_id]["logs"].append(log_entry)


# Global job tracker instance
job_tracker = JobTracker()