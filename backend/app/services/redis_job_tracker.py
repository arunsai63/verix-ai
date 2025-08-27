import os
import json
import redis
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from enum import Enum
import uuid
import logging

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    """Job status enumeration."""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class RedisJobTracker:
    """Redis-based job tracking for document processing."""
    
    def __init__(self, redis_url: Optional[str] = None):
        """Initialize Redis connection."""
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
        self.job_ttl = 3600  # 1 hour TTL for job data
    
    async def create_job(
        self,
        dataset_name: str,
        total_files: int,
        file_names: List[str],
        job_type: str = "document_processing"
    ) -> str:
        """
        Create a new job entry.
        
        Args:
            dataset_name: Name of the dataset
            total_files: Total number of files to process
            file_names: List of file names
            job_type: Type of job
        
        Returns:
            Job ID
        """
        job_id = str(uuid.uuid4())
        
        job_data = {
            "id": job_id,
            "type": job_type,
            "dataset_name": dataset_name,
            "total_files": str(total_files),
            "files": json.dumps(file_names),  # Serialize list to JSON string
            "status": JobStatus.QUEUED.value,
            "progress": "0",
            "documents_processed": "0",
            "documents_failed": "0",
            "chunks_created": "0",
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "started_at": "",  # Use empty string instead of None
            "completed_at": "",  # Use empty string instead of None
            "error": "",  # Use empty string instead of None
            "logs": json.dumps([])  # Serialize empty list to JSON string
        }
        
        # Store job data
        job_key = f"job:{job_id}"
        self.redis_client.hset(job_key, mapping=job_data)
        self.redis_client.expire(job_key, self.job_ttl)
        
        # Add to job list
        self.redis_client.lpush("jobs:all", job_id)
        self.redis_client.ltrim("jobs:all", 0, 999)  # Keep last 1000 jobs
        
        # Add to active jobs set
        self.redis_client.sadd("jobs:active", job_id)
        
        # Create file tracking entries
        for file_name in file_names:
            file_key = f"job:{job_id}:file:{file_name}"
            file_data = {
                "name": file_name,
                "status": "pending",
                "chunks": "0",
                "error": "",
                "started_at": "",
                "completed_at": ""
            }
            self.redis_client.hset(file_key, mapping=file_data)
            self.redis_client.expire(file_key, self.job_ttl)
        
        logger.info(f"Created job {job_id} for dataset {dataset_name}")
        return job_id
    
    async def update_job_status(
        self,
        job_id: str,
        status: JobStatus,
        error: Optional[str] = None,
        **kwargs
    ):
        """Update job status."""
        job_key = f"job:{job_id}"
        
        if not self.redis_client.exists(job_key):
            logger.warning(f"Job {job_id} not found")
            return
        
        updates = {
            "status": status.value,
            "updated_at": datetime.utcnow().isoformat()
        }
        
        if status == JobStatus.PROCESSING:
            updates["started_at"] = datetime.utcnow().isoformat()
        elif status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.TIMEOUT]:
            updates["completed_at"] = datetime.utcnow().isoformat()
            # Remove from active jobs
            self.redis_client.srem("jobs:active", job_id)
        
        if error:
            updates["error"] = error
        
        updates.update(kwargs)
        
        self.redis_client.hset(job_key, mapping=updates)
        logger.info(f"Updated job {job_id} status to {status.value}")
    
    async def update_job_progress(
        self,
        job_id: str,
        progress: float,
        documents_processed: int = 0,
        documents_failed: int = 0,
        chunks_created: int = 0
    ):
        """Update job progress."""
        job_key = f"job:{job_id}"
        
        if not self.redis_client.exists(job_key):
            return
        
        updates = {
            "progress": round(progress, 2),
            "documents_processed": documents_processed,
            "documents_failed": documents_failed,
            "chunks_created": chunks_created,
            "updated_at": datetime.utcnow().isoformat()
        }
        
        self.redis_client.hset(job_key, mapping=updates)
    
    async def update_file_progress(
        self,
        job_id: str,
        file_name: str,
        status: str,
        chunks: int = 0,
        error: Optional[str] = None
    ):
        """Update individual file progress."""
        file_key = f"job:{job_id}:file:{file_name}"
        
        if not self.redis_client.exists(file_key):
            logger.warning(f"File tracking not found: {file_key}")
            return
        
        updates = {
            "status": status,
            "chunks": chunks,
            "updated_at": datetime.utcnow().isoformat()
        }
        
        if status == "processing":
            updates["started_at"] = datetime.utcnow().isoformat()
        elif status in ["completed", "failed", "timeout"]:
            updates["completed_at"] = datetime.utcnow().isoformat()
        
        if error:
            updates["error"] = error
        
        self.redis_client.hset(file_key, mapping=updates)
    
    async def add_log(
        self,
        job_id: str,
        message: str,
        level: str = "info"
    ):
        """Add log entry for a job."""
        log_key = f"job:{job_id}:logs"
        
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": level,
            "message": message
        }
        
        self.redis_client.rpush(log_key, json.dumps(log_entry))
        self.redis_client.expire(log_key, self.job_ttl)
        
        # Limit logs to last 1000 entries
        self.redis_client.ltrim(log_key, -1000, -1)
    
    async def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job status and details."""
        job_key = f"job:{job_id}"
        
        job_data = self.redis_client.hgetall(job_key)
        if not job_data:
            # Try to get from Celery result backend
            from celery.result import AsyncResult
            from app.celery_app import celery_app
            
            result = AsyncResult(job_id, app=celery_app)
            
            if result.id:
                return {
                    "id": job_id,
                    "status": result.state,
                    "result": result.result if result.ready() else None,
                    "info": result.info if not result.ready() else None,
                    "celery_status": True
                }
            
            return None
        
        # Get file statuses
        file_pattern = f"job:{job_id}:file:*"
        file_keys = self.redis_client.keys(file_pattern)
        files = []
        
        for file_key in file_keys:
            file_data = self.redis_client.hgetall(file_key)
            if file_data:
                files.append(file_data)
        
        # Deserialize files list from JSON
        if "files" in job_data and isinstance(job_data["files"], str):
            try:
                job_data["files"] = json.loads(job_data["files"])
            except json.JSONDecodeError:
                job_data["files"] = []
        
        # Add file details
        job_data["file_details"] = files
        
        # Get logs
        log_key = f"job:{job_id}:logs"
        logs = self.redis_client.lrange(log_key, 0, -1)
        job_data["logs"] = [json.loads(log) for log in logs]
        
        # Deserialize logs if needed
        if "logs" in job_data and isinstance(job_data["logs"], str):
            try:
                job_data["logs"] = json.loads(job_data["logs"])
            except json.JSONDecodeError:
                job_data["logs"] = []
        
        # Calculate estimated time remaining
        if job_data.get("status") == JobStatus.PROCESSING.value:
            progress = float(job_data.get("progress", 0))
            if progress > 0 and job_data.get("started_at"):
                started = datetime.fromisoformat(job_data["started_at"])
                elapsed = (datetime.utcnow() - started).total_seconds()
                if progress > 0:
                    total_estimated = elapsed / (progress / 100)
                    remaining = total_estimated - elapsed
                    job_data["estimated_time_remaining"] = max(0, int(remaining))
        
        return job_data
    
    async def get_all_jobs(
        self,
        limit: int = 100,
        offset: int = 0,
        status_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get all jobs with optional filtering."""
        job_ids = self.redis_client.lrange("jobs:all", offset, offset + limit - 1)
        
        jobs = []
        for job_id in job_ids:
            job_data = await self.get_job_status(job_id)
            if job_data:
                if status_filter is None or job_data.get("status") == status_filter:
                    # Don't include full file list and logs in list view
                    job_summary = {
                        k: v for k, v in job_data.items()
                        if k not in ["files", "logs"]
                    }
                    jobs.append(job_summary)
        
        return jobs
    
    async def get_active_jobs(self) -> List[Dict[str, Any]]:
        """Get all active jobs."""
        job_ids = self.redis_client.smembers("jobs:active")
        
        jobs = []
        for job_id in job_ids:
            job_data = await self.get_job_status(job_id)
            if job_data:
                jobs.append(job_data)
        
        return jobs
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a job."""
        from celery.result import AsyncResult
        from app.celery_app import celery_app
        
        # Cancel Celery task
        result = AsyncResult(job_id, app=celery_app)
        result.revoke(terminate=True)
        
        # Update job status
        await self.update_job_status(
            job_id,
            JobStatus.CANCELLED,
            error="Job cancelled by user"
        )
        
        return True
    
    async def cleanup_old_jobs(self, hours: int = 24):
        """Clean up jobs older than specified hours."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        
        all_jobs = self.redis_client.lrange("jobs:all", 0, -1)
        removed = 0
        
        for job_id in all_jobs:
            job_key = f"job:{job_id}"
            job_data = self.redis_client.hgetall(job_key)
            
            if job_data and "created_at" in job_data:
                created = datetime.fromisoformat(job_data["created_at"])
                if created < cutoff:
                    # Delete job data
                    self.redis_client.delete(job_key)
                    
                    # Delete file data
                    file_pattern = f"job:{job_id}:file:*"
                    file_keys = self.redis_client.keys(file_pattern)
                    for file_key in file_keys:
                        self.redis_client.delete(file_key)
                    
                    # Delete logs
                    log_key = f"job:{job_id}:logs"
                    self.redis_client.delete(log_key)
                    
                    # Remove from lists
                    self.redis_client.lrem("jobs:all", 0, job_id)
                    self.redis_client.srem("jobs:active", job_id)
                    
                    removed += 1
        
        logger.info(f"Cleaned up {removed} old jobs")
        return removed
    
    async def get_job_metrics(self) -> Dict[str, Any]:
        """Get overall job metrics."""
        total_jobs = self.redis_client.llen("jobs:all")
        active_jobs = self.redis_client.scard("jobs:active")
        
        # Get status distribution
        all_jobs = self.redis_client.lrange("jobs:all", 0, 99)  # Sample first 100
        status_counts = {}
        
        for job_id in all_jobs:
            job_key = f"job:{job_id}"
            status = self.redis_client.hget(job_key, "status")
            if status:
                status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            "total_jobs": total_jobs,
            "active_jobs": active_jobs,
            "status_distribution": status_counts,
            "timestamp": datetime.utcnow().isoformat()
        }


# Create global instance
redis_job_tracker = RedisJobTracker()