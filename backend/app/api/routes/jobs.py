"""
Job Management API Routes
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException

from app.services.redis_job_tracker import redis_job_tracker

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/jobs", tags=["jobs"])


@router.get("/{job_id}/status")
async def get_job_status(job_id: str):
    """Get the status of a document processing job."""
    job_status = await redis_job_tracker.get_job_status(job_id)
    
    if not job_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return job_status


@router.get("")
async def get_all_jobs(
    limit: int = 100,
    offset: int = 0,
    status: Optional[str] = None
):
    """Get all jobs with pagination and filtering."""
    jobs = await redis_job_tracker.get_all_jobs(limit, offset, status)
    return {
        "jobs": jobs,
        "total": len(jobs),
        "limit": limit,
        "offset": offset
    }


@router.get("/active")
async def get_active_jobs():
    """Get all currently active jobs."""
    jobs = await redis_job_tracker.get_active_jobs()
    return {"active_jobs": jobs, "total": len(jobs)}


@router.delete("/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a running job."""
    success = await redis_job_tracker.cancel_job(job_id)
    if success:
        return {"status": "success", "message": f"Job {job_id} cancelled"}
    else:
        raise HTTPException(status_code=404, detail="Job not found or already completed")


@router.post("/cleanup")
async def cleanup_old_jobs(hours: int = 24):
    """Clean up jobs older than specified hours."""
    removed = await redis_job_tracker.cleanup_old_jobs(hours)
    return {
        "status": "success",
        "removed_jobs": removed,
        "message": f"Removed {removed} jobs older than {hours} hours"
    }


@router.get("/metrics")
async def get_job_metrics():
    """Get overall job processing metrics."""
    metrics = await redis_job_tracker.get_job_metrics()
    return metrics