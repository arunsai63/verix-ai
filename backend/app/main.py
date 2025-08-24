import logging
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Optional
import uvicorn
from pathlib import Path
import aiofiles
import os
import asyncio

from app.core.config import settings
from app.agents.document_agent import DocumentOrchestrator
from app.agents.multi_agent_system import MultiAgentOrchestrator
from app.schemas.requests import QueryRequest, DatasetRequest
from app.schemas.responses import QueryResponse, IngestionResponse, DatasetInfo
from app.api.providers import router as providers_router
from app.services.redis_job_tracker import redis_job_tracker
from app.tasks.document_tasks import process_document_batch, process_with_multi_agent
from celery.result import AsyncResult
from app.celery_app import celery_app

logging.basicConfig(level=getattr(logging, settings.log_level))
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="AI assistant for document analysis with citations"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(providers_router)

# Initialize orchestrators
orchestrator = DocumentOrchestrator()
multi_agent_orchestrator = MultiAgentOrchestrator()

# Determine which orchestrator to use based on configuration
use_multi_agent = settings.multi_agent_enabled
use_async_processing = settings.async_processing_enabled

Path(settings.upload_directory).mkdir(parents=True, exist_ok=True)
Path(settings.datasets_directory).mkdir(parents=True, exist_ok=True)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "endpoints": {
            "upload": "/api/upload",
            "query": "/api/query",
            "datasets": "/api/datasets",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": settings.app_name}


@app.get("/api/system/status")
async def system_status():
    """Get system status and configuration."""
    return {
        "status": "operational",
        "configuration": {
            "multi_agent_enabled": use_multi_agent,
            "async_processing": os.getenv("ASYNC_PROCESSING_ENABLED", "false").lower() == "true",
            "llm_provider": settings.llm_provider,
            "embedding_provider": settings.embedding_provider,
            "max_parallel_documents": int(os.getenv("MAX_PARALLEL_DOCUMENTS", "5")),
            "max_agents_per_query": int(os.getenv("MAX_AGENTS_PER_QUERY", "6"))
        },
        "agents": [
            "DocumentIngestionAgent",
            "ParallelRetrievalAgent", 
            "RankingAgent",
            "SummarizationAgent",
            "CitationAgent",
            "ValidationAgent"
        ] if use_multi_agent else ["DocumentOrchestrator"]
    }



@app.post("/api/upload")
async def upload_documents(
    files: List[UploadFile] = File(...),
    dataset_name: str = Form(...),
    metadata: Optional[str] = Form(None),
    use_celery: Optional[bool] = Form(True)
):
    """
    Upload and process documents into a dataset using Celery queue.
    
    Args:
        files: List of files to upload
        dataset_name: Name of the dataset
        metadata: Optional JSON metadata string
        use_celery: Use Celery for background processing (default: True)
    
    Returns:
        Job ID for tracking the processing status
    """
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")
        
        file_paths = []
        dataset_dir = Path(settings.datasets_directory) / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        for file in files:
            if not file.filename:
                continue
            
            file_extension = Path(file.filename).suffix.lower()[1:]
            if file_extension not in settings.allowed_extensions:
                logger.warning(f"Skipping unsupported file type: {file.filename}")
                continue
            
            file_path = dataset_dir / file.filename
            
            async with aiofiles.open(file_path, 'wb') as f:
                content = await file.read()
                await f.write(content)
            
            file_paths.append(str(file_path))
        
        if not file_paths:
            raise HTTPException(
                status_code=400,
                detail="No valid files to process"
            )
        
        metadata_dict = {}
        if metadata:
            import json
            try:
                metadata_dict = json.loads(metadata)
            except json.JSONDecodeError:
                logger.warning("Invalid metadata JSON, ignoring")
        
        if use_celery:
            # Use Celery for background processing
            if use_multi_agent and settings.multi_agent_enabled:
                # Use multi-agent processing task
                task = process_with_multi_agent.apply_async(
                    args=[file_paths, dataset_name, metadata_dict],
                    queue="documents"
                )
            else:
                # Use batch processing task
                task = process_document_batch.apply_async(
                    args=[file_paths, dataset_name, metadata_dict],
                    kwargs={"batch_size": min(5, len(file_paths))},
                    queue="documents"
                )
            
            return {
                "status": "queued",
                "job_id": task.id,
                "dataset_name": dataset_name,
                "total_files": len(file_paths),
                "message": f"Processing {len(file_paths)} documents in background. Use job_id to track progress.",
                "tracking_url": f"/api/jobs/{task.id}/status"
            }
        else:
            # Fallback to synchronous processing (not recommended for production)
            logger.warning("Using synchronous processing - not recommended for large files")
            
            if use_multi_agent:
                result = await multi_agent_orchestrator.ingest_documents_multi_agent(
                    file_paths, dataset_name, metadata_dict
                )
            else:
                result = await orchestrator.ingest_dataset(
                    file_paths, dataset_name, metadata_dict
                )
            
            if result["status"] == "error":
                raise HTTPException(status_code=500, detail=result.get("error"))
            
            return IngestionResponse(
                status="success",
                dataset_name=dataset_name,
                documents_processed=result.get("documents_processed", 0),
                chunks_created=result.get("chunks_created", 0),
                message=f"Successfully processed {len(file_paths)} documents"
            )
        
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query documents and get AI-generated answer with citations.
    
    Args:
        request: Query request with question and parameters
    """
    try:
        if not request.query:
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Use multi-agent system if enabled
        if use_multi_agent:
            result = await multi_agent_orchestrator.process_query_multi_agent(
                query=request.query,
                dataset_names=request.dataset_names,
                role=request.role,
                k=request.max_results
            )
        else:
            result = await orchestrator.process_query(
                query=request.query,
                dataset_names=request.dataset_names,
                role=request.role,
                k=request.max_results
            )
        
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result.get("error"))
        
        return QueryResponse(
            status=result["status"],
            query=request.query,
            answer=result.get("answer", ""),
            citations=result.get("citations", []),
            highlights=result.get("highlights", []),
            confidence=result.get("confidence", "medium"),
            role=request.role,
            sources_count=result.get("sources_count", 0)
        )
        
    except Exception as e:
        logger.error(f"Query error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/datasets", response_model=List[DatasetInfo])
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


@app.get("/api/jobs/{job_id}/status")
async def get_job_status(job_id: str):
    """Get the status of a document processing job."""
    job_status = await redis_job_tracker.get_job_status(job_id)
    
    if not job_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return job_status


@app.get("/api/jobs")
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


@app.get("/api/jobs/active")
async def get_active_jobs():
    """Get all currently active jobs."""
    jobs = await redis_job_tracker.get_active_jobs()
    return {"active_jobs": jobs, "total": len(jobs)}


@app.delete("/api/jobs/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a running job."""
    success = await redis_job_tracker.cancel_job(job_id)
    if success:
        return {"status": "success", "message": f"Job {job_id} cancelled"}
    else:
        raise HTTPException(status_code=404, detail="Job not found or already completed")


@app.post("/api/jobs/cleanup")
async def cleanup_old_jobs(hours: int = 24):
    """Clean up jobs older than specified hours."""
    removed = await redis_job_tracker.cleanup_old_jobs(hours)
    return {
        "status": "success",
        "removed_jobs": removed,
        "message": f"Removed {removed} jobs older than {hours} hours"
    }


@app.get("/api/jobs/metrics")
async def get_job_metrics():
    """Get overall job processing metrics."""
    metrics = await redis_job_tracker.get_job_metrics()
    return metrics


@app.delete("/api/datasets/{dataset_name}")
async def delete_dataset(dataset_name: str):
    """Delete a dataset and all its documents."""
    try:
        from app.services.vector_store import VectorStoreService
        vector_store = VectorStoreService()
        
        success = vector_store.delete_dataset(dataset_name)
        
        dataset_dir = Path(settings.datasets_directory) / dataset_name
        if dataset_dir.exists():
            import shutil
            shutil.rmtree(dataset_dir)
        
        if success:
            return {"status": "success", "message": f"Dataset {dataset_name} deleted"}
        else:
            raise HTTPException(status_code=500, detail="Failed to delete dataset")
            
    except Exception as e:
        logger.error(f"Error deleting dataset: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/datasets/{dataset_name}/stats")
async def get_dataset_stats(dataset_name: str):
    """Get statistics for a specific dataset."""
    try:
        from app.services.vector_store import VectorStoreService
        vector_store = VectorStoreService()
        
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


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        workers=1 if settings.debug else settings.workers
    )