"""
Document Upload and Query API Routes
"""

import logging
from typing import List, Optional
from pathlib import Path
import aiofiles
import json

from fastapi import APIRouter, HTTPException, UploadFile, File, Form

from app.core.config import settings
from app.agents.document_agent import DocumentOrchestrator
from app.agents.multi_agent_system import MultiAgentOrchestrator
from app.schemas.requests import QueryRequest
from app.schemas.responses import QueryResponse, IngestionResponse
from app.services.redis_job_tracker import redis_job_tracker
from app.tasks.document_tasks import process_document_batch, process_with_multi_agent

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["documents"])

# Initialize orchestrators
orchestrator = DocumentOrchestrator()
multi_agent_orchestrator = MultiAgentOrchestrator()

# Determine which orchestrator to use based on configuration
use_multi_agent = settings.multi_agent_enabled
use_async_processing = settings.async_processing_enabled


@router.post("/upload")
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
            raise HTTPException(status_code=400, detail="No valid files to process")
        
        metadata_dict = json.loads(metadata) if metadata else {}
        
        if use_celery:
            # Use Celery for background processing
            if use_multi_agent:
                task = process_with_multi_agent.delay(
                    file_paths=file_paths,
                    dataset_name=dataset_name,
                    metadata=metadata_dict
                )
            else:
                task = process_document_batch.delay(
                    file_paths=file_paths,
                    dataset_name=dataset_name,
                    metadata=metadata_dict
                )
            
            # Create job entry
            job_id = await redis_job_tracker.create_job(
                job_type="document_processing",
                dataset_name=dataset_name,
                total_items=len(file_paths),
                metadata={
                    "files": [Path(fp).name for fp in file_paths],
                    "celery_task_id": task.id
                }
            )
            
            return {
                "status": "processing",
                "job_id": job_id,
                "message": f"Processing {len(file_paths)} documents in background",
                "celery_task_id": task.id
            }
        else:
            # Process synchronously
            if use_multi_agent:
                result = await multi_agent_orchestrator.process_documents_multi_agent(
                    file_paths=file_paths,
                    dataset_name=dataset_name,
                    metadata=metadata_dict
                )
            else:
                result = await orchestrator.process_documents(
                    file_paths=file_paths,
                    dataset_name=dataset_name,
                    metadata=metadata_dict
                )
            
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


@router.post("/query", response_model=QueryResponse)
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