import os
import logging
import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path
from celery import Task, group, chord
from celery.exceptions import SoftTimeLimitExceeded
import redis
import json
from datetime import datetime

from app.celery_app import celery_app
from app.services.unified_file_processor import UnifiedFileProcessor
from app.services.vector_store import VectorStoreService
from app.agents.multi_agent_system import MultiAgentOrchestrator
from app.core.config import settings

logger = logging.getLogger(__name__)

# Redis client for job tracking
redis_client = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))

class DocumentTask(Task):
    """Base task with error handling and progress tracking."""
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Handle task failure."""
        job_id = kwargs.get("job_id")
        if job_id:
            self._update_job_status(job_id, "failed", error=str(exc))
        logger.error(f"Task {task_id} failed: {exc}")
    
    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Handle task retry."""
        job_id = kwargs.get("job_id")
        if job_id:
            self._add_job_log(job_id, f"Retrying task: {exc}", level="warning")
        logger.warning(f"Task {task_id} retrying: {exc}")
    
    def on_success(self, retval, task_id, args, kwargs):
        """Handle task success."""
        job_id = kwargs.get("job_id")
        if job_id:
            self._add_job_log(job_id, "Task completed successfully")
    
    def _update_job_status(self, job_id: str, status: str, **kwargs):
        """Update job status in Redis."""
        job_key = f"job:{job_id}"
        job_data = redis_client.get(job_key)
        if job_data:
            job = json.loads(job_data)
            job["status"] = status
            job["updated_at"] = datetime.utcnow().isoformat()
            job.update(kwargs)
            redis_client.set(job_key, json.dumps(job))
            redis_client.expire(job_key, 3600)  # Expire after 1 hour
    
    def _add_job_log(self, job_id: str, message: str, level: str = "info"):
        """Add log entry for job."""
        log_key = f"job:{job_id}:logs"
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": level,
            "message": message
        }
        redis_client.rpush(log_key, json.dumps(log_entry))
        redis_client.expire(log_key, 3600)


@celery_app.task(
    base=DocumentTask,
    bind=True,
    max_retries=3,
    default_retry_delay=60,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_jitter=True
)
def process_document(
    self,
    file_path: str,
    dataset_name: str,
    metadata: Optional[Dict[str, Any]] = None,
    job_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Process a single document using unified streaming approach.
    
    Args:
        file_path: Path to the document
        dataset_name: Name of the dataset
        metadata: Optional metadata
        job_id: Job ID for tracking
    
    Returns:
        Processing result
    """
    try:
        # Update task progress
        if job_id:
            file_name = Path(file_path).name
            self._update_file_progress(job_id, file_name, "processing")
        
        # Check file size for queue routing
        file_size = Path(file_path).stat().st_size
        if file_size > 50 * 1024 * 1024:  # > 50MB
            logger.info(f"Large file detected ({file_size / 1024 / 1024:.2f}MB), routing to large_documents queue")
            return process_large_document.apply_async(
                args=[file_path, dataset_name, metadata, job_id],
                queue="large_documents"
            ).get()
        
        # Use unified processor for all files
        processor = UnifiedFileProcessor()
        file_ext = Path(file_path).suffix.lower()
        
        # Validate file
        validation = processor.validate_file(file_path)
        if not validation["valid"]:
            raise ValueError(f"File validation failed: {validation['error']}")
        
        all_chunks = []
        total_chunks = 0
        
        # Process based on file type
        if file_ext == '.pdf':
            for batch in processor.process_pdf(file_path, dataset_name, metadata):
                all_chunks.extend(batch["chunks"])
                total_chunks = batch["total_chunks_so_far"]
                if job_id:
                    self._add_job_log(job_id, f"Processed {total_chunks} chunks")
        else:
            for batch in processor.process_file_generic(file_path, dataset_name, metadata):
                all_chunks.extend(batch["chunks"])
                total_chunks = batch["total_chunks_so_far"]
                if job_id:
                    self._add_job_log(job_id, f"Processed {total_chunks} chunks")
        
        # Store in vector database in batches
        vector_store = VectorStoreService()
        batch_size = 100
        chunks_stored = 0
        
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i + batch_size]
            stored = vector_store.add_document(
                dataset_name=dataset_name,
                chunks=batch,
                metadata=metadata
            )
            chunks_stored += stored
        
        # Update progress
        if job_id:
            self._update_file_progress(
                job_id, 
                file_name, 
                "completed",
                chunks=len(all_chunks)
            )
        
        return {
            "status": "success",
            "file": file_path,
            "chunks_created": len(all_chunks),
            "chunks_stored": chunks_stored
        }
        
    except SoftTimeLimitExceeded:
        logger.error(f"Task timeout for file: {file_path}")
        if job_id:
            self._update_file_progress(job_id, Path(file_path).name, "timeout")
        raise
    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        if job_id:
            self._update_file_progress(job_id, Path(file_path).name, "failed", error=str(e))
        raise
    
    def _update_file_progress(self, job_id: str, file_name: str, status: str, **kwargs):
        """Update individual file progress."""
        file_key = f"job:{job_id}:file:{file_name}"
        file_data = {
            "status": status,
            "updated_at": datetime.utcnow().isoformat(),
            **kwargs
        }
        redis_client.set(file_key, json.dumps(file_data))
        redis_client.expire(file_key, 3600)


@celery_app.task(
    base=DocumentTask,
    bind=True,
    max_retries=2,
    default_retry_delay=120
)
def process_document_batch(
    self,
    file_paths: List[str],
    dataset_name: str,
    metadata: Optional[Dict[str, Any]] = None,
    job_id: Optional[str] = None,
    batch_size: int = 5
) -> Dict[str, Any]:
    """
    Process multiple documents in parallel batches.
    
    Args:
        file_paths: List of file paths
        dataset_name: Dataset name
        metadata: Optional metadata
        job_id: Job ID for tracking
        batch_size: Number of documents to process in parallel
    
    Returns:
        Batch processing results
    """
    try:
        total_files = len(file_paths)
        successful = 0
        failed = 0
        total_chunks = 0
        
        # Create job if not provided
        if not job_id:
            job_id = self.request.id
            self._create_job(job_id, dataset_name, total_files, file_paths)
        
        # Process files in batches
        for i in range(0, total_files, batch_size):
            batch = file_paths[i:i + batch_size]
            
            # Create group of parallel tasks
            job_group = group(
                process_document.s(fp, dataset_name, metadata, job_id)
                for fp in batch
            )
            
            # Execute batch and collect results
            results = job_group.apply_async().get(timeout=300)
            
            # Process results
            for result in results:
                if result["status"] == "success":
                    successful += 1
                    total_chunks += result["chunks_created"]
                else:
                    failed += 1
            
            # Update job progress
            progress = ((i + len(batch)) / total_files) * 100
            self._update_job_progress(job_id, progress, successful, failed)
        
        # Final job update
        self._update_job_status(
            job_id,
            "completed" if failed == 0 else "completed_with_errors",
            documents_processed=successful,
            documents_failed=failed,
            chunks_created=total_chunks
        )
        
        return {
            "status": "success",
            "job_id": job_id,
            "documents_processed": successful,
            "documents_failed": failed,
            "chunks_created": total_chunks
        }
        
    except Exception as e:
        logger.error(f"Batch processing failed: {str(e)}")
        if job_id:
            self._update_job_status(job_id, "failed", error=str(e))
        raise
    
    def _create_job(self, job_id: str, dataset_name: str, total_files: int, file_paths: List[str]):
        """Create a new job in Redis."""
        job_data = {
            "id": job_id,
            "dataset_name": dataset_name,
            "total_files": total_files,
            "files": [Path(fp).name for fp in file_paths],
            "status": "processing",
            "progress": 0,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }
        redis_client.set(f"job:{job_id}", json.dumps(job_data))
        redis_client.expire(f"job:{job_id}", 3600)
        
        # Add to job list
        redis_client.lpush("jobs", job_id)
        redis_client.ltrim("jobs", 0, 99)  # Keep last 100 jobs
    
    def _update_job_progress(self, job_id: str, progress: float, successful: int, failed: int):
        """Update job progress."""
        job_key = f"job:{job_id}"
        job_data = redis_client.get(job_key)
        if job_data:
            job = json.loads(job_data)
            job["progress"] = round(progress, 2)
            job["documents_processed"] = successful
            job["documents_failed"] = failed
            job["updated_at"] = datetime.utcnow().isoformat()
            redis_client.set(job_key, json.dumps(job))
            redis_client.expire(job_key, 3600)


@celery_app.task(
    base=DocumentTask,
    bind=True,
    time_limit=3600,  # 1 hour hard limit
    soft_time_limit=3000  # 50 minutes soft limit
)
def process_large_document(
    self,
    file_path: str,
    dataset_name: str,
    metadata: Optional[Dict[str, Any]] = None,
    job_id: Optional[str] = None,
    chunk_size: int = 500,  # Smaller chunks for large files
    max_chunks: int = 10000  # Maximum chunks to prevent memory issues
) -> Dict[str, Any]:
    """
    Process large documents (>50MB) with optimized settings.
    
    Args:
        file_path: Path to large document
        dataset_name: Dataset name
        metadata: Optional metadata
        job_id: Job ID for tracking
        chunk_size: Size of each chunk
        max_chunks: Maximum number of chunks
    
    Returns:
        Processing result
    """
    try:
        file_size = Path(file_path).stat().st_size
        logger.info(f"Processing large document: {file_path} ({file_size / 1024 / 1024:.2f}MB)")
        
        if job_id:
            self._add_job_log(
                job_id, 
                f"Processing large file: {Path(file_path).name} ({file_size / 1024 / 1024:.2f}MB)"
            )
        
        # Check file type and use appropriate processor
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.pdf':
            # Use unified processor with streaming
            processor = UnifiedFileProcessor()
            
            # Validate file first
            validation = processor.validate_file(file_path)
            if not validation["valid"]:
                raise ValueError(f"File validation failed: {validation['error']}")
            
            all_chunks = []
            
            # Process PDF in streaming batches
            for batch in processor.process_pdf(
                file_path,
                dataset_name,
                metadata,
                progress_callback=lambda p: self._add_job_log(
                    job_id,
                    f"Processing progress: {p['progress']:.1f}% ({p['chunks_created']} chunks)"
                ) if job_id else None
            ):
                all_chunks.extend(batch["chunks"])
                
                # Log progress
                if job_id:
                    self._add_job_log(
                        job_id,
                        f"Processed batch: {batch['batch_size']} chunks (total: {batch['total_chunks_so_far']})"
                    )
            
            result = {
                "chunks": all_chunks,
                "metadata": batch["metadata"]  # Use last batch metadata
            }
        else:
            # Use unified processor for other file types
            processor = UnifiedFileProcessor()
            
            all_chunks = []
            for batch in processor.process_file_generic(
                file_path,
                dataset_name,
                metadata
            ):
                all_chunks.extend(batch["chunks"])
                
                if job_id:
                    self._add_job_log(
                        job_id,
                        f"Processed batch: {batch['batch_size']} chunks (total: {batch['total_chunks_so_far']})"
                    )
            
            result = {
                "chunks": all_chunks,
                "metadata": batch["metadata"]
            }
        
        # Check chunk limit
        chunks = result["chunks"]
        if len(chunks) > max_chunks:
            logger.warning(f"Chunk limit exceeded ({len(chunks)} > {max_chunks}), truncating")
            chunks = chunks[:max_chunks]
            if job_id:
                self._add_job_log(
                    job_id,
                    f"Large file truncated to {max_chunks} chunks",
                    level="warning"
                )
        
        # Store chunks in batches to avoid memory issues
        vector_store = VectorStoreService()
        batch_size = 100
        total_stored = 0
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            stored = vector_store.add_document(
                dataset_name=dataset_name,
                chunks=batch,
                metadata=result["metadata"]
            )
            total_stored += stored
            
            # Update progress
            if job_id:
                progress = ((i + len(batch)) / len(chunks)) * 100
                self._add_job_log(
                    job_id,
                    f"Stored {total_stored}/{len(chunks)} chunks ({progress:.1f}%)"
                )
        
        # Update final status
        if job_id:
            file_name = Path(file_path).name
            self._update_file_progress(
                job_id,
                file_name,
                "completed",
                chunks=len(chunks),
                file_size_mb=file_size / 1024 / 1024
            )
        
        return {
            "status": "success",
            "file": file_path,
            "file_size_mb": file_size / 1024 / 1024,
            "chunks_created": len(chunks),
            "chunks_stored": total_stored
        }
        
    except SoftTimeLimitExceeded:
        logger.error(f"Large document processing timeout: {file_path}")
        if job_id:
            self._update_file_progress(
                job_id,
                Path(file_path).name,
                "timeout",
                error="Processing timeout for large file"
            )
        raise
    except Exception as e:
        logger.error(f"Error processing large document {file_path}: {str(e)}")
        if job_id:
            self._update_file_progress(
                job_id,
                Path(file_path).name,
                "failed",
                error=str(e)
            )
        raise
    
    def _update_file_progress(self, job_id: str, file_name: str, status: str, **kwargs):
        """Update individual file progress."""
        file_key = f"job:{job_id}:file:{file_name}"
        file_data = {
            "status": status,
            "updated_at": datetime.utcnow().isoformat(),
            **kwargs
        }
        redis_client.set(file_key, json.dumps(file_data))
        redis_client.expire(file_key, 3600)


@celery_app.task(bind=True)
def process_with_multi_agent(
    self,
    file_paths: List[str],
    dataset_name: str,
    metadata: Optional[Dict[str, Any]] = None,
    job_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Process documents using multi-agent system.
    
    Args:
        file_paths: List of file paths
        dataset_name: Dataset name
        metadata: Optional metadata
        job_id: Job ID for tracking
    
    Returns:
        Processing result
    """
    try:
        # Run multi-agent processing
        async def run_multi_agent():
            orchestrator = MultiAgentOrchestrator()
            result = await orchestrator.ingest_documents_multi_agent(
                file_paths, dataset_name, metadata
            )
            return result
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(run_multi_agent())
        loop.close()
        
        return result
        
    except Exception as e:
        logger.error(f"Multi-agent processing failed: {str(e)}")
        raise