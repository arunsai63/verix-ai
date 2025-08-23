import logging
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
import uvicorn
from pathlib import Path
import aiofiles
import os

from app.core.config import settings
from app.agents.document_agent import DocumentOrchestrator
from app.schemas.requests import QueryRequest, DatasetRequest
from app.schemas.responses import QueryResponse, IngestionResponse, DatasetInfo
from app.api.providers import router as providers_router

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

orchestrator = DocumentOrchestrator()

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


@app.post("/api/upload", response_model=IngestionResponse)
async def upload_documents(
    files: List[UploadFile] = File(...),
    dataset_name: str = Form(...),
    metadata: Optional[str] = Form(None)
):
    """
    Upload and process documents into a dataset.
    
    Args:
        files: List of files to upload
        dataset_name: Name of the dataset
        metadata: Optional JSON metadata string
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