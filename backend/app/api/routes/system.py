"""
System and Health Check API Routes
"""

import os
from fastapi import APIRouter

from app.core.config import settings

router = APIRouter(tags=["system"])


@router.get("/")
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
            "health": "/health",
            "docs": "/docs"
        }
    }


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": settings.app_name}


@router.get("/api/system/status")
async def system_status():
    """Get system status and configuration."""
    return {
        "status": "operational",
        "configuration": {
            "multi_agent_enabled": settings.multi_agent_enabled,
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
        ] if settings.multi_agent_enabled else ["DocumentOrchestrator"]
    }