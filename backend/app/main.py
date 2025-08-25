"""
VerixAI Main Application Module

This module initializes the FastAPI application and configures all middleware,
routers, and settings. All API endpoints are organized in separate router modules
for better maintainability and separation of concerns.

Router Organization:
- system: Health checks and system status
- documents: Document upload and querying
- datasets: Dataset management
- jobs: Background job tracking
- summarization: Document summarization features
- chat: Interactive chat with documents
- analytics: CSV data analytics
- providers: LLM provider management
"""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pathlib import Path

from app.core.config import settings
from app.api.providers import router as providers_router
from app.api.routes import (
    system_router,
    documents_router,
    datasets_router,
    jobs_router,
    summarization_router,
    chat_router,
    analytics_router
)

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

# Create necessary directories
Path(settings.upload_directory).mkdir(parents=True, exist_ok=True)
Path(settings.datasets_directory).mkdir(parents=True, exist_ok=True)

# Include all API routers
app.include_router(system_router)
app.include_router(documents_router)
app.include_router(datasets_router)
app.include_router(jobs_router)
app.include_router(providers_router)
app.include_router(summarization_router)
app.include_router(chat_router)
app.include_router(analytics_router)

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        workers=1 if settings.debug else settings.workers
    )