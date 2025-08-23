from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class Citation(BaseModel):
    """Citation model for source references."""
    source_number: int
    filename: str
    dataset: str
    chunk_index: int
    snippet: str
    relevance_score: float


class QueryResponse(BaseModel):
    """Response model for document queries."""
    status: str
    query: str
    answer: str
    citations: List[Citation]
    highlights: List[str]
    confidence: str
    role: str
    sources_count: int
    disclaimer: Optional[str] = None
    suggested_followup: Optional[str] = None


class IngestionResponse(BaseModel):
    """Response model for document ingestion."""
    status: str
    dataset_name: str
    documents_processed: int
    chunks_created: int
    message: str
    errors: Optional[List[Dict[str, str]]] = None


class DatasetInfo(BaseModel):
    """Information about a dataset."""
    name: str
    document_count: int
    created_at: float
    size_bytes: int
    last_updated: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    detail: Optional[str] = None
    status_code: int = 500