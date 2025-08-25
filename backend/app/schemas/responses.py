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


class SummarizationResponse(BaseModel):
    """Response model for document summarization."""
    status: str
    summary: str
    summary_type: str
    length: str
    word_count: int
    key_topics: List[str]
    confidence_score: float
    processing_time: float
    citations: Optional[List[Dict[str, Any]]] = None
    cached: bool = False


class ChatSessionResponse(BaseModel):
    """Response model for chat session creation."""
    status: str
    session_id: str
    dataset_names: List[str]
    created_at: str
    message: str = "Chat session created successfully"


class ChatMessageResponse(BaseModel):
    """Response model for chat message."""
    status: str
    session_id: str
    message: str
    citations: Optional[List[Dict[str, Any]]] = None
    analytics: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: str


class ChatHistoryResponse(BaseModel):
    """Response model for chat history."""
    status: str
    session_id: str
    dataset_names: List[str]
    messages: List[Dict[str, Any]]
    created_at: str
    updated_at: str
    total_messages: int


class CSVAnalyticsResponse(BaseModel):
    """Response model for CSV analytics."""
    status: str
    analysis_results: Dict[str, Any]
    summary: str
    data_preview: Dict[str, Any]
    visualizations: Optional[List[Dict[str, Any]]] = None
    statistics: Optional[Dict[str, Any]] = None
    query_sql: Optional[str] = None
    processing_time: float