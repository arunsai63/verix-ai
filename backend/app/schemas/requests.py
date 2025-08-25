from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal
from enum import Enum


class SummaryType(str, Enum):
    """Types of summaries available"""
    EXECUTIVE = "executive"
    KEY_POINTS = "key_points"
    CHAPTER_WISE = "chapter_wise"
    TECHNICAL = "technical"
    BULLET_POINTS = "bullet_points"
    ABSTRACT = "abstract"


class SummaryLength(str, Enum):
    """Available summary lengths"""
    BRIEF = "brief"
    STANDARD = "standard"
    DETAILED = "detailed"


class QueryRequest(BaseModel):
    """Request model for document queries."""
    query: str = Field(..., description="The search query or question")
    dataset_names: Optional[List[str]] = Field(
        None, 
        description="Optional list of dataset names to search within"
    )
    role: str = Field(
        "general", 
        description="User role: doctor, lawyer, hr, or general"
    )
    max_results: int = Field(
        10, 
        description="Maximum number of results to retrieve",
        ge=1,
        le=50
    )


class DatasetRequest(BaseModel):
    """Request model for dataset operations."""
    name: str = Field(..., description="Dataset name")
    description: Optional[str] = Field(None, description="Dataset description")
    metadata: Optional[Dict[str, Any]] = Field(
        None, 
        description="Additional metadata for the dataset"
    )


class DocumentUploadRequest(BaseModel):
    """Request model for document upload."""
    dataset_name: str = Field(..., description="Target dataset name")
    metadata: Optional[Dict[str, Any]] = Field(
        None, 
        description="Document metadata"
    )
    process_immediately: bool = Field(
        True, 
        description="Whether to process documents immediately"
    )


class SummarizationRequest(BaseModel):
    """Request model for document summarization."""
    dataset_name: Optional[str] = Field(None, description="Dataset to summarize")
    document_name: Optional[str] = Field(None, description="Specific document to summarize")
    content: Optional[str] = Field(None, description="Direct content to summarize")
    summary_type: SummaryType = Field(
        SummaryType.EXECUTIVE,
        description="Type of summary to generate"
    )
    length: SummaryLength = Field(
        SummaryLength.STANDARD,
        description="Length of the summary"
    )
    custom_instructions: Optional[str] = Field(
        None,
        description="Custom instructions for summarization"
    )
    include_citations: bool = Field(
        True,
        description="Include source citations in summary"
    )


class ChatSessionRequest(BaseModel):
    """Request model for creating chat session."""
    dataset_names: List[str] = Field(
        ...,
        description="Datasets to chat with"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Session metadata"
    )


class ChatMessageRequest(BaseModel):
    """Request model for chat message."""
    session_id: str = Field(..., description="Chat session ID")
    message: str = Field(..., description="User message")
    stream: bool = Field(
        False,
        description="Stream response (not implemented yet)"
    )


class CSVAnalyticsRequest(BaseModel):
    """Request model for CSV analytics."""
    dataset_name: str = Field(..., description="Dataset containing CSV")
    file_name: Optional[str] = Field(None, description="Specific CSV file")
    query: str = Field(..., description="Natural language analytics query")
    visualize: bool = Field(True, description="Generate visualizations")
    filters: Optional[Dict[str, Any]] = Field(None, description="Data filters")
    export_format: Optional[str] = Field(
        None,
        description="Export format (json, csv, html)"
    )