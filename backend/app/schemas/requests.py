from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


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