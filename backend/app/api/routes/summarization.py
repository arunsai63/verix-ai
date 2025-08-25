"""
Summarization API Routes
"""

import logging
from fastapi import APIRouter, HTTPException
from pathlib import Path

from app.schemas.requests import SummarizationRequest
from app.schemas.responses import SummarizationResponse
from app.services.summarization_service import (
    SummarizationService, 
    SummaryRequest as ServiceSummaryRequest,
    SummaryType,
    SummaryLength
)
from app.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/summarize", tags=["summarization"])

# Initialize service
summarization_service = SummarizationService()


@router.post("", response_model=SummarizationResponse)
async def summarize_document(request: SummarizationRequest):
    """
    Generate a summary of a document or dataset.
    
    Args:
        request: Summarization request with content or dataset info
    """
    try:
        # Validate input
        if not any([request.dataset_name, request.document_name, request.content]):
            raise HTTPException(
                status_code=400,
                detail="Must provide either dataset_name, document_name, or content"
            )
        
        # Handle dataset summarization
        if request.dataset_name and not request.content:
            response = await summarization_service.summarize_dataset(
                dataset_name=request.dataset_name,
                summary_type=request.summary_type,
                length=request.length
            )
        else:
            # Handle direct content summarization
            content = request.content
            if not content and request.document_name:
                # Load document content
                dataset_dir = Path(settings.datasets_directory) / request.dataset_name
                doc_path = dataset_dir / request.document_name
                if doc_path.exists():
                    with open(doc_path, 'r') as f:
                        content = f.read()
                else:
                    raise HTTPException(status_code=404, detail="Document not found")
            
            # Create service request
            service_request = ServiceSummaryRequest(
                content=content,
                summary_type=request.summary_type,
                length=request.length,
                dataset_name=request.dataset_name,
                document_name=request.document_name,
                custom_instructions=request.custom_instructions,
                include_citations=request.include_citations
            )
            
            response = await summarization_service.summarize_document(service_request)
        
        return SummarizationResponse(
            status="success",
            summary=response.summary,
            summary_type=response.summary_type,
            length=response.length,
            word_count=response.word_count,
            key_topics=response.key_topics,
            confidence_score=response.confidence_score,
            processing_time=response.processing_time,
            citations=response.citations,
            cached=response.cached
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Summarization error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/types")
async def get_summary_types():
    """Get available summary types and lengths."""
    return {
        "summary_types": [t.value for t in SummaryType],
        "summary_lengths": [l.value for l in SummaryLength],
        "descriptions": {
            "types": {
                "executive": "Executive summary for leadership",
                "key_points": "Main points and highlights",
                "chapter_wise": "Section-by-section breakdown",
                "technical": "Technical details and specifications",
                "bullet_points": "Concise bullet point format",
                "abstract": "Academic-style abstract"
            },
            "lengths": {
                "brief": "1-2 paragraphs (150-300 words)",
                "standard": "1 page (500-750 words)",
                "detailed": "2-3 pages (1500-2000 words)"
            }
        }
    }