"""
Enhanced API Routes for Advanced Retrieval and Processing.
Exposes the new Phase 1 capabilities via REST endpoints.
"""

import logging
from typing import List, Optional, Dict, Any, Literal
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Query
from pydantic import BaseModel, Field

from app.services.enhanced_document_processor import EnhancedDocumentProcessor
from app.services.enhanced_retrieval_service import EnhancedRetrievalService
from app.services.dataset_service import DatasetService
from app.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/enhanced", tags=["enhanced"])


class EnhancedUploadRequest(BaseModel):
    """Request model for enhanced document upload."""
    dataset_name: str = Field(..., description="Dataset name")
    chunking_strategy: Literal["semantic", "hierarchical", "dynamic", "auto", "hybrid"] = Field(
        "auto", description="Chunking strategy to use"
    )
    max_chunk_size: int = Field(1500, description="Maximum chunk size")
    min_chunk_size: int = Field(100, description="Minimum chunk size")
    enable_hierarchical: bool = Field(True, description="Preserve document hierarchy")
    optimize_chunking: bool = Field(False, description="Optimize chunking parameters")


class EnhancedQueryRequest(BaseModel):
    """Request model for enhanced query."""
    query: str = Field(..., description="Search query")
    dataset_name: str = Field(..., description="Dataset to search")
    top_k: int = Field(10, description="Number of results")
    retrieval_strategy: Literal["hybrid", "semantic", "keyword", "auto"] = Field(
        "hybrid", description="Retrieval strategy"
    )
    enable_reranking: bool = Field(True, description="Enable cross-encoder reranking")
    enable_query_expansion: bool = Field(True, description="Enable query expansion")
    enable_hyde: bool = Field(True, description="Enable HyDE")
    role: Optional[str] = Field(None, description="User role for personalization")
    filters: Optional[Dict[str, Any]] = Field(None, description="Metadata filters")


class MultiHopQueryRequest(BaseModel):
    """Request model for multi-hop query."""
    query: str = Field(..., description="Initial query")
    dataset_name: str = Field(..., description="Dataset to search")
    max_hops: int = Field(3, description="Maximum retrieval hops")
    top_k_per_hop: int = Field(5, description="Results per hop")


class ChunkingMetricsResponse(BaseModel):
    """Response model for chunking metrics."""
    total_chunks: int
    avg_coherence: float
    size_consistency: float
    coverage: float
    avg_size: float
    size_std: float


# Initialize services
enhanced_processor = EnhancedDocumentProcessor()
enhanced_retrieval = EnhancedRetrievalService()
dataset_service = DatasetService()


@router.post("/upload")
async def enhanced_upload(
    files: List[UploadFile] = File(...),
    dataset_name: str = Form(...),
    chunking_strategy: str = Form("auto"),
    max_chunk_size: int = Form(1500),
    min_chunk_size: int = Form(100),
    enable_hierarchical: bool = Form(True),
    optimize_chunking: bool = Form(False)
):
    """
    Upload documents with enhanced chunking capabilities.
    
    Features:
    - Semantic chunking based on meaning boundaries
    - Hierarchical structure preservation
    - Dynamic chunk sizing based on content complexity
    - Automatic strategy selection
    - Hybrid chunking combining multiple strategies
    """
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")
        
        # Save files
        file_paths = []
        dataset_dir = Path(settings.datasets_directory) / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        for file in files:
            if not file.filename:
                continue
            
            file_path = dataset_dir / file.filename
            content = await file.read()
            
            with open(file_path, 'wb') as f:
                f.write(content)
            
            file_paths.append(str(file_path))
        
        # Optimize chunking if requested
        if optimize_chunking and len(file_paths) >= 3:
            optimized_params = enhanced_processor.optimize_chunking_for_dataset(
                sample_files=file_paths[:3]
            )
            logger.info(f"Optimized parameters: {optimized_params}")
        
        # Process files with enhanced chunking
        processed_docs = []
        for file_path in file_paths:
            doc = enhanced_processor.process_file(
                file_path=file_path,
                dataset_name=dataset_name,
                chunking_strategy=chunking_strategy,
                enable_hierarchical=enable_hierarchical
            )
            processed_docs.append(doc)
        
        # Store in dataset
        await dataset_service.store_documents(
            dataset_name=dataset_name,
            documents=processed_docs
        )
        
        # Calculate overall metrics
        all_chunks = []
        for doc in processed_docs:
            all_chunks.extend(doc.get("chunks", []))
        
        overall_metrics = enhanced_processor.chunker.evaluate_chunks(all_chunks)
        
        return {
            "status": "success",
            "dataset_name": dataset_name,
            "files_processed": len(processed_docs),
            "total_chunks": len(all_chunks),
            "chunking_strategy": chunking_strategy,
            "metrics": overall_metrics,
            "documents": [
                {
                    "filename": doc["metadata"]["filename"],
                    "chunks": len(doc["chunks"]),
                    "metrics": doc.get("chunking_metrics", {})
                }
                for doc in processed_docs
            ]
        }
        
    except Exception as e:
        logger.error(f"Enhanced upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query")
async def enhanced_query(request: EnhancedQueryRequest):
    """
    Perform enhanced retrieval with advanced RAG mechanisms.
    
    Features:
    - Hybrid retrieval (BM25 + semantic)
    - Query expansion with T5/BERT
    - HyDE (Hypothetical Document Embeddings)
    - Cross-encoder reranking
    - Cascade reranking pipeline
    - MMR diversity optimization
    """
    try:
        result = await enhanced_retrieval.retrieve(
            query=request.query,
            dataset_name=request.dataset_name,
            top_k=request.top_k,
            filters=request.filters,
            role=request.role,
            use_adaptive=request.retrieval_strategy == "auto"
        )
        
        return {
            "status": "success",
            **result
        }
        
    except Exception as e:
        logger.error(f"Enhanced query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query/multi-hop")
async def multi_hop_query(request: MultiHopQueryRequest):
    """
    Perform multi-hop retrieval for complex queries.
    
    Iteratively retrieves information, using previous results
    to generate follow-up queries for comprehensive coverage.
    """
    try:
        result = await enhanced_retrieval.retrieve_multi_hop(
            query=request.query,
            dataset_name=request.dataset_name,
            max_hops=request.max_hops,
            top_k_per_hop=request.top_k_per_hop
        )
        
        return {
            "status": "success",
            **result
        }
        
    except Exception as e:
        logger.error(f"Multi-hop query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/chunking/strategies")
async def get_chunking_strategies():
    """Get available chunking strategies and their descriptions."""
    return {
        "strategies": {
            "semantic": "Chunks based on semantic boundaries using sentence embeddings",
            "hierarchical": "Preserves document structure with parent-child relationships",
            "dynamic": "Adjusts chunk size based on content complexity",
            "auto": "Automatically selects best strategy based on content",
            "hybrid": "Combines multiple strategies for optimal results"
        }
    }


@router.get("/retrieval/strategies")
async def get_retrieval_strategies():
    """Get available retrieval strategies and their descriptions."""
    return {
        "strategies": {
            "hybrid": "Combines BM25 keyword search with semantic similarity",
            "semantic": "Pure semantic similarity search using embeddings",
            "keyword": "BM25-based keyword search with TF-IDF scoring",
            "auto": "Automatically selects strategy based on query characteristics"
        }
    }


@router.post("/analyze/chunks")
async def analyze_chunks(
    dataset_name: str = Form(...),
    sample_size: int = Form(100)
):
    """
    Analyze chunk quality for a dataset.
    
    Returns metrics on coherence, size distribution, and coverage.
    """
    try:
        # Get sample chunks from dataset
        chunks = await dataset_service.get_sample_chunks(
            dataset_name=dataset_name,
            sample_size=sample_size
        )
        
        if not chunks:
            raise HTTPException(status_code=404, detail="No chunks found")
        
        # Evaluate chunks
        metrics = enhanced_processor.chunker.evaluate_chunks(chunks)
        
        # Additional analysis
        sizes = [len(c.get("content", "")) for c in chunks]
        
        return {
            "status": "success",
            "dataset": dataset_name,
            "sample_size": len(chunks),
            "metrics": metrics,
            "size_distribution": {
                "min": min(sizes),
                "max": max(sizes),
                "mean": sum(sizes) / len(sizes),
                "median": sorted(sizes)[len(sizes) // 2]
            }
        }
        
    except Exception as e:
        logger.error(f"Chunk analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Check health of enhanced services."""
    return {
        "status": "healthy",
        "services": {
            "enhanced_processor": "active",
            "enhanced_retrieval": "active",
            "chunking_strategies": ["semantic", "hierarchical", "dynamic", "auto", "hybrid"],
            "retrieval_features": ["query_expansion", "hyde", "reranking", "multi_hop"]
        }
    }