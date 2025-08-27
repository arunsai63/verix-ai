"""
Enhanced Document Processor with Advanced Chunking and Retrieval.
Integrates semantic chunking, hierarchical structure, and dynamic sizing.
"""

import os
import hashlib
from typing import Dict, List, Optional, Any, Literal
from pathlib import Path
import logging
from markitdown import MarkItDown
from datetime import datetime
import json

from app.services.advanced_retrieval.integrated_chunker import IntegratedChunker
from app.services.advanced_retrieval.hybrid_retriever import HybridRetriever
from app.services.advanced_retrieval.integrated_ranker import IntegratedRanker
from app.core.config import settings

logger = logging.getLogger(__name__)


class EnhancedDocumentProcessor:
    """Enhanced document processor with advanced chunking strategies."""
    
    def __init__(
        self,
        chunking_strategy: Literal["semantic", "hierarchical", "dynamic", "auto", "hybrid"] = "auto",
        max_chunk_size: int = 1500,
        min_chunk_size: int = 100
    ):
        """
        Initialize enhanced document processor.
        
        Args:
            chunking_strategy: Strategy for chunking documents
            max_chunk_size: Maximum chunk size
            min_chunk_size: Minimum chunk size
        """
        self.converter = MarkItDown()
        self.supported_extensions = [
            '.pdf', '.docx', '.pptx', '.html', '.txt', '.md', 
            '.csv', '.xlsx', '.xml', '.rtf', '.odt', '.epub'
        ]
        
        # Initialize advanced chunker
        self.chunker = IntegratedChunker(
            default_strategy=chunking_strategy,
            max_chunk_size=max_chunk_size,
            min_chunk_size=min_chunk_size
        )
        
        # Initialize hybrid retriever for advanced search
        self.retriever = HybridRetriever()
        
        # Initialize integrated ranker for result optimization
        self.ranker = IntegratedRanker()
        
        logger.info(f"Enhanced processor initialized with {chunking_strategy} chunking")
    
    def process_file(
        self,
        file_path: str,
        dataset_name: str,
        metadata: Optional[Dict[str, Any]] = None,
        chunking_strategy: Optional[str] = None,
        enable_hierarchical: bool = True
    ) -> Dict[str, Any]:
        """
        Process a file with advanced chunking strategies.
        
        Args:
            file_path: Path to the file
            dataset_name: Dataset name
            metadata: Additional metadata
            chunking_strategy: Override default chunking strategy
            enable_hierarchical: Enable hierarchical structure preservation
            
        Returns:
            Processed document with advanced chunks
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            if file_path.suffix.lower() not in self.supported_extensions:
                raise ValueError(f"Unsupported file type: {file_path.suffix}")
            
            # Convert to markdown
            result = self.converter.convert(str(file_path))
            markdown_content = result.text_content
            
            # Calculate file hash
            file_hash = self._calculate_file_hash(file_path)
            
            # Prepare document metadata
            doc_metadata = {
                "filename": file_path.name,
                "file_path": str(file_path),
                "file_extension": file_path.suffix.lower(),
                "file_size": file_path.stat().st_size,
                "file_hash": file_hash,
                "dataset_name": dataset_name,
                "processed_at": datetime.utcnow().isoformat(),
                "source_type": self._get_source_type(file_path.suffix.lower()),
                **(metadata or {})
            }
            
            # Apply advanced chunking
            chunks = self.chunker.chunk(
                content=markdown_content,
                strategy=chunking_strategy,
                metadata=doc_metadata
            )
            
            # Evaluate chunk quality
            quality_metrics = self.chunker.evaluate_chunks(chunks)
            
            processed_doc = {
                "content": markdown_content,
                "metadata": doc_metadata,
                "chunks": chunks,
                "chunking_metrics": quality_metrics,
                "chunking_strategy": chunking_strategy or self.chunker.default_strategy
            }
            
            # Add hierarchical structure if enabled
            if enable_hierarchical and len(chunks) > 0:
                hierarchy = self._extract_hierarchy(chunks)
                if hierarchy:
                    processed_doc["hierarchy"] = hierarchy
            
            logger.info(
                f"Processed {file_path.name}: "
                f"{len(chunks)} chunks, "
                f"coherence={quality_metrics.get('avg_coherence', 0):.2f}"
            )
            
            return processed_doc
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            raise
    
    def process_batch_parallel(
        self,
        file_paths: List[str],
        dataset_name: str,
        metadata: Optional[Dict[str, Any]] = None,
        max_workers: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Process multiple files in parallel with advanced chunking.
        
        Args:
            file_paths: List of file paths
            dataset_name: Dataset name
            metadata: Additional metadata
            max_workers: Maximum parallel workers
            
        Returns:
            List of processed documents
        """
        import concurrent.futures
        
        processed_docs = []
        errors = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(
                    self.process_file,
                    file_path,
                    dataset_name,
                    metadata
                ): file_path
                for file_path in file_paths
            }
            
            for future in concurrent.futures.as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    doc = future.result()
                    processed_docs.append(doc)
                except Exception as e:
                    logger.error(f"Failed to process {file_path}: {str(e)}")
                    errors.append({"file": file_path, "error": str(e)})
        
        if errors:
            logger.warning(f"Processing completed with {len(errors)} errors")
        
        return processed_docs
    
    def optimize_chunking_for_dataset(
        self,
        sample_files: List[str],
        target_metrics: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Optimize chunking parameters based on sample files.
        
        Args:
            sample_files: Sample files for optimization
            target_metrics: Target quality metrics
            
        Returns:
            Optimized parameters
        """
        if not target_metrics:
            target_metrics = {
                "coherence_weight": 0.5,
                "consistency_weight": 0.3,
                "coverage_weight": 0.2
            }
        
        # Process sample files
        sample_docs = []
        for file_path in sample_files[:3]:  # Limit to 3 files for speed
            try:
                result = self.converter.convert(file_path)
                sample_docs.append(result.text_content)
            except Exception as e:
                logger.warning(f"Skipping sample file {file_path}: {str(e)}")
        
        if not sample_docs:
            logger.warning("No valid sample documents for optimization")
            return {}
        
        # Optimize parameters
        optimized = self.chunker.optimize_parameters(sample_docs, target_metrics)
        
        # Update chunker with optimized parameters
        if optimized.get('similarity_threshold'):
            self.chunker.semantic_chunker.similarity_threshold = optimized['similarity_threshold']
        if optimized.get('max_chunk_size'):
            self.chunker.max_chunk_size = optimized['max_chunk_size']
        
        logger.info(f"Optimized chunking parameters: {optimized}")
        return optimized
    
    def _extract_hierarchy(self, chunks: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Extract hierarchical structure from chunks."""
        hierarchical_chunks = [
            c for c in chunks 
            if c.get('metadata', {}).get('chunk_type') == 'hierarchical'
        ]
        
        if not hierarchical_chunks:
            return None
        
        # Build hierarchy tree
        hierarchy = {
            "total_nodes": len(hierarchical_chunks),
            "max_depth": max(
                c['metadata'].get('level', 0) 
                for c in hierarchical_chunks
            ),
            "root_nodes": []
        }
        
        # Create node map
        node_map = {}
        for chunk in hierarchical_chunks:
            chunk_id = chunk['metadata'].get('chunk_id')
            if chunk_id:
                node_map[chunk_id] = {
                    'id': chunk_id,
                    'title': chunk['metadata'].get('title', 'Untitled'),
                    'level': chunk['metadata'].get('level', 0),
                    'children': []
                }
        
        # Build relationships
        for chunk in hierarchical_chunks:
            chunk_id = chunk['metadata'].get('chunk_id')
            parent_id = chunk['metadata'].get('parent_id')
            
            if parent_id and parent_id in node_map and chunk_id in node_map:
                node_map[parent_id]['children'].append(node_map[chunk_id])
            elif chunk_id in node_map and not parent_id:
                hierarchy['root_nodes'].append(node_map[chunk_id])
        
        return hierarchy
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def _get_source_type(self, extension: str) -> str:
        """Get source type from file extension."""
        type_mapping = {
            '.pdf': 'pdf',
            '.docx': 'word',
            '.pptx': 'powerpoint',
            '.html': 'web',
            '.txt': 'text',
            '.md': 'markdown',
            '.csv': 'spreadsheet',
            '.xlsx': 'spreadsheet',
            '.xml': 'structured',
            '.rtf': 'document',
            '.odt': 'document',
            '.epub': 'ebook'
        }
        return type_mapping.get(extension, 'unknown')