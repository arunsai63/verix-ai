"""
Integrated Chunking System combining semantic, hierarchical, and dynamic chunking.
Provides a unified interface for intelligent document chunking.
"""

import logging
from typing import List, Dict, Any, Optional, Generator
import time

from .semantic_chunker import SemanticChunker, ChunkConfig
from .hierarchical_chunker import HierarchicalChunker
from .dynamic_chunker import DynamicChunker, DynamicChunkConfig

logger = logging.getLogger(__name__)


class IntegratedChunker:
    """
    Unified chunking system integrating multiple chunking strategies.
    """
    
    def __init__(
        self,
        default_strategy: str = "semantic",
        max_chunk_size: int = 1500,
        min_chunk_size: int = 100
    ):
        """
        Initialize integrated chunker.
        
        Args:
            default_strategy: Default chunking strategy
            max_chunk_size: Maximum chunk size
            min_chunk_size: Minimum chunk size
        """
        self.default_strategy = default_strategy
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        
        # Initialize chunkers
        self.semantic_chunker = SemanticChunker(
            config=ChunkConfig(
                max_chunk_size=max_chunk_size,
                min_chunk_size=min_chunk_size
            )
        )
        
        self.hierarchical_chunker = HierarchicalChunker(
            max_chunk_size=max_chunk_size,
            min_chunk_size=min_chunk_size
        )
        
        self.dynamic_chunker = DynamicChunker(
            config=DynamicChunkConfig(
                base_size=max_chunk_size // 2,
                max_size=max_chunk_size,
                min_size=min_chunk_size
            )
        )
        
        logger.info(f"IntegratedChunker initialized with default strategy: {default_strategy}")
    
    def chunk(
        self,
        content: str,
        strategy: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Chunk content using specified strategy.
        
        Args:
            content: Content to chunk
            strategy: Chunking strategy (semantic, hierarchical, dynamic, auto)
            metadata: Document metadata
            **kwargs: Strategy-specific parameters
            
        Returns:
            List of chunks
        """
        strategy = strategy or self.default_strategy
        start_time = time.time()
        
        if strategy == "semantic":
            chunks = self.semantic_chunker.chunk_document(
                content,
                min_size=kwargs.get('min_size', self.min_chunk_size),
                max_size=kwargs.get('max_size', self.max_chunk_size),
                similarity_threshold=kwargs.get('similarity_threshold', 0.75)
            )
        
        elif strategy == "hierarchical":
            chunks = self.hierarchical_chunker.create_hierarchical_chunks(
                content,
                metadata=metadata,
                include_parent_context=kwargs.get('include_parent_context', True)
            )
        
        elif strategy == "dynamic":
            chunks = self.dynamic_chunker.chunk_document(
                content,
                base_size=kwargs.get('base_size'),
                preserve_format=kwargs.get('preserve_format', True)
            )
        
        elif strategy == "auto":
            # Automatically select best strategy
            chunks = self._auto_select_strategy(content, metadata)
        
        elif strategy == "hybrid":
            # Combine multiple strategies
            chunks = self._hybrid_chunking(content, metadata, **kwargs)
        
        else:
            logger.warning(f"Unknown strategy: {strategy}, using default")
            chunks = self.semantic_chunker.chunk_document(content)
        
        # Add metadata
        for i, chunk in enumerate(chunks):
            if 'metadata' not in chunk:
                chunk['metadata'] = {}
            chunk['metadata'].update({
                'strategy': strategy,
                'chunk_index': i,
                'total_chunks': len(chunks),
                **(metadata or {})
            })
        
        elapsed = time.time() - start_time
        logger.info(f"Chunked using {strategy} strategy: {len(chunks)} chunks in {elapsed:.2f}s")
        
        return chunks
    
    def _auto_select_strategy(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Automatically select best chunking strategy."""
        # Analyze content characteristics
        has_structure = bool(re.search(r'^#+ ', content, re.MULTILINE))
        has_code = bool(re.search(r'```|def |class ', content))
        content_length = len(content)
        
        # Select strategy based on characteristics
        if has_structure and content_length > 5000:
            strategy = "hierarchical"
        elif has_code or self.dynamic_chunker._detect_format(content):
            strategy = "dynamic"
        else:
            strategy = "semantic"
        
        logger.info(f"Auto-selected strategy: {strategy}")
        
        return self.chunk(content, strategy, metadata)
    
    def _hybrid_chunking(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Combine multiple chunking strategies."""
        # First pass: hierarchical for structure
        hierarchical_chunks = self.hierarchical_chunker.create_hierarchical_chunks(
            content,
            metadata=metadata
        )
        
        # Second pass: semantic chunking on leaf nodes
        hybrid_chunks = []
        
        for h_chunk in hierarchical_chunks:
            # Only apply semantic chunking to leaf nodes
            if not h_chunk['metadata'].get('children_ids'):
                if len(h_chunk['content']) > self.max_chunk_size:
                    # Apply semantic chunking
                    sub_chunks = self.semantic_chunker.chunk_document(
                        h_chunk['content'],
                        max_size=self.max_chunk_size
                    )
                    
                    for sub_chunk in sub_chunks:
                        sub_chunk['metadata'].update(h_chunk['metadata'])
                        sub_chunk['metadata']['hybrid_type'] = 'semantic_leaf'
                        hybrid_chunks.append(sub_chunk)
                else:
                    h_chunk['metadata']['hybrid_type'] = 'hierarchical'
                    hybrid_chunks.append(h_chunk)
            else:
                h_chunk['metadata']['hybrid_type'] = 'hierarchical_parent'
                hybrid_chunks.append(h_chunk)
        
        return hybrid_chunks
    
    def stream_chunks(
        self,
        content: str,
        strategy: Optional[str] = None,
        chunk_size: int = 1000,
        **kwargs
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Stream chunks for memory-efficient processing.
        
        Args:
            content: Content to chunk
            strategy: Chunking strategy
            chunk_size: Size for streaming chunks
            **kwargs: Strategy-specific parameters
            
        Yields:
            Individual chunks
        """
        # For now, simple streaming by breaking into segments
        segments = []
        current_segment = []
        current_size = 0
        
        lines = content.split('\n')
        
        for line in lines:
            if current_size + len(line) > chunk_size * 10:  # Process in batches
                segment_content = '\n'.join(current_segment)
                chunks = self.chunk(segment_content, strategy, **kwargs)
                
                for chunk in chunks:
                    yield chunk
                
                current_segment = [line]
                current_size = len(line)
            else:
                current_segment.append(line)
                current_size += len(line) + 1
        
        # Process remaining
        if current_segment:
            segment_content = '\n'.join(current_segment)
            chunks = self.chunk(segment_content, strategy, **kwargs)
            
            for chunk in chunks:
                yield chunk
    
    def evaluate_chunks(
        self,
        chunks: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Evaluate quality of chunks.
        
        Args:
            chunks: List of chunks to evaluate
            
        Returns:
            Quality metrics
        """
        if not chunks:
            return {
                'avg_coherence': 0.0,
                'size_consistency': 0.0,
                'coverage': 0.0
            }
        
        # Calculate coherence
        coherence_scores = []
        for chunk in chunks[:10]:  # Sample for efficiency
            coherence = self.semantic_chunker.calculate_coherence(chunk['content'])
            coherence_scores.append(coherence)
        
        avg_coherence = np.mean(coherence_scores) if coherence_scores else 0.0
        
        # Calculate size consistency
        sizes = [len(c['content']) for c in chunks]
        size_std = np.std(sizes)
        size_mean = np.mean(sizes)
        size_consistency = 1.0 - (size_std / (size_mean + 1))
        
        # Calculate coverage (no gaps)
        total_content_length = sum(sizes)
        coverage = min(1.0, total_content_length / (len(chunks) * self.max_chunk_size))
        
        return {
            'avg_coherence': float(avg_coherence),
            'size_consistency': float(size_consistency),
            'coverage': float(coverage),
            'total_chunks': len(chunks),
            'avg_size': float(size_mean),
            'size_std': float(size_std)
        }
    
    def optimize_parameters(
        self,
        sample_documents: List[str],
        target_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Optimize chunking parameters based on sample documents.
        
        Args:
            sample_documents: Sample documents for optimization
            target_metrics: Target quality metrics
            
        Returns:
            Optimized parameters
        """
        best_params = {
            'similarity_threshold': 0.75,
            'max_chunk_size': self.max_chunk_size,
            'min_chunk_size': self.min_chunk_size
        }
        best_score = float('-inf')
        
        # Grid search over parameters
        for threshold in [0.6, 0.7, 0.75, 0.8, 0.85]:
            for max_size in [1000, 1500, 2000]:
                # Test parameters
                test_chunks = []
                for doc in sample_documents[:3]:  # Use subset for speed
                    chunks = self.semantic_chunker.chunk_document(
                        doc,
                        similarity_threshold=threshold,
                        max_size=max_size
                    )
                    test_chunks.extend(chunks)
                
                # Evaluate
                metrics = self.evaluate_chunks(test_chunks)
                
                # Calculate score
                score = (
                    metrics['avg_coherence'] * target_metrics.get('coherence_weight', 0.5) +
                    metrics['size_consistency'] * target_metrics.get('consistency_weight', 0.3) +
                    metrics['coverage'] * target_metrics.get('coverage_weight', 0.2)
                )
                
                if score > best_score:
                    best_score = score
                    best_params = {
                        'similarity_threshold': threshold,
                        'max_chunk_size': max_size,
                        'min_chunk_size': self.min_chunk_size,
                        'score': score,
                        'metrics': metrics
                    }
        
        logger.info(f"Optimized parameters: {best_params}")
        return best_params