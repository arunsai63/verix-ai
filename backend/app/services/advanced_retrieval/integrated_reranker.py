"""
Integrated Reranking System combining all reranking components.
Provides a unified interface for advanced reranking capabilities.
"""

import logging
from typing import List, Dict, Any, Optional
import time

from .cross_encoder_ranker import CrossEncoderRanker, MonoT5Ranker
from .cascade_ranker import CascadeRanker, CascadeConfig
from .diversity_ranker import DiversityRanker, DiversityConfig

logger = logging.getLogger(__name__)


class IntegratedReranker:
    """
    Unified reranking system integrating cross-encoder, cascade, and diversity components.
    """
    
    def __init__(
        self,
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        use_monot5: bool = False,
        enable_cascade: bool = True,
        enable_diversity: bool = True
    ):
        """
        Initialize integrated reranker.
        
        Args:
            cross_encoder_model: Model for cross-encoder reranking
            use_monot5: Whether to use MonoT5 instead of MiniLM
            enable_cascade: Enable cascade reranking
            enable_diversity: Enable diversity optimization
        """
        # Initialize components
        if use_monot5:
            self.cross_encoder = MonoT5Ranker()
        else:
            self.cross_encoder = CrossEncoderRanker(model_name=cross_encoder_model)
        
        self.diversity_ranker = DiversityRanker() if enable_diversity else None
        
        # Setup cascade if enabled
        if enable_cascade:
            cascade_config = CascadeConfig(
                stages=self._get_cascade_stages(enable_diversity),
                early_stopping=True,
                progressive_filtering=True
            )
            self.cascade_ranker = CascadeRanker(
                config=cascade_config,
                cross_encoder=self.cross_encoder,
                diversity_ranker=self.diversity_ranker
            )
        else:
            self.cascade_ranker = None
        
        self.enable_cascade = enable_cascade
        self.enable_diversity = enable_diversity
        
        logger.info(f"IntegratedReranker initialized - cascade: {enable_cascade}, diversity: {enable_diversity}")
    
    def _get_cascade_stages(self, include_diversity: bool) -> List[str]:
        """Get cascade stages based on configuration."""
        stages = ["initial", "cross_encoder"]
        if include_diversity:
            stages.append("diversity")
        return stages
    
    def rerank(
        self,
        query: str,
        candidates: List[Any],
        use_cross_encoder: bool = True,
        use_diversity: bool = True,
        top_k: int = 10,
        batch_size: Optional[int] = None
    ) -> List[Any]:
        """
        Perform integrated reranking.
        
        Args:
            query: Search query
            candidates: Initial candidates
            use_cross_encoder: Whether to use cross-encoder
            use_diversity: Whether to apply diversity
            top_k: Number of results
            batch_size: Batch size for processing
            
        Returns:
            Reranked candidates
        """
        start_time = time.time()
        
        if not candidates:
            return []
        
        # Use cascade if enabled
        if self.enable_cascade and self.cascade_ranker:
            stages = []
            if use_cross_encoder:
                stages.append("cross_encoder")
            if use_diversity and self.enable_diversity:
                stages.append("diversity")
            
            if stages:
                stages = ["initial"] + stages
                results = self.cascade_ranker.rerank(
                    query,
                    candidates,
                    stages=stages,
                    top_k=top_k
                )
            else:
                results = candidates[:top_k]
        else:
            # Manual pipeline
            results = candidates
            
            if use_cross_encoder and self.cross_encoder:
                results = self.cross_encoder.rerank(
                    query,
                    results,
                    top_k=min(top_k * 2, len(results))  # Keep more for diversity
                )
            
            if use_diversity and self.diversity_ranker:
                results = self.diversity_ranker.rerank_mmr(
                    query,
                    results,
                    top_k=top_k
                )
            else:
                results = results[:top_k]
        
        elapsed = time.time() - start_time
        logger.info(f"Integrated reranking completed in {elapsed:.3f}s")
        
        return results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = {
            "cross_encoder": self.cross_encoder.get_model_info() if self.cross_encoder else None,
            "cascade": self.cascade_ranker.get_execution_summary() if self.cascade_ranker else None,
            "diversity_cache": len(self.diversity_ranker.embedding_cache) if self.diversity_ranker and self.diversity_ranker.embedding_cache else 0
        }
        return stats