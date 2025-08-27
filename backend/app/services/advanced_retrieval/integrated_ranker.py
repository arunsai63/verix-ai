"""
Integrated Ranking System combining cross-encoder reranking, cascade pipeline, and diversity optimization.
"""

import logging
from typing import List, Dict, Any, Optional, Literal
import numpy as np

from .cross_encoder_ranker import CrossEncoderRanker
from .cascade_ranker import CascadeRanker
from .diversity_ranker import DiversityRanker

logger = logging.getLogger(__name__)


class IntegratedRanker:
    """
    Integrated ranking system combining multiple reranking strategies.
    """
    
    def __init__(
        self,
        enable_cross_encoder: bool = True,
        enable_cascade: bool = True,
        enable_diversity: bool = True
    ):
        """
        Initialize integrated ranker.
        
        Args:
            enable_cross_encoder: Enable cross-encoder reranking
            enable_cascade: Enable cascade pipeline
            enable_diversity: Enable diversity optimization
        """
        self.cross_encoder = CrossEncoderRanker() if enable_cross_encoder else None
        self.cascade = CascadeRanker() if enable_cascade else None
        self.diversity = DiversityRanker() if enable_diversity else None
        
        self.enable_cross_encoder = enable_cross_encoder
        self.enable_cascade = enable_cascade
        self.enable_diversity = enable_diversity
        
        logger.info(
            f"IntegratedRanker initialized: "
            f"cross_encoder={enable_cross_encoder}, "
            f"cascade={enable_cascade}, "
            f"diversity={enable_diversity}"
        )
    
    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int = 10,
        strategy: Literal["cross_encoder", "cascade", "diversity", "combined"] = "combined",
        diversity_lambda: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Rerank results using specified strategy.
        
        Args:
            query: Search query
            results: Initial retrieval results
            top_k: Number of results to return
            strategy: Reranking strategy to use
            diversity_lambda: Weight for diversity in MMR
            
        Returns:
            Reranked results
        """
        if not results:
            return []
        
        if strategy == "cross_encoder" and self.cross_encoder:
            return self._rerank_cross_encoder(query, results, top_k)
        
        elif strategy == "cascade" and self.cascade:
            return self._rerank_cascade(query, results, top_k)
        
        elif strategy == "diversity" and self.diversity:
            return self._rerank_diversity(query, results, top_k, diversity_lambda)
        
        elif strategy == "combined":
            return self._rerank_combined(query, results, top_k, diversity_lambda)
        
        else:
            # Fallback to original order
            logger.warning(f"Strategy {strategy} not available, returning original order")
            return results[:top_k]
    
    def _rerank_cross_encoder(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Rerank using cross-encoder."""
        try:
            reranked = self.cross_encoder.rerank(
                query=query,
                results=results,
                top_k=min(top_k * 2, len(results))  # Get more for diversity
            )
            return reranked[:top_k]
        except Exception as e:
            logger.error(f"Cross-encoder reranking failed: {str(e)}")
            return results[:top_k]
    
    def _rerank_cascade(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Rerank using cascade pipeline."""
        try:
            # Configure cascade stages
            stages = [
                {"name": "initial", "top_k": min(top_k * 3, len(results))},
                {"name": "refinement", "top_k": min(top_k * 2, len(results))},
                {"name": "final", "top_k": top_k}
            ]
            
            reranked = self.cascade.cascade_rerank(
                query=query,
                results=results,
                stages=stages
            )
            return reranked
        except Exception as e:
            logger.error(f"Cascade reranking failed: {str(e)}")
            return results[:top_k]
    
    def _rerank_diversity(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int,
        diversity_lambda: float
    ) -> List[Dict[str, Any]]:
        """Rerank for diversity using MMR."""
        try:
            # Get document contents
            documents = [r.get("content", "") for r in results]
            
            # Apply MMR
            selected_indices = self.diversity.mmr_select(
                query=query,
                documents=documents,
                scores=[r.get("score", 0) for r in results],
                top_k=top_k,
                lambda_param=diversity_lambda
            )
            
            # Return selected results
            return [results[i] for i in selected_indices]
        except Exception as e:
            logger.error(f"Diversity reranking failed: {str(e)}")
            return results[:top_k]
    
    def _rerank_combined(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int,
        diversity_lambda: float
    ) -> List[Dict[str, Any]]:
        """Combined reranking strategy."""
        current_results = results
        
        # Step 1: Cross-encoder reranking if available
        if self.enable_cross_encoder and self.cross_encoder:
            try:
                current_results = self.cross_encoder.rerank(
                    query=query,
                    results=current_results,
                    top_k=min(top_k * 3, len(current_results))
                )
                logger.info(f"Cross-encoder reranked to {len(current_results)} results")
            except Exception as e:
                logger.warning(f"Cross-encoder step failed: {str(e)}")
        
        # Step 2: Cascade refinement if available
        if self.enable_cascade and self.cascade and len(current_results) > top_k * 2:
            try:
                stages = [
                    {"name": "refinement", "top_k": min(top_k * 2, len(current_results))},
                    {"name": "final", "top_k": top_k}
                ]
                current_results = self.cascade.cascade_rerank(
                    query=query,
                    results=current_results,
                    stages=stages
                )
                logger.info(f"Cascade refined to {len(current_results)} results")
            except Exception as e:
                logger.warning(f"Cascade step failed: {str(e)}")
        
        # Step 3: Diversity optimization if available
        if self.enable_diversity and self.diversity:
            try:
                documents = [r.get("content", "") for r in current_results]
                scores = [r.get("score", 0) for r in current_results]
                
                selected_indices = self.diversity.mmr_select(
                    query=query,
                    documents=documents,
                    scores=scores,
                    top_k=top_k,
                    lambda_param=diversity_lambda
                )
                
                current_results = [current_results[i] for i in selected_indices]
                logger.info(f"Diversity optimized to {len(current_results)} results")
            except Exception as e:
                logger.warning(f"Diversity step failed: {str(e)}")
        
        # Ensure we return top_k results
        return current_results[:top_k]
    
    def batch_rerank(
        self,
        queries: List[str],
        results_list: List[List[Dict[str, Any]]],
        top_k: int = 10,
        strategy: str = "combined"
    ) -> List[List[Dict[str, Any]]]:
        """
        Rerank multiple query-results pairs.
        
        Args:
            queries: List of queries
            results_list: List of result lists
            top_k: Number of results per query
            strategy: Reranking strategy
            
        Returns:
            List of reranked result lists
        """
        reranked_list = []
        
        for query, results in zip(queries, results_list):
            reranked = self.rerank(
                query=query,
                results=results,
                top_k=top_k,
                strategy=strategy
            )
            reranked_list.append(reranked)
        
        return reranked_list
    
    def evaluate_reranking(
        self,
        original_results: List[Dict[str, Any]],
        reranked_results: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Evaluate reranking quality.
        
        Args:
            original_results: Original results
            reranked_results: Reranked results
            
        Returns:
            Evaluation metrics
        """
        if not reranked_results:
            return {
                "score_improvement": 0.0,
                "diversity_score": 0.0,
                "position_changes": 0.0
            }
        
        # Calculate score improvement
        original_scores = [r.get("score", 0) for r in original_results[:len(reranked_results)]]
        reranked_scores = [r.get("score", 0) for r in reranked_results]
        
        score_improvement = (
            np.mean(reranked_scores) - np.mean(original_scores)
            if original_scores else 0.0
        )
        
        # Calculate diversity (unique sources)
        sources = set()
        for r in reranked_results:
            if "metadata" in r and "source" in r["metadata"]:
                sources.add(r["metadata"]["source"])
        
        diversity_score = len(sources) / len(reranked_results)
        
        # Calculate average position change
        position_changes = []
        for i, result in enumerate(reranked_results):
            result_id = result.get("id") or result.get("content", "")[:50]
            
            # Find original position
            original_pos = None
            for j, orig in enumerate(original_results):
                orig_id = orig.get("id") or orig.get("content", "")[:50]
                if orig_id == result_id:
                    original_pos = j
                    break
            
            if original_pos is not None:
                position_changes.append(abs(i - original_pos))
        
        avg_position_change = np.mean(position_changes) if position_changes else 0.0
        
        return {
            "score_improvement": float(score_improvement),
            "diversity_score": float(diversity_score),
            "position_changes": float(avg_position_change),
            "total_reranked": len(reranked_results)
        }