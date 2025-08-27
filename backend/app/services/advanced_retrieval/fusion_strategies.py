"""
Fusion strategies for combining results from multiple retrievers.
Implements Reciprocal Rank Fusion (RRF) and weighted fusion methods.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)


class RecipocalRankFusion:
    """
    Implements Reciprocal Rank Fusion (RRF) for combining rankings from multiple retrievers.
    
    RRF is a simple yet effective fusion method that combines rankings without
    requiring relevance scores, making it robust across different retrieval methods.
    
    Formula: RRF_score(d) = Σ 1 / (k + rank_i(d))
    where k is a constant (default: 60) and rank_i is the rank of document d in retriever i.
    """
    
    def __init__(self, k: int = 60):
        """
        Initialize RRF with the k parameter.
        
        Args:
            k: Constant that mitigates the impact of high rankings (default: 60)
        """
        self.k = k
        logger.info(f"RecipocalRankFusion initialized with k={k}")
    
    def fuse_rankings(
        self,
        rankings_list: List[List[Dict[str, Any]]],
        k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Fuse multiple rankings using Reciprocal Rank Fusion.
        
        Args:
            rankings_list: List of rankings from different retrievers.
                          Each ranking is a list of dicts with 'doc_id' and optionally 'rank'
            k: Override the default k parameter for this fusion
            
        Returns:
            Fused ranking sorted by RRF scores
        """
        k = k or self.k
        rrf_scores = defaultdict(float)
        doc_metadata = {}
        
        for rankings in rankings_list:
            for position, doc in enumerate(rankings):
                # Use provided rank or position-based rank (1-indexed)
                rank = doc.get('rank', position + 1)
                doc_id = doc.get('doc_id') or doc.get('id') or str(position)
                
                # Calculate RRF contribution
                rrf_scores[doc_id] += 1.0 / (k + rank)
                
                # Store document metadata (from first occurrence)
                if doc_id not in doc_metadata:
                    doc_metadata[doc_id] = {
                        k: v for k, v in doc.items() 
                        if k not in ['rank', 'score', 'rrf_score']
                    }
        
        # Create fused results
        fused_results = []
        for doc_id, rrf_score in rrf_scores.items():
            result = doc_metadata.get(doc_id, {})
            result.update({
                'doc_id': doc_id,
                'rrf_score': rrf_score,
                'fusion_method': 'rrf'
            })
            fused_results.append(result)
        
        # Sort by RRF score (descending)
        fused_results.sort(key=lambda x: x['rrf_score'], reverse=True)
        
        # Add final ranks
        for rank, doc in enumerate(fused_results, 1):
            doc['final_rank'] = rank
        
        logger.debug(f"RRF fusion completed: {len(rankings_list)} rankings -> "
                    f"{len(fused_results)} unique documents")
        
        return fused_results
    
    def weighted_fusion(
        self,
        rankings_list: List[List[Dict[str, Any]]],
        weights: List[float],
        k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform weighted RRF fusion giving different importance to each retriever.
        
        Args:
            rankings_list: List of rankings from different retrievers
            weights: Weight for each retriever (should sum to 1.0)
            k: Override the default k parameter
            
        Returns:
            Weighted fused ranking
        """
        if len(rankings_list) != len(weights):
            raise ValueError(f"Number of rankings ({len(rankings_list)}) must match "
                           f"number of weights ({len(weights)})")
        
        # Normalize weights if they don't sum to 1
        weight_sum = sum(weights)
        if abs(weight_sum - 1.0) > 0.01:
            weights = [w / weight_sum for w in weights]
            logger.warning(f"Weights normalized to sum to 1.0: {weights}")
        
        k = k or self.k
        weighted_scores = defaultdict(float)
        doc_metadata = {}
        
        for rankings, weight in zip(rankings_list, weights):
            for position, doc in enumerate(rankings):
                rank = doc.get('rank', position + 1)
                doc_id = doc.get('doc_id') or doc.get('id') or str(position)
                
                # Weighted RRF contribution
                weighted_scores[doc_id] += weight * (1.0 / (k + rank))
                
                if doc_id not in doc_metadata:
                    doc_metadata[doc_id] = {
                        k: v for k, v in doc.items() 
                        if k not in ['rank', 'score', 'rrf_score']
                    }
        
        # Create fused results
        fused_results = []
        for doc_id, weighted_score in weighted_scores.items():
            result = doc_metadata.get(doc_id, {})
            result.update({
                'doc_id': doc_id,
                'weighted_rrf_score': weighted_score,
                'fusion_method': 'weighted_rrf',
                'weights': weights
            })
            fused_results.append(result)
        
        # Sort by weighted RRF score
        fused_results.sort(key=lambda x: x['weighted_rrf_score'], reverse=True)
        
        # Add final ranks
        for rank, doc in enumerate(fused_results, 1):
            doc['final_rank'] = rank
        
        return fused_results


class ScoreFusion:
    """
    Fusion strategies based on relevance scores rather than ranks.
    Useful when all retrievers provide comparable scores.
    """
    
    @staticmethod
    def linear_combination(
        results_list: List[List[Dict[str, Any]]],
        weights: List[float]
    ) -> List[Dict[str, Any]]:
        """
        Combine results using weighted linear combination of scores.
        
        Args:
            results_list: List of results from different retrievers with 'score' field
            weights: Weight for each retriever
            
        Returns:
            Fused results sorted by combined score
        """
        if len(results_list) != len(weights):
            raise ValueError("Number of result lists must match number of weights")
        
        # Normalize weights
        weight_sum = sum(weights)
        weights = [w / weight_sum for w in weights]
        
        combined_scores = defaultdict(float)
        doc_metadata = {}
        
        for results, weight in zip(results_list, weights):
            for doc in results:
                doc_id = doc.get('doc_id') or doc.get('id')
                score = doc.get('score', 0.0)
                
                combined_scores[doc_id] += weight * score
                
                if doc_id not in doc_metadata:
                    doc_metadata[doc_id] = {
                        k: v for k, v in doc.items() 
                        if k not in ['score', 'combined_score']
                    }
        
        # Create fused results
        fused_results = []
        for doc_id, combined_score in combined_scores.items():
            result = doc_metadata.get(doc_id, {})
            result.update({
                'doc_id': doc_id,
                'combined_score': combined_score,
                'fusion_method': 'linear_combination',
                'weights': weights
            })
            fused_results.append(result)
        
        # Sort by combined score
        fused_results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return fused_results
    
    @staticmethod
    def max_score(
        results_list: List[List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """
        Take maximum score for each document across all retrievers.
        
        Args:
            results_list: List of results from different retrievers
            
        Returns:
            Fused results with maximum scores
        """
        max_scores = {}
        doc_metadata = {}
        
        for retriever_idx, results in enumerate(results_list):
            for doc in results:
                doc_id = doc.get('doc_id') or doc.get('id')
                score = doc.get('score', 0.0)
                
                if doc_id not in max_scores or score > max_scores[doc_id]['score']:
                    max_scores[doc_id] = {
                        'score': score,
                        'retriever_idx': retriever_idx
                    }
                
                if doc_id not in doc_metadata:
                    doc_metadata[doc_id] = {
                        k: v for k, v in doc.items() 
                        if k not in ['score']
                    }
        
        # Create fused results
        fused_results = []
        for doc_id, score_info in max_scores.items():
            result = doc_metadata.get(doc_id, {})
            result.update({
                'doc_id': doc_id,
                'max_score': score_info['score'],
                'best_retriever': score_info['retriever_idx'],
                'fusion_method': 'max_score'
            })
            fused_results.append(result)
        
        # Sort by max score
        fused_results.sort(key=lambda x: x['max_score'], reverse=True)
        
        return fused_results
    
    @staticmethod
    def normalize_and_combine(
        results_list: List[List[Dict[str, Any]]],
        weights: Optional[List[float]] = None,
        normalization: str = 'min_max'
    ) -> List[Dict[str, Any]]:
        """
        Normalize scores across retrievers before combining.
        
        Args:
            results_list: List of results from different retrievers
            weights: Optional weights for each retriever
            normalization: Type of normalization ('min_max' or 'z_score')
            
        Returns:
            Fused results with normalized and combined scores
        """
        if weights is None:
            weights = [1.0 / len(results_list)] * len(results_list)
        
        normalized_results = []
        
        for results in results_list:
            if not results:
                normalized_results.append([])
                continue
            
            scores = [doc.get('score', 0.0) for doc in results]
            
            if normalization == 'min_max':
                min_score = min(scores)
                max_score = max(scores)
                score_range = max_score - min_score
                
                if score_range > 0:
                    normalized = []
                    for doc in results:
                        norm_doc = doc.copy()
                        norm_doc['normalized_score'] = (
                            (doc.get('score', 0.0) - min_score) / score_range
                        )
                        normalized.append(norm_doc)
                    normalized_results.append(normalized)
                else:
                    # All scores are the same
                    normalized = []
                    for doc in results:
                        norm_doc = doc.copy()
                        norm_doc['normalized_score'] = 1.0
                        normalized.append(norm_doc)
                    normalized_results.append(normalized)
                    
            elif normalization == 'z_score':
                mean_score = np.mean(scores)
                std_score = np.std(scores)
                
                if std_score > 0:
                    normalized = []
                    for doc in results:
                        norm_doc = doc.copy()
                        z_score = (doc.get('score', 0.0) - mean_score) / std_score
                        # Convert z-score to 0-1 range using sigmoid
                        norm_doc['normalized_score'] = 1 / (1 + np.exp(-z_score))
                        normalized.append(norm_doc)
                    normalized_results.append(normalized)
                else:
                    # All scores are the same
                    normalized = []
                    for doc in results:
                        norm_doc = doc.copy()
                        norm_doc['normalized_score'] = 0.5
                        normalized.append(norm_doc)
                    normalized_results.append(normalized)
            else:
                raise ValueError(f"Unknown normalization method: {normalization}")
        
        # Now combine normalized scores
        combined_scores = defaultdict(float)
        doc_metadata = {}
        
        for norm_results, weight in zip(normalized_results, weights):
            for doc in norm_results:
                doc_id = doc.get('doc_id') or doc.get('id')
                norm_score = doc.get('normalized_score', 0.0)
                
                combined_scores[doc_id] += weight * norm_score
                
                if doc_id not in doc_metadata:
                    doc_metadata[doc_id] = {
                        k: v for k, v in doc.items() 
                        if k not in ['score', 'normalized_score', 'combined_score']
                    }
        
        # Create fused results
        fused_results = []
        for doc_id, combined_score in combined_scores.items():
            result = doc_metadata.get(doc_id, {})
            result.update({
                'doc_id': doc_id,
                'combined_normalized_score': combined_score,
                'fusion_method': f'normalized_{normalization}',
                'weights': weights
            })
            fused_results.append(result)
        
        # Sort by combined normalized score
        fused_results.sort(key=lambda x: x['combined_normalized_score'], reverse=True)
        
        return fused_results