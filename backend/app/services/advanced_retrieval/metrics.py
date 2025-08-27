"""
Retrieval evaluation metrics for measuring search quality.
Implements MRR, NDCG, Recall@K, Precision@K, and other metrics.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class RetrievalMetrics:
    """
    Calculates various retrieval quality metrics.
    
    Metrics:
    - MRR (Mean Reciprocal Rank)
    - NDCG (Normalized Discounted Cumulative Gain)
    - Recall@K
    - Precision@K
    - F1@K
    - MAP (Mean Average Precision)
    """
    
    def calculate_mrr(self, rankings: List[Dict[str, Any]]) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR).
        
        Args:
            rankings: List of dicts with 'query' and 'relevant_pos' (1-indexed position)
            
        Returns:
            MRR score
        """
        if not rankings:
            return 0.0
        
        reciprocal_ranks = []
        for ranking in rankings:
            relevant_pos = ranking.get('relevant_pos', float('inf'))
            if relevant_pos != float('inf'):
                reciprocal_ranks.append(1.0 / relevant_pos)
            else:
                reciprocal_ranks.append(0.0)
        
        return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
    
    def calculate_ndcg(
        self,
        retrieved: List[float],
        ideal: List[float],
        k: Optional[int] = None
    ) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain (NDCG).
        
        Args:
            retrieved: Relevance scores of retrieved documents
            ideal: Ideal relevance scores (sorted)
            k: Cutoff position (None for all)
            
        Returns:
            NDCG@K score
        """
        if not retrieved:
            return 0.0
        
        k = k or len(retrieved)
        retrieved = retrieved[:k]
        ideal = ideal[:k]
        
        def dcg(scores):
            """Calculate DCG."""
            return sum(
                (2**score - 1) / np.log2(i + 2)
                for i, score in enumerate(scores)
            )
        
        dcg_retrieved = dcg(retrieved)
        dcg_ideal = dcg(ideal)
        
        if dcg_ideal == 0:
            return 0.0
        
        return dcg_retrieved / dcg_ideal
    
    def calculate_recall_at_k(
        self,
        retrieved: List[str],
        relevant: List[str],
        k: int
    ) -> float:
        """
        Calculate Recall@K.
        
        Args:
            retrieved: List of retrieved document IDs
            relevant: List of relevant document IDs
            k: Cutoff position
            
        Returns:
            Recall@K score
        """
        if not relevant:
            return 0.0
        
        retrieved_set = set(retrieved[:k])
        relevant_set = set(relevant)
        
        intersection = retrieved_set.intersection(relevant_set)
        
        return len(intersection) / len(relevant_set)
    
    def calculate_precision_at_k(
        self,
        retrieved: List[str],
        relevant: List[str],
        k: int
    ) -> float:
        """
        Calculate Precision@K.
        
        Args:
            retrieved: List of retrieved document IDs
            relevant: List of relevant document IDs
            k: Cutoff position
            
        Returns:
            Precision@K score
        """
        if k == 0:
            return 0.0
        
        retrieved_set = set(retrieved[:k])
        relevant_set = set(relevant)
        
        intersection = retrieved_set.intersection(relevant_set)
        
        return len(intersection) / k
    
    def calculate_f1_at_k(
        self,
        retrieved: List[str],
        relevant: List[str],
        k: int
    ) -> float:
        """
        Calculate F1@K score.
        
        Args:
            retrieved: List of retrieved document IDs
            relevant: List of relevant document IDs
            k: Cutoff position
            
        Returns:
            F1@K score
        """
        precision = self.calculate_precision_at_k(retrieved, relevant, k)
        recall = self.calculate_recall_at_k(retrieved, relevant, k)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def calculate_map(
        self,
        queries_results: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate Mean Average Precision (MAP).
        
        Args:
            queries_results: List of query results with 'retrieved' and 'relevant' lists
            
        Returns:
            MAP score
        """
        if not queries_results:
            return 0.0
        
        average_precisions = []
        
        for query_result in queries_results:
            retrieved = query_result.get('retrieved', [])
            relevant = set(query_result.get('relevant', []))
            
            if not relevant:
                continue
            
            precisions = []
            num_relevant = 0
            
            for i, doc_id in enumerate(retrieved, 1):
                if doc_id in relevant:
                    num_relevant += 1
                    precisions.append(num_relevant / i)
            
            if precisions:
                ap = np.mean(precisions)
            else:
                ap = 0.0
            
            average_precisions.append(ap)
        
        return np.mean(average_precisions) if average_precisions else 0.0
    
    def evaluate_retrieval(
        self,
        results: List[Dict[str, Any]],
        ground_truth: List[Dict[str, Any]],
        k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation of retrieval results.
        
        Args:
            results: List of retrieval results
            ground_truth: Ground truth relevance data
            k_values: List of K values for metrics
            
        Returns:
            Dictionary of all metrics
        """
        metrics = {}
        
        # Calculate metrics for each K
        for k in k_values:
            retrieved_ids = [r.get('doc_id', str(i)) for i, r in enumerate(results[:k])]
            relevant_ids = [gt.get('doc_id') for gt in ground_truth if gt.get('is_relevant')]
            
            metrics[f'recall@{k}'] = self.calculate_recall_at_k(
                retrieved_ids, relevant_ids, k
            )
            metrics[f'precision@{k}'] = self.calculate_precision_at_k(
                retrieved_ids, relevant_ids, k
            )
            metrics[f'f1@{k}'] = self.calculate_f1_at_k(
                retrieved_ids, relevant_ids, k
            )
        
        # Calculate NDCG
        relevance_scores = [r.get('relevance_score', 0) for r in results[:10]]
        ideal_scores = sorted(relevance_scores, reverse=True)
        metrics['ndcg@10'] = self.calculate_ndcg(relevance_scores, ideal_scores, 10)
        
        # Calculate MRR
        first_relevant_pos = None
        for i, result in enumerate(results, 1):
            if result.get('doc_id') in relevant_ids:
                first_relevant_pos = i
                break
        
        if first_relevant_pos:
            metrics['mrr'] = 1.0 / first_relevant_pos
        else:
            metrics['mrr'] = 0.0
        
        return metrics
    
    def compare_retrievers(
        self,
        retriever_results: Dict[str, List[Dict[str, Any]]],
        ground_truth: List[Dict[str, Any]],
        k: int = 10
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple retrievers on the same queries.
        
        Args:
            retriever_results: Dict mapping retriever name to results
            ground_truth: Ground truth data
            k: Cutoff for metrics
            
        Returns:
            Metrics for each retriever
        """
        comparison = {}
        
        for retriever_name, results in retriever_results.items():
            metrics = self.evaluate_retrieval(
                results,
                ground_truth,
                k_values=[k]
            )
            comparison[retriever_name] = metrics
        
        # Calculate relative improvements
        if len(comparison) > 1:
            baseline = list(comparison.values())[0]
            for name, metrics in comparison.items():
                if name != list(comparison.keys())[0]:
                    comparison[name]['relative_improvement'] = {
                        metric: (value - baseline.get(metric, 0)) / (baseline.get(metric, 1) or 1)
                        for metric, value in metrics.items()
                        if metric in baseline
                    }
        
        return comparison