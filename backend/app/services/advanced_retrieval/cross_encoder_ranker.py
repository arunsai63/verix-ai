"""
Cross-Encoder Reranking implementation for improved relevance scoring.
Uses transformer-based cross-encoders to rerank search results.
"""

import logging
import numpy as np
import torch
from typing import List, Dict, Any, Tuple, Optional
from sentence_transformers import CrossEncoder
from dataclasses import dataclass
import time
from functools import lru_cache
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class RerankCandidate:
    """Candidate document for reranking."""
    doc_id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    cross_encoder_score: Optional[float] = None
    final_score: Optional[float] = None


class CrossEncoderRanker:
    """
    Cross-Encoder based reranking for improved relevance scoring.
    
    Cross-encoders jointly encode query-document pairs for better
    understanding of relevance compared to bi-encoders.
    """
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,
        batch_size: int = 32,
        max_length: int = 512,
        use_cache: bool = True
    ):
        """
        Initialize Cross-Encoder ranker.
        
        Args:
            model_name: Pretrained cross-encoder model name
            device: Device to run model on (cuda/cpu)
            batch_size: Batch size for inference
            max_length: Maximum sequence length
            use_cache: Whether to cache scores
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.max_length = max_length
        self.use_cache = use_cache
        
        # Initialize model
        self._model = None
        self._load_model()
        
        # Score cache
        self.score_cache = {} if use_cache else None
        
        logger.info(f"CrossEncoderRanker initialized with {model_name} on {self.device}")
    
    def _load_model(self):
        """Load the cross-encoder model."""
        try:
            self._model = CrossEncoder(
                self.model_name,
                max_length=self.max_length,
                device=self.device
            )
            logger.info(f"Loaded cross-encoder model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load cross-encoder model: {str(e)}")
            # Fallback to CPU if CUDA fails
            if self.device == 'cuda':
                self.device = 'cpu'
                self._model = CrossEncoder(
                    self.model_name,
                    max_length=self.max_length,
                    device='cpu'
                )
                logger.info("Fell back to CPU for cross-encoder")
    
    @property
    def model(self):
        """Get the cross-encoder model."""
        if self._model is None:
            self._load_model()
        return self._model
    
    def _get_cache_key(self, query: str, document: str) -> str:
        """Generate cache key for query-document pair."""
        combined = f"{query}||{document}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def score_pairs(
        self,
        query: str,
        documents: List[str],
        batch_size: Optional[int] = None,
        use_cache: Optional[bool] = None
    ) -> np.ndarray:
        """
        Score query-document pairs using cross-encoder.
        
        Args:
            query: Search query
            documents: List of document texts
            batch_size: Override default batch size
            use_cache: Override cache setting
            
        Returns:
            Array of relevance scores
        """
        batch_size = batch_size or self.batch_size
        use_cache = use_cache if use_cache is not None else self.use_cache
        
        # Check cache
        scores = []
        uncached_indices = []
        uncached_docs = []
        
        if use_cache and self.score_cache is not None:
            for i, doc in enumerate(documents):
                cache_key = self._get_cache_key(query, doc)
                if cache_key in self.score_cache:
                    scores.append(self.score_cache[cache_key])
                else:
                    scores.append(None)
                    uncached_indices.append(i)
                    uncached_docs.append(doc)
        else:
            uncached_docs = documents
            uncached_indices = list(range(len(documents)))
            scores = [None] * len(documents)
        
        # Score uncached pairs
        if uncached_docs:
            try:
                # Create query-document pairs
                pairs = [[query, doc] for doc in uncached_docs]
                
                # Score in batches
                batch_scores = []
                for i in range(0, len(pairs), batch_size):
                    batch = pairs[i:i + batch_size]
                    with torch.no_grad():
                        batch_score = self.model.predict(batch)
                    batch_scores.extend(batch_score)
                
                # Convert to numpy array
                batch_scores = np.array(batch_scores)
                
                # Apply sigmoid for probability scores
                batch_scores = 1 / (1 + np.exp(-batch_scores))
                
                # Update cache and results
                for idx, score in zip(uncached_indices, batch_scores):
                    scores[idx] = score
                    if use_cache and self.score_cache is not None:
                        cache_key = self._get_cache_key(query, documents[idx])
                        self.score_cache[cache_key] = score
                        
            except Exception as e:
                logger.error(f"Cross-encoder scoring failed: {str(e)}")
                # Fallback to uniform scores
                for idx in uncached_indices:
                    scores[idx] = 0.5
        
        return np.array(scores)
    
    def rerank(
        self,
        query: str,
        candidates: List[Any],
        top_k: Optional[int] = None,
        combine_scores: bool = True,
        alpha: float = 0.7
    ) -> List[Any]:
        """
        Rerank candidates using cross-encoder scores.
        
        Args:
            query: Search query
            candidates: List of candidates (must have 'content' attribute/key)
            top_k: Number of top results to return
            combine_scores: Whether to combine with original scores
            alpha: Weight for cross-encoder score (1-alpha for original)
            
        Returns:
            Reranked list of candidates
        """
        if not candidates:
            return []
        
        # Extract content from candidates
        if isinstance(candidates[0], dict):
            contents = [c.get('content', '') for c in candidates]
            original_scores = [c.get('score', 0.5) for c in candidates]
        else:
            contents = [getattr(c, 'content', str(c)) for c in candidates]
            original_scores = [getattr(c, 'score', 0.5) for c in candidates]
        
        # Get cross-encoder scores
        ce_scores = self.score_pairs(query, contents)
        
        # Combine scores if requested
        if combine_scores:
            final_scores = alpha * ce_scores + (1 - alpha) * np.array(original_scores)
        else:
            final_scores = ce_scores
        
        # Create reranked candidates
        reranked = []
        for i, candidate in enumerate(candidates):
            # Create new candidate with scores
            if isinstance(candidate, dict):
                new_candidate = candidate.copy()
                new_candidate['cross_encoder_score'] = float(ce_scores[i])
                new_candidate['final_score'] = float(final_scores[i])
            else:
                # Assuming it's a dataclass or object
                new_candidate = RerankCandidate(
                    doc_id=getattr(candidate, 'doc_id', str(i)),
                    content=contents[i],
                    score=original_scores[i],
                    metadata=getattr(candidate, 'metadata', {}),
                    cross_encoder_score=float(ce_scores[i]),
                    final_score=float(final_scores[i])
                )
            reranked.append(new_candidate)
        
        # Sort by final score
        reranked.sort(key=lambda x: x.get('final_score', x.final_score) if isinstance(x, dict) else x.final_score, reverse=True)
        
        # Return top-k if specified
        if top_k:
            reranked = reranked[:top_k]
        
        return reranked
    
    def calibrate_scores(self, scores: np.ndarray) -> np.ndarray:
        """
        Calibrate raw scores to probabilities.
        
        Args:
            scores: Raw model scores
            
        Returns:
            Calibrated probability scores
        """
        # Apply temperature scaling
        temperature = 2.0
        scores = scores / temperature
        
        # Apply sigmoid if not already probabilities
        if scores.min() < 0 or scores.max() > 1:
            scores = 1 / (1 + np.exp(-scores))
        
        # Normalize to ensure valid probabilities
        scores = np.clip(scores, 0, 1)
        
        return scores
    
    def batch_rerank(
        self,
        queries: List[str],
        candidates_list: List[List[Any]],
        top_k: Optional[int] = None
    ) -> List[List[Any]]:
        """
        Rerank multiple queries in batch.
        
        Args:
            queries: List of queries
            candidates_list: List of candidate lists
            top_k: Number of top results per query
            
        Returns:
            List of reranked results for each query
        """
        results = []
        
        for query, candidates in zip(queries, candidates_list):
            reranked = self.rerank(query, candidates, top_k)
            results.append(reranked)
        
        return results
    
    def clear_cache(self):
        """Clear the score cache."""
        if self.score_cache is not None:
            self.score_cache.clear()
            logger.info("Cleared cross-encoder score cache")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "cache_enabled": self.use_cache,
            "cache_size": len(self.score_cache) if self.score_cache else 0
        }


class MonoT5Ranker(CrossEncoderRanker):
    """
    MonoT5-based reranker for document-level reranking.
    Specialized for longer documents and more nuanced ranking.
    """
    
    def __init__(
        self,
        model_name: str = "castorini/monot5-base-msmarco",
        device: Optional[str] = None,
        batch_size: int = 16,
        max_length: int = 512
    ):
        """Initialize MonoT5 ranker."""
        super().__init__(
            model_name=model_name,
            device=device,
            batch_size=batch_size,
            max_length=max_length
        )
        
    def score_pairs(
        self,
        query: str,
        documents: List[str],
        batch_size: Optional[int] = None,
        use_cache: Optional[bool] = None
    ) -> np.ndarray:
        """
        Score with MonoT5 specific preprocessing.
        
        MonoT5 expects input in format: "Query: {query} Document: {document}"
        """
        # Format inputs for MonoT5
        formatted_docs = [
            f"Query: {query} Document: {doc}" 
            for doc in documents
        ]
        
        # Use parent class scoring with formatted inputs
        return super().score_pairs("", formatted_docs, batch_size, use_cache)