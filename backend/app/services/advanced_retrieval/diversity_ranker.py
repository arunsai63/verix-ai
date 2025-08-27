"""
Diversity Optimization for search results using MMR and clustering.
Ensures diverse and comprehensive search results.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class DiversityConfig:
    """Configuration for diversity ranking."""
    lambda_param: float = 0.5  # Balance between relevance and diversity
    similarity_threshold: float = 0.8  # Threshold for considering documents similar
    method: str = "mmr"  # mmr, clustering, coverage
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    use_cache: bool = True


class DiversityRanker:
    """
    Diversity optimization for search results.
    
    Implements various methods to ensure diverse results:
    - MMR (Maximal Marginal Relevance)
    - Clustering-based diversity
    - Topic coverage optimization
    """
    
    def __init__(
        self,
        config: Optional[DiversityConfig] = None,
        embedding_model: Optional[SentenceTransformer] = None
    ):
        """
        Initialize diversity ranker.
        
        Args:
            config: Diversity configuration
            embedding_model: Pre-loaded embedding model
        """
        self.config = config or DiversityConfig()
        self.lambda_param = self.config.lambda_param
        self.similarity_threshold = self.config.similarity_threshold
        self.method = self.config.method
        
        # Initialize embedding model
        self._embedding_model = embedding_model
        if self._embedding_model is None and self.config.embedding_model:
            self._load_embedding_model()
        
        # Cache for embeddings
        self.embedding_cache = {} if self.config.use_cache else None
        
        logger.info(f"DiversityRanker initialized with method: {self.method}")
    
    def _load_embedding_model(self):
        """Load the embedding model."""
        try:
            self._embedding_model = SentenceTransformer(self.config.embedding_model)
            logger.info(f"Loaded embedding model: {self.config.embedding_model}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            self._embedding_model = None
    
    @property
    def embedding_model(self):
        """Get the embedding model."""
        if self._embedding_model is None:
            self._load_embedding_model()
        return self._embedding_model
    
    def _get_content(self, candidate: Any) -> str:
        """Extract content from candidate."""
        if isinstance(candidate, dict):
            return candidate.get('content', '')
        elif hasattr(candidate, 'content'):
            return candidate.content
        else:
            return str(candidate)
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text with caching."""
        if self.embedding_cache is not None:
            cache_key = hashlib.md5(text.encode()).hexdigest()
            if cache_key in self.embedding_cache:
                return self.embedding_cache[cache_key]
        
        embedding = self.embedding_model.encode([text])[0]
        
        if self.embedding_cache is not None:
            self.embedding_cache[cache_key] = embedding
        
        return embedding
    
    def compute_similarity(self, doc1: Any, doc2: Any) -> float:
        """
        Compute similarity between two documents.
        
        Args:
            doc1: First document
            doc2: Second document
            
        Returns:
            Similarity score (0-1)
        """
        content1 = self._get_content(doc1)
        content2 = self._get_content(doc2)
        
        if not content1 or not content2:
            return 0.0
        
        emb1 = self._get_embedding(content1)
        emb2 = self._get_embedding(content2)
        
        similarity = cosine_similarity([emb1], [emb2])[0][0]
        return float(similarity)
    
    def rerank_mmr(
        self,
        query: str,
        candidates: List[Any],
        lambda_param: Optional[float] = None,
        top_k: Optional[int] = None
    ) -> List[Any]:
        """
        Rerank using Maximal Marginal Relevance (MMR).
        
        MMR iteratively selects documents that are relevant to the query
        while being dissimilar to already selected documents.
        
        Args:
            query: Search query
            candidates: List of candidates
            lambda_param: Balance parameter (default: config value)
            top_k: Number of results to return
            
        Returns:
            Diverse reranked candidates
        """
        if not candidates:
            return []
        
        lambda_param = lambda_param if lambda_param is not None else self.lambda_param
        top_k = top_k or len(candidates)
        
        # Get query embedding
        query_embedding = self._get_embedding(query)
        
        # Get candidate embeddings
        candidate_embeddings = []
        for candidate in candidates:
            content = self._get_content(candidate)
            embedding = self._get_embedding(content)
            candidate_embeddings.append(embedding)
        
        candidate_embeddings = np.array(candidate_embeddings)
        
        # Calculate relevance scores (similarity to query)
        relevance_scores = cosine_similarity([query_embedding], candidate_embeddings)[0]
        
        # MMR selection
        selected_indices = []
        remaining_indices = list(range(len(candidates)))
        
        # Select first document (highest relevance)
        first_idx = np.argmax(relevance_scores)
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)
        
        # Iteratively select diverse documents
        while len(selected_indices) < min(top_k, len(candidates)):
            if not remaining_indices:
                break
            
            # Calculate MMR scores for remaining documents
            mmr_scores = []
            
            for idx in remaining_indices:
                # Relevance to query
                relevance = relevance_scores[idx]
                
                # Maximum similarity to selected documents
                max_similarity = 0
                for selected_idx in selected_indices:
                    similarity = cosine_similarity(
                        [candidate_embeddings[idx]],
                        [candidate_embeddings[selected_idx]]
                    )[0][0]
                    max_similarity = max(max_similarity, similarity)
                
                # MMR score
                mmr = lambda_param * relevance - (1 - lambda_param) * max_similarity
                mmr_scores.append(mmr)
            
            # Select document with highest MMR score
            best_idx = remaining_indices[np.argmax(mmr_scores)]
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
        
        # Return reranked candidates
        reranked = [candidates[i] for i in selected_indices]
        
        # Add diversity scores
        for i, candidate in enumerate(reranked):
            if isinstance(candidate, dict):
                candidate['diversity_rank'] = i + 1
                candidate['mmr_score'] = relevance_scores[selected_indices[i]]
            else:
                candidate.diversity_rank = i + 1
                candidate.mmr_score = relevance_scores[selected_indices[i]]
        
        return reranked
    
    def select_next_mmr(
        self,
        query: str,
        selected: List[Any],
        remaining: List[Any],
        lambda_param: Optional[float] = None
    ) -> Optional[Any]:
        """
        Select next document using MMR (incremental selection).
        
        Args:
            query: Search query
            selected: Already selected documents
            remaining: Remaining candidates
            lambda_param: Balance parameter
            
        Returns:
            Next best document or None
        """
        if not remaining:
            return None
        
        lambda_param = lambda_param if lambda_param is not None else self.lambda_param
        
        # Get query embedding
        query_embedding = self._get_embedding(query)
        
        best_candidate = None
        best_mmr = float('-inf')
        
        for candidate in remaining:
            content = self._get_content(candidate)
            embedding = self._get_embedding(content)
            
            # Relevance to query
            relevance = cosine_similarity([query_embedding], [embedding])[0][0]
            
            # Maximum similarity to selected
            max_similarity = 0
            for selected_doc in selected:
                selected_content = self._get_content(selected_doc)
                selected_embedding = self._get_embedding(selected_content)
                similarity = cosine_similarity([embedding], [selected_embedding])[0][0]
                max_similarity = max(max_similarity, similarity)
            
            # MMR score
            mmr = lambda_param * relevance - (1 - lambda_param) * max_similarity
            
            if mmr > best_mmr:
                best_mmr = mmr
                best_candidate = candidate
        
        return best_candidate
    
    def rerank_clustering(
        self,
        candidates: List[Any],
        n_clusters: int = 5,
        top_k: Optional[int] = None
    ) -> List[Any]:
        """
        Rerank using clustering-based diversity.
        
        Groups similar documents and selects representatives from each cluster.
        
        Args:
            candidates: List of candidates
            n_clusters: Number of clusters
            top_k: Number of results to return
            
        Returns:
            Diverse candidates from different clusters
        """
        if not candidates:
            return []
        
        top_k = top_k or len(candidates)
        n_clusters = min(n_clusters, len(candidates))
        
        # Get embeddings
        embeddings = []
        for candidate in candidates:
            content = self._get_content(candidate)
            embedding = self._get_embedding(content)
            embeddings.append(embedding)
        
        embeddings = np.array(embeddings)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Select representatives from each cluster
        selected = []
        clusters_covered = set()
        
        # First, select best from each cluster
        for cluster_id in range(n_clusters):
            if len(selected) >= top_k:
                break
            
            cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
            if not cluster_indices:
                continue
            
            # Select document closest to cluster center
            cluster_embeddings = embeddings[cluster_indices]
            center = kmeans.cluster_centers_[cluster_id]
            distances = np.linalg.norm(cluster_embeddings - center, axis=1)
            best_idx = cluster_indices[np.argmin(distances)]
            
            selected.append(candidates[best_idx])
            clusters_covered.add(cluster_id)
        
        # If need more documents, add from uncovered clusters or best remaining
        if len(selected) < top_k:
            remaining_indices = [i for i in range(len(candidates)) if candidates[i] not in selected]
            
            # Sort by original scores if available
            if remaining_indices and hasattr(candidates[0], 'score'):
                remaining_indices.sort(
                    key=lambda i: getattr(candidates[i], 'score', 0),
                    reverse=True
                )
            
            for idx in remaining_indices:
                if len(selected) >= top_k:
                    break
                selected.append(candidates[idx])
        
        return selected
    
    def cluster_documents(
        self,
        documents: List[Any],
        n_clusters: int = 5
    ) -> List[int]:
        """
        Cluster documents and return cluster assignments.
        
        Args:
            documents: List of documents
            n_clusters: Number of clusters
            
        Returns:
            List of cluster IDs for each document
        """
        if not documents:
            return []
        
        # Get embeddings
        embeddings = []
        for doc in documents:
            content = self._get_content(doc)
            embedding = self._get_embedding(content)
            embeddings.append(embedding)
        
        embeddings = np.array(embeddings)
        
        # Cluster
        kmeans = KMeans(n_clusters=min(n_clusters, len(documents)), random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        
        return labels.tolist()
    
    def rerank_coverage(
        self,
        candidates: List[Any],
        coverage_field: str = "topic",
        top_k: Optional[int] = None
    ) -> List[Any]:
        """
        Rerank to maximize coverage of different topics/categories.
        
        Args:
            candidates: List of candidates
            coverage_field: Metadata field to optimize coverage for
            top_k: Number of results
            
        Returns:
            Candidates with maximum topic coverage
        """
        if not candidates:
            return []
        
        top_k = top_k or len(candidates)
        
        # Extract topics/categories
        topic_candidates = {}
        for candidate in candidates:
            if isinstance(candidate, dict):
                topic = candidate.get('metadata', {}).get(coverage_field)
            else:
                topic = getattr(candidate, 'metadata', {}).get(coverage_field)
            
            if topic:
                if topic not in topic_candidates:
                    topic_candidates[topic] = []
                topic_candidates[topic].append(candidate)
        
        # Select diverse candidates
        selected = []
        topics_covered = set()
        
        # First pass: one from each topic
        for topic, topic_cands in topic_candidates.items():
            if len(selected) >= top_k:
                break
            
            # Select best from this topic
            if hasattr(topic_cands[0], 'score'):
                best = max(topic_cands, key=lambda x: getattr(x, 'score', 0))
            else:
                best = topic_cands[0]
            
            selected.append(best)
            topics_covered.add(topic)
        
        # Second pass: fill remaining slots
        if len(selected) < top_k:
            remaining = [c for c in candidates if c not in selected]
            remaining_sorted = sorted(
                remaining,
                key=lambda x: getattr(x, 'score', 0) if hasattr(x, 'score') else 0,
                reverse=True
            )
            
            for candidate in remaining_sorted:
                if len(selected) >= top_k:
                    break
                selected.append(candidate)
        
        return selected
    
    def calculate_diversity_metrics(
        self,
        documents: List[Any]
    ) -> Dict[str, float]:
        """
        Calculate diversity metrics for a set of documents.
        
        Args:
            documents: List of documents
            
        Returns:
            Dictionary of diversity metrics
        """
        if len(documents) <= 1:
            return {
                "avg_similarity": 0.0,
                "min_similarity": 0.0,
                "max_similarity": 0.0,
                "coverage_score": 1.0
            }
        
        # Calculate pairwise similarities
        similarities = []
        for i in range(len(documents)):
            for j in range(i + 1, len(documents)):
                sim = self.compute_similarity(documents[i], documents[j])
                similarities.append(sim)
        
        # Calculate metrics
        metrics = {
            "avg_similarity": np.mean(similarities) if similarities else 0.0,
            "min_similarity": np.min(similarities) if similarities else 0.0,
            "max_similarity": np.max(similarities) if similarities else 0.0,
            "coverage_score": 1.0 - np.mean(similarities) if similarities else 1.0
        }
        
        return metrics
    
    def clear_cache(self):
        """Clear the embedding cache."""
        if self.embedding_cache is not None:
            self.embedding_cache.clear()
            logger.info("Cleared diversity ranker embedding cache")