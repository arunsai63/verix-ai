"""
Hybrid Retriever that combines BM25, dense embeddings, query expansion, and HyDE.
Provides a unified interface for advanced document retrieval.
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import time
import numpy as np

from .bm25_retriever import BM25Retriever
from .fusion_strategies import RecipocalRankFusion, ScoreFusion
from .query_expansion import QueryExpansion
from .hyde_generator import HyDEGenerator, HyDEConfig
from app.services.vector_store import VectorStoreService
from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass 
class HybridSearchConfig:
    """Configuration for hybrid search."""
    use_bm25: bool = True
    use_dense: bool = True
    use_hyde: bool = True
    use_expansion: bool = True
    
    bm25_weight: float = 0.3
    dense_weight: float = 0.4
    hyde_weight: float = 0.3
    
    expansion_methods: List[str] = None
    fusion_strategy: str = "rrf"  # rrf, weighted, linear
    rrf_k: int = 60
    
    def __post_init__(self):
        if self.expansion_methods is None:
            self.expansion_methods = ['bert']  # Default to BERT expansion


class HybridRetriever:
    """
    Unified hybrid retriever combining multiple retrieval strategies.
    
    Components:
    - BM25 sparse retrieval
    - Dense vector similarity search  
    - Query expansion (T5, BERT, pseudo-relevance)
    - HyDE (Hypothetical Document Embeddings)
    - Reciprocal Rank Fusion (RRF)
    """
    
    def __init__(
        self,
        config: Optional[HybridSearchConfig] = None,
        vector_store: Optional[VectorStoreService] = None
    ):
        """
        Initialize hybrid retriever with all components.
        
        Args:
            config: Hybrid search configuration
            vector_store: Vector store service for dense search
        """
        self.config = config or HybridSearchConfig()
        
        # Initialize components
        self.bm25_retriever = BM25Retriever()
        self.vector_store = vector_store or VectorStoreService()
        self.query_expander = QueryExpansion()
        self.hyde_generator = HyDEGenerator()
        self.hyde_generator.set_vector_store(self.vector_store)
        
        # Initialize fusion strategies
        self.rrf = RecipocalRankFusion(k=self.config.rrf_k)
        self.score_fusion = ScoreFusion()
        
        # Cache for BM25 corpus
        self.bm25_corpus = []
        self.bm25_metadata = []
        
        logger.info("HybridRetriever initialized with components: "
                   f"BM25={self.config.use_bm25}, Dense={self.config.use_dense}, "
                   f"HyDE={self.config.use_hyde}, Expansion={self.config.use_expansion}")
    
    def set_weights(self, weights: Dict[str, float]):
        """Update component weights."""
        self.config.bm25_weight = weights.get('bm25', self.config.bm25_weight)
        self.config.dense_weight = weights.get('dense', self.config.dense_weight)
        self.config.hyde_weight = weights.get('hyde', self.config.hyde_weight)
        
        # Normalize weights
        total = self.config.bm25_weight + self.config.dense_weight + self.config.hyde_weight
        if total > 0:
            self.config.bm25_weight /= total
            self.config.dense_weight /= total
            self.config.hyde_weight /= total
    
    @property
    def weights(self) -> Dict[str, float]:
        """Get current component weights."""
        return {
            'bm25': self.config.bm25_weight,
            'dense': self.config.dense_weight,
            'hyde': self.config.hyde_weight
        }
    
    async def index_documents(
        self,
        documents: List[Dict[str, Any]],
        dataset_name: str
    ):
        """
        Index documents for both BM25 and dense retrieval.
        
        Args:
            documents: List of documents with 'content' and 'metadata'
            dataset_name: Name of the dataset
        """
        try:
            # Extract content for BM25
            contents = []
            for doc in documents:
                if 'chunks' in doc:
                    # Document already chunked
                    for chunk in doc['chunks']:
                        contents.append(chunk['content'])
                        self.bm25_metadata.append({
                            **chunk.get('metadata', {}),
                            'dataset_name': dataset_name
                        })
                else:
                    # Single document
                    contents.append(doc['content'])
                    self.bm25_metadata.append({
                        **doc.get('metadata', {}),
                        'dataset_name': dataset_name
                    })
            
            # Update BM25 index
            if contents:
                if self.bm25_corpus:
                    self.bm25_retriever.update_corpus(contents)
                else:
                    self.bm25_retriever.fit(contents)
                self.bm25_corpus.extend(contents)
            
            # Dense vectors are handled by vector_store.add_documents
            
            logger.info(f"Indexed {len(contents)} chunks for hybrid retrieval")
            
        except Exception as e:
            logger.error(f"Failed to index documents: {str(e)}")
            raise
    
    async def bm25_search(
        self,
        query: str,
        k: int = 10,
        dataset_filter: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform BM25 sparse retrieval.
        
        Args:
            query: Search query
            k: Number of results
            dataset_filter: Filter by dataset names
            
        Returns:
            BM25 search results
        """
        try:
            if not self.bm25_corpus:
                logger.warning("BM25 corpus is empty")
                return []
            
            # Get BM25 scores
            results = self.bm25_retriever.get_top_k(query, k=k*2)  # Get more for filtering
            
            # Filter by dataset if needed
            filtered_results = []
            for result in results:
                idx = result['index']
                metadata = self.bm25_metadata[idx] if idx < len(self.bm25_metadata) else {}
                
                if dataset_filter and metadata.get('dataset_name') not in dataset_filter:
                    continue
                
                filtered_results.append({
                    'doc_id': f"bm25_{idx}",
                    'content': self.bm25_corpus[idx],
                    'metadata': metadata,
                    'score': result['score'],
                    'rank': result['rank'],
                    'retriever': 'bm25'
                })
                
                if len(filtered_results) >= k:
                    break
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"BM25 search failed: {str(e)}")
            return []
    
    async def dense_search(
        self,
        query: str,
        k: int = 10,
        dataset_names: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform dense vector similarity search.
        
        Args:
            query: Search query
            k: Number of results
            dataset_names: Filter by datasets
            
        Returns:
            Dense search results
        """
        try:
            results = self.vector_store.search(
                query=query,
                dataset_names=dataset_names,
                k=k
            )
            
            formatted_results = []
            for rank, (doc, score) in enumerate(results, 1):
                formatted_results.append({
                    'doc_id': doc.metadata.get('document_id', f"dense_{rank}"),
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'score': score,
                    'rank': rank,
                    'retriever': 'dense'
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Dense search failed: {str(e)}")
            return []
    
    async def hyde_search(
        self,
        query: str,
        k: int = 10,
        dataset_names: Optional[List[str]] = None,
        role: str = "general"
    ) -> List[Dict[str, Any]]:
        """
        Perform HyDE search.
        
        Args:
            query: Search query
            k: Number of results
            dataset_names: Filter by datasets
            role: Role context for HyDE generation
            
        Returns:
            HyDE search results
        """
        try:
            results = await self.hyde_generator.hyde_search(
                query=query,
                k=k,
                dataset_names=dataset_names,
                role_context=role,
                use_multiple=False
            )
            
            # Add rank information
            for rank, result in enumerate(results, 1):
                result['rank'] = rank
                result['retriever'] = 'hyde'
                if 'doc_id' not in result:
                    result['doc_id'] = f"hyde_{rank}"
            
            return results
            
        except Exception as e:
            logger.error(f"HyDE search failed: {str(e)}")
            return []
    
    async def search(
        self,
        query: str,
        dataset_names: Optional[List[str]] = None,
        role: str = "general",
        k: int = 10,
        use_bm25: Optional[bool] = None,
        use_dense: Optional[bool] = None,
        use_hyde: Optional[bool] = None,
        use_expansion: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining all enabled retrieval methods.
        
        Args:
            query: Search query
            dataset_names: Filter by datasets
            role: Role context for generation
            k: Number of results to return
            use_bm25: Override config for BM25
            use_dense: Override config for dense search
            use_hyde: Override config for HyDE
            use_expansion: Override config for query expansion
            
        Returns:
            Fused and ranked search results
        """
        start_time = time.time()
        
        # Use provided flags or fall back to config
        use_bm25 = use_bm25 if use_bm25 is not None else self.config.use_bm25
        use_dense = use_dense if use_dense is not None else self.config.use_dense
        use_hyde = use_hyde if use_hyde is not None else self.config.use_hyde
        use_expansion = use_expansion if use_expansion is not None else self.config.use_expansion
        
        # Expand query if enabled
        queries = [query]
        if use_expansion:
            try:
                expanded = await self.query_expander.combine_expansions(
                    query,
                    methods=self.config.expansion_methods,
                    max_expansions=3
                )
                queries = expanded[:3]  # Use top 3 expansions
                logger.debug(f"Expanded query to {len(queries)} variants")
            except Exception as e:
                logger.warning(f"Query expansion failed: {str(e)}")
        
        # Collect results from all enabled methods
        all_rankings = []
        all_scores = []
        retriever_names = []
        
        # Run searches in parallel for each query variant
        for q in queries:
            tasks = []
            
            if use_bm25 and self.bm25_corpus:
                tasks.append(self.bm25_search(q, k, dataset_names))
            
            if use_dense:
                tasks.append(self.dense_search(q, k, dataset_names))
            
            if use_hyde:
                tasks.append(self.hyde_search(q, k, dataset_names, role))
            
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for i, result in enumerate(results):
                    if isinstance(result, list) and result:
                        all_rankings.append(result)
                        all_scores.append(result)
                        
                        # Track which retriever produced these results
                        if i == 0 and use_bm25:
                            retriever_names.append('bm25')
                        elif (i == 1 and use_dense) or (i == 0 and not use_bm25 and use_dense):
                            retriever_names.append('dense')
                        elif use_hyde:
                            retriever_names.append('hyde')
        
        if not all_rankings:
            logger.warning("No results from any retrieval method")
            return []
        
        # Fuse results based on strategy
        if self.config.fusion_strategy == "rrf":
            # Use RRF for ranking-based fusion
            if len(all_rankings) > 1:
                weights = self._calculate_weights(retriever_names)
                fused_results = self.rrf.weighted_fusion(
                    all_rankings,
                    weights,
                    k=self.config.rrf_k
                )
            else:
                fused_results = all_rankings[0]
        
        elif self.config.fusion_strategy == "linear":
            # Use linear combination for score-based fusion
            if len(all_scores) > 1:
                weights = self._calculate_weights(retriever_names)
                fused_results = self.score_fusion.linear_combination(
                    all_scores,
                    weights
                )
            else:
                fused_results = all_scores[0]
        
        elif self.config.fusion_strategy == "weighted":
            # Use weighted normalization
            if len(all_scores) > 1:
                weights = self._calculate_weights(retriever_names)
                fused_results = self.score_fusion.normalize_and_combine(
                    all_scores,
                    weights,
                    normalization='min_max'
                )
            else:
                fused_results = all_scores[0]
        else:
            # Default to first result set
            fused_results = all_rankings[0]
        
        # Ensure we have the right number of results
        final_results = fused_results[:k]
        
        # Add final scoring and metadata
        for i, result in enumerate(final_results):
            result['final_rank'] = i + 1
            if 'final_score' not in result:
                result['final_score'] = result.get(
                    'rrf_score',
                    result.get('combined_score', result.get('score', 0))
                )
            result['retrieval_method'] = 'hybrid'
            result['components_used'] = retriever_names
        
        elapsed = time.time() - start_time
        logger.info(f"Hybrid search completed in {elapsed:.2f}s - "
                   f"Used: {', '.join(retriever_names)} - "
                   f"Returned {len(final_results)} results")
        
        return final_results
    
    def _calculate_weights(self, retriever_names: List[str]) -> List[float]:
        """Calculate weights for each retriever based on config."""
        weights = []
        for name in retriever_names:
            if name == 'bm25':
                weights.append(self.config.bm25_weight)
            elif name == 'dense':
                weights.append(self.config.dense_weight)
            elif name == 'hyde':
                weights.append(self.config.hyde_weight)
            else:
                weights.append(1.0 / len(retriever_names))
        
        # Normalize
        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]
        
        return weights
    
    async def adaptive_search(
        self,
        query: str,
        dataset_names: Optional[List[str]] = None,
        role: str = "general",
        k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Adaptively choose retrieval methods based on query characteristics.
        
        Args:
            query: Search query
            dataset_names: Filter by datasets
            role: Role context
            k: Number of results
            
        Returns:
            Search results using adapted configuration
        """
        # Analyze query characteristics
        query_length = len(query.split())
        has_technical_terms = any(term in query.lower() for term in 
                                 ['algorithm', 'method', 'process', 'system', 'model'])
        is_question = query.strip().endswith('?')
        
        # Adapt configuration based on query
        if query_length < 3:
            # Short query: rely more on expansion and HyDE
            use_bm25 = True
            use_dense = True
            use_hyde = True
            use_expansion = True
        elif query_length > 15 or has_technical_terms:
            # Long/technical query: rely more on exact matching
            use_bm25 = True
            use_dense = True
            use_hyde = False
            use_expansion = False
        elif is_question:
            # Question: use HyDE for better semantic understanding
            use_bm25 = False
            use_dense = True
            use_hyde = True
            use_expansion = True
        else:
            # Default: use all methods
            use_bm25 = True
            use_dense = True
            use_hyde = True
            use_expansion = True
        
        logger.debug(f"Adaptive search config for '{query[:50]}...': "
                    f"BM25={use_bm25}, Dense={use_dense}, "
                    f"HyDE={use_hyde}, Expansion={use_expansion}")
        
        return await self.search(
            query=query,
            dataset_names=dataset_names,
            role=role,
            k=k,
            use_bm25=use_bm25,
            use_dense=use_dense,
            use_hyde=use_hyde,
            use_expansion=use_expansion
        )