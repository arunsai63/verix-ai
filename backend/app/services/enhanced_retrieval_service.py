"""
Enhanced Retrieval Service with Advanced RAG Mechanisms.
Integrates hybrid retrieval, cross-encoder reranking, and query expansion.
"""

import logging
from typing import List, Dict, Any, Optional, Literal
import numpy as np
from datetime import datetime
import asyncio

from app.services.advanced_retrieval.hybrid_retriever import HybridRetriever
from app.services.advanced_retrieval.integrated_ranker import IntegratedRanker
from app.services.advanced_retrieval.query_expansion import QueryExpansion
from app.services.advanced_retrieval.hyde_generator import HyDEGenerator
from app.services.llm_service import LLMService
from app.services.embedding_service import EmbeddingService
from app.core.config import settings

logger = logging.getLogger(__name__)


class EnhancedRetrievalService:
    """Enhanced retrieval service with advanced RAG capabilities."""
    
    def __init__(
        self,
        retrieval_strategy: Literal["hybrid", "semantic", "keyword", "auto"] = "hybrid",
        enable_reranking: bool = True,
        enable_query_expansion: bool = True,
        enable_hyde: bool = True
    ):
        """
        Initialize enhanced retrieval service.
        
        Args:
            retrieval_strategy: Strategy for retrieval
            enable_reranking: Enable cross-encoder reranking
            enable_query_expansion: Enable query expansion
            enable_hyde: Enable HyDE generation
        """
        # Core services
        self.llm_service = LLMService()
        self.embedding_service = EmbeddingService()
        
        # Advanced components
        self.hybrid_retriever = HybridRetriever()
        self.ranker = IntegratedRanker() if enable_reranking else None
        self.query_expander = QueryExpansion() if enable_query_expansion else None
        self.hyde_generator = HyDEGenerator() if enable_hyde else None
        
        self.retrieval_strategy = retrieval_strategy
        self.enable_reranking = enable_reranking
        self.enable_query_expansion = enable_query_expansion
        self.enable_hyde = enable_hyde
        
        logger.info(
            f"Enhanced retrieval initialized: "
            f"strategy={retrieval_strategy}, "
            f"reranking={enable_reranking}, "
            f"expansion={enable_query_expansion}, "
            f"hyde={enable_hyde}"
        )
    
    async def retrieve(
        self,
        query: str,
        dataset_name: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        role: Optional[str] = None,
        use_adaptive: bool = True
    ) -> Dict[str, Any]:
        """
        Perform enhanced retrieval with advanced mechanisms.
        
        Args:
            query: Search query
            dataset_name: Dataset to search
            top_k: Number of results
            filters: Metadata filters
            role: User role for personalization
            use_adaptive: Use adaptive strategy selection
            
        Returns:
            Retrieval results with metadata
        """
        start_time = datetime.utcnow()
        
        # Expand query if enabled
        expanded_queries = [query]
        if self.enable_query_expansion and self.query_expander:
            try:
                expansion_result = self.query_expander.expand_query(query)
                expanded_queries.extend(expansion_result.get('expanded_queries', []))
                logger.info(f"Expanded query to {len(expanded_queries)} variations")
            except Exception as e:
                logger.warning(f"Query expansion failed: {str(e)}")
        
        # Generate HyDE if enabled
        hyde_doc = None
        if self.enable_hyde and self.hyde_generator:
            try:
                hyde_doc = await self._generate_hyde_async(query, role)
                logger.info("Generated HyDE document for retrieval")
            except Exception as e:
                logger.warning(f"HyDE generation failed: {str(e)}")
        
        # Perform hybrid retrieval
        retrieval_results = await self._perform_retrieval(
            queries=expanded_queries,
            hyde_doc=hyde_doc,
            dataset_name=dataset_name,
            filters=filters,
            top_k=top_k * 2 if self.enable_reranking else top_k,
            use_adaptive=use_adaptive
        )
        
        # Apply reranking if enabled
        if self.enable_reranking and self.ranker and retrieval_results:
            try:
                reranked_results = self.ranker.rerank(
                    query=query,
                    results=retrieval_results[:top_k * 2],
                    top_k=top_k,
                    strategy="cascade"
                )
                retrieval_results = reranked_results
                logger.info(f"Reranked {len(reranked_results)} results")
            except Exception as e:
                logger.warning(f"Reranking failed: {str(e)}")
        
        # Limit to top_k
        retrieval_results = retrieval_results[:top_k]
        
        # Calculate retrieval metrics
        metrics = self._calculate_metrics(retrieval_results)
        
        elapsed_time = (datetime.utcnow() - start_time).total_seconds()
        
        return {
            "query": query,
            "expanded_queries": expanded_queries,
            "results": retrieval_results,
            "metrics": metrics,
            "metadata": {
                "dataset": dataset_name,
                "top_k": top_k,
                "strategy": self.retrieval_strategy,
                "hyde_used": hyde_doc is not None,
                "query_expanded": len(expanded_queries) > 1,
                "reranking_applied": self.enable_reranking,
                "retrieval_time": elapsed_time,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    
    async def retrieve_multi_hop(
        self,
        query: str,
        dataset_name: str,
        max_hops: int = 3,
        top_k_per_hop: int = 5
    ) -> Dict[str, Any]:
        """
        Perform multi-hop retrieval for complex queries.
        
        Args:
            query: Initial query
            dataset_name: Dataset to search
            max_hops: Maximum number of retrieval hops
            top_k_per_hop: Results per hop
            
        Returns:
            Multi-hop retrieval results
        """
        hops = []
        current_query = query
        accumulated_context = []
        
        for hop_num in range(max_hops):
            # Retrieve for current query
            hop_results = await self.retrieve(
                query=current_query,
                dataset_name=dataset_name,
                top_k=top_k_per_hop,
                use_adaptive=True
            )
            
            if not hop_results["results"]:
                break
            
            hops.append({
                "hop": hop_num + 1,
                "query": current_query,
                "results": hop_results["results"][:top_k_per_hop]
            })
            
            # Accumulate context
            for result in hop_results["results"][:3]:
                accumulated_context.append(result["content"][:500])
            
            # Generate next query based on results
            if hop_num < max_hops - 1:
                next_query = await self._generate_followup_query(
                    original_query=query,
                    current_query=current_query,
                    context="\n".join(accumulated_context)
                )
                
                if not next_query or next_query == current_query:
                    break
                
                current_query = next_query
                logger.info(f"Hop {hop_num + 2} query: {current_query}")
        
        return {
            "original_query": query,
            "hops": hops,
            "total_hops": len(hops),
            "final_context": accumulated_context,
            "metadata": {
                "dataset": dataset_name,
                "max_hops": max_hops,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    
    async def _perform_retrieval(
        self,
        queries: List[str],
        hyde_doc: Optional[str],
        dataset_name: str,
        filters: Optional[Dict[str, Any]],
        top_k: int,
        use_adaptive: bool
    ) -> List[Dict[str, Any]]:
        """Perform the actual retrieval."""
        all_results = []
        
        # Retrieve for each expanded query
        for query in queries:
            if use_adaptive and self.retrieval_strategy == "hybrid":
                # Use adaptive hybrid retrieval
                results = self.hybrid_retriever.adaptive_search(
                    query=query,
                    top_k=top_k,
                    filters=filters
                )
            else:
                # Use specified strategy
                results = self.hybrid_retriever.search(
                    query=query,
                    top_k=top_k,
                    search_type=self.retrieval_strategy,
                    filters=filters
                )
            
            all_results.extend(results)
        
        # Add HyDE results if available
        if hyde_doc:
            hyde_results = self.hybrid_retriever.search(
                query=hyde_doc,
                top_k=top_k // 2,
                search_type="semantic",
                filters=filters
            )
            all_results.extend(hyde_results)
        
        # Deduplicate results
        seen_ids = set()
        unique_results = []
        for result in all_results:
            result_id = result.get("id") or result.get("content", "")[:100]
            if result_id not in seen_ids:
                seen_ids.add(result_id)
                unique_results.append(result)
        
        # Sort by score
        unique_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        return unique_results
    
    async def _generate_hyde_async(self, query: str, role: Optional[str]) -> str:
        """Generate HyDE document asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.hyde_generator.generate_hypothesis,
            query,
            role
        )
    
    async def _generate_followup_query(
        self,
        original_query: str,
        current_query: str,
        context: str
    ) -> Optional[str]:
        """Generate follow-up query for multi-hop retrieval."""
        prompt = f"""Based on the original question and the context found so far, 
        generate a follow-up search query to find additional relevant information.
        
        Original question: {original_query}
        Current query: {current_query}
        
        Context found so far:
        {context[:1000]}
        
        Generate a follow-up query that explores a different aspect or seeks missing information.
        Return only the query text, nothing else.
        """
        
        try:
            response = await self.llm_service.generate(
                prompt=prompt,
                max_tokens=100,
                temperature=0.7
            )
            
            if response and response.strip() and response.strip() != current_query:
                return response.strip()
        except Exception as e:
            logger.error(f"Failed to generate follow-up query: {str(e)}")
        
        return None
    
    def _calculate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate retrieval quality metrics."""
        if not results:
            return {
                "avg_score": 0.0,
                "score_variance": 0.0,
                "coverage": 0.0,
                "diversity": 0.0
            }
        
        scores = [r.get("score", 0) for r in results]
        
        # Calculate diversity (unique sources)
        sources = set()
        for r in results:
            if "metadata" in r and "source" in r["metadata"]:
                sources.add(r["metadata"]["source"])
        
        diversity = len(sources) / len(results) if results else 0
        
        return {
            "avg_score": float(np.mean(scores)),
            "score_variance": float(np.var(scores)),
            "coverage": len(results) / 10,  # Assuming target of 10
            "diversity": diversity,
            "max_score": float(max(scores)) if scores else 0.0,
            "min_score": float(min(scores)) if scores else 0.0
        }