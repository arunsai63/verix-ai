"""
HyDE (Hypothetical Document Embeddings) implementation for zero-shot retrieval.
Generates hypothetical documents from queries to improve semantic search.
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
from app.services.ai_providers import AIProviderFactory
from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class HyDEConfig:
    """Configuration for HyDE generation."""
    max_length: int = 300
    temperature: float = 0.7
    num_hypotheticals: int = 1
    combine_strategy: str = "weighted"  # weighted, max, average
    hyde_weight: float = 0.6
    original_weight: float = 0.4


class HyDEGenerator:
    """
    Implements Hypothetical Document Embeddings (HyDE) for improved zero-shot retrieval.
    
    HyDE works by:
    1. Generating a hypothetical document that would answer the query
    2. Embedding the hypothetical document
    3. Searching with the hypothetical document embedding
    4. Combining results with original query search
    """
    
    def __init__(
        self,
        config: Optional[HyDEConfig] = None,
        llm_provider: Optional[str] = None,
        embedding_provider: Optional[str] = None
    ):
        """
        Initialize HyDE generator with LLM and embedding models.
        
        Args:
            config: HyDE configuration
            llm_provider: LLM provider for document generation
            embedding_provider: Provider for embeddings
        """
        self.config = config or HyDEConfig()
        
        # Initialize LLM for hypothetical document generation
        llm_provider_name = llm_provider or settings.llm_provider
        llm_factory = AIProviderFactory.get_provider(llm_provider_name)
        self.llm = llm_factory.get_chat_model(
            temperature=self.config.temperature,
            max_tokens=self.config.max_length
        )
        
        # Initialize embeddings model
        embed_provider_name = embedding_provider or settings.embedding_provider
        embed_factory = AIProviderFactory.get_provider(embed_provider_name)
        self.embeddings_model = embed_factory.get_embeddings_model()
        
        # Store vector store reference (will be injected)
        self.vector_store = None
        
        logger.info(f"HyDEGenerator initialized with {llm_provider_name} LLM "
                   f"and {embed_provider_name} embeddings")
    
    def set_vector_store(self, vector_store):
        """Set the vector store for searching."""
        self.vector_store = vector_store
    
    async def generate_hypothetical_doc(
        self,
        query: str,
        role_context: str = "general",
        additional_context: Optional[str] = None
    ) -> str:
        """
        Generate a hypothetical document that would answer the query.
        
        Args:
            query: The user's question/query
            role_context: Role context (doctor, lawyer, hr, general)
            additional_context: Any additional context to consider
            
        Returns:
            Generated hypothetical document
        """
        try:
            # Create role-specific prompts
            role_prompts = {
                "doctor": "You are writing a medical document that would answer this query. Include relevant medical terminology, procedures, and clinical information.",
                "lawyer": "You are writing a legal document that would answer this query. Include relevant legal principles, precedents, and regulatory information.",
                "hr": "You are writing an HR policy document that would answer this query. Include relevant procedures, compliance requirements, and best practices.",
                "general": "You are writing an informative document that would comprehensively answer this query."
            }
            
            role_instruction = role_prompts.get(role_context, role_prompts["general"])
            
            # Build the prompt
            prompt = f"""{role_instruction}

Query: {query}

{f"Additional Context: {additional_context}" if additional_context else ""}

Write a detailed, factual document that would be retrieved to answer this query. 
The document should be comprehensive and directly address the question.
Do not include phrases like "This document answers..." or "To answer your question...".
Write as if this is an actual document that exists in a database.

Document:"""
            
            # Generate hypothetical document
            loop = asyncio.get_event_loop()
            hypothetical = await loop.run_in_executor(
                None,
                self.llm.invoke,
                prompt
            )
            
            # Extract text from response
            if hasattr(hypothetical, 'content'):
                hypothetical_text = hypothetical.content
            else:
                hypothetical_text = str(hypothetical)
            
            logger.debug(f"Generated hypothetical document for query '{query[:50]}...' "
                        f"({len(hypothetical_text)} chars)")
            
            return hypothetical_text
            
        except Exception as e:
            logger.error(f"Failed to generate hypothetical document: {str(e)}")
            # Return the query itself as fallback
            return query
    
    async def generate_multiple_hypotheticals(
        self,
        query: str,
        num_hypotheticals: int = 3,
        role_context: str = "general"
    ) -> List[str]:
        """
        Generate multiple hypothetical documents for diversity.
        
        Args:
            query: The user's query
            num_hypotheticals: Number of hypothetical documents to generate
            role_context: Role context for generation
            
        Returns:
            List of hypothetical documents
        """
        tasks = []
        for i in range(num_hypotheticals):
            # Vary the prompt slightly for diversity
            context = f"Perspective {i+1}: " if i > 0 else None
            tasks.append(
                self.generate_hypothetical_doc(query, role_context, context)
            )
        
        hypotheticals = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_hypotheticals = [
            h for h in hypotheticals 
            if isinstance(h, str) and len(h) > 0
        ]
        
        if not valid_hypotheticals:
            logger.warning("No valid hypothetical documents generated")
            return [query]  # Fallback to original query
        
        return valid_hypotheticals
    
    def embed_document(self, document: str) -> List[float]:
        """
        Generate embedding for a document.
        
        Args:
            document: Document text to embed
            
        Returns:
            Document embedding vector
        """
        try:
            embedding = self.embeddings_model.embed_documents([document])[0]
            return embedding
        except Exception as e:
            logger.error(f"Failed to embed document: {str(e)}")
            return None
    
    async def hyde_search(
        self,
        query: str,
        k: int = 10,
        dataset_names: Optional[List[str]] = None,
        role_context: str = "general",
        use_multiple: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Perform HyDE search: generate hypothetical document and search with it.
        
        Args:
            query: User's query
            k: Number of results to return
            dataset_names: Datasets to search in
            role_context: Role context for generation
            use_multiple: Whether to use multiple hypothetical documents
            
        Returns:
            Search results based on hypothetical document
        """
        if not self.vector_store:
            logger.error("Vector store not set for HyDE search")
            return []
        
        try:
            # Generate hypothetical document(s)
            if use_multiple:
                hypotheticals = await self.generate_multiple_hypotheticals(
                    query,
                    self.config.num_hypotheticals,
                    role_context
                )
            else:
                hypothetical = await self.generate_hypothetical_doc(
                    query,
                    role_context
                )
                hypotheticals = [hypothetical]
            
            # Search with each hypothetical document
            all_results = []
            for hypothetical in hypotheticals:
                # Use the hypothetical document as the search query
                results = self.vector_store.search(
                    hypothetical,
                    dataset_names=dataset_names,
                    k=k
                )
                
                # Add metadata about HyDE generation
                for doc, score in results:
                    all_results.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": score,
                        "search_type": "hyde",
                        "hypothetical_used": hypothetical[:100] + "..."
                    })
            
            # Deduplicate and sort by score
            seen = set()
            unique_results = []
            for result in sorted(all_results, key=lambda x: x['score'], reverse=True):
                doc_id = result['metadata'].get('document_id', result['content'][:50])
                if doc_id not in seen:
                    seen.add(doc_id)
                    unique_results.append(result)
            
            return unique_results[:k]
            
        except Exception as e:
            logger.error(f"HyDE search failed: {str(e)}")
            return []
    
    async def combine_with_original(
        self,
        hyde_results: List[Dict[str, Any]],
        original_results: List[Dict[str, Any]],
        hyde_weight: Optional[float] = None,
        strategy: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Combine HyDE results with original query results.
        
        Args:
            hyde_results: Results from HyDE search
            original_results: Results from original query search
            hyde_weight: Weight for HyDE results (0-1)
            strategy: Combination strategy (weighted, max, average)
            
        Returns:
            Combined and re-ranked results
        """
        hyde_weight = hyde_weight or self.config.hyde_weight
        original_weight = 1 - hyde_weight
        strategy = strategy or self.config.combine_strategy
        
        # Create document ID to result mapping
        combined_docs = {}
        
        # Process HyDE results
        for result in hyde_results:
            doc_id = result.get('doc_id') or result['metadata'].get(
                'document_id',
                result['content'][:50]
            )
            
            if doc_id not in combined_docs:
                combined_docs[doc_id] = {
                    **result,
                    'hyde_score': result['score'],
                    'original_score': 0
                }
            else:
                combined_docs[doc_id]['hyde_score'] = result['score']
        
        # Process original results
        for result in original_results:
            doc_id = result.get('doc_id') or result.get('metadata', {}).get(
                'document_id',
                result.get('content', '')[:50]
            )
            
            if doc_id not in combined_docs:
                combined_docs[doc_id] = {
                    **result,
                    'hyde_score': 0,
                    'original_score': result.get('score', 0)
                }
            else:
                combined_docs[doc_id]['original_score'] = result.get('score', 0)
        
        # Calculate combined scores based on strategy
        for doc_id, doc in combined_docs.items():
            hyde_score = doc['hyde_score']
            original_score = doc['original_score']
            
            if strategy == "weighted":
                doc['combined_score'] = (
                    hyde_weight * hyde_score + 
                    original_weight * original_score
                )
            elif strategy == "max":
                doc['combined_score'] = max(hyde_score, original_score)
            elif strategy == "average":
                scores = [s for s in [hyde_score, original_score] if s > 0]
                doc['combined_score'] = np.mean(scores) if scores else 0
            else:
                # Default to weighted
                doc['combined_score'] = (
                    hyde_weight * hyde_score + 
                    original_weight * original_score
                )
            
            doc['combination_method'] = f"hyde_{strategy}"
        
        # Sort by combined score
        combined_results = sorted(
            combined_docs.values(),
            key=lambda x: x['combined_score'],
            reverse=True
        )
        
        logger.debug(f"Combined {len(hyde_results)} HyDE and {len(original_results)} "
                    f"original results into {len(combined_results)} unique documents")
        
        return combined_results
    
    async def adaptive_hyde(
        self,
        query: str,
        query_complexity: float,
        k: int = 10
    ) -> Dict[str, Any]:
        """
        Adaptively decide whether to use HyDE based on query complexity.
        
        Args:
            query: User's query
            query_complexity: Estimated query complexity (0-1)
            k: Number of results
            
        Returns:
            Search configuration based on complexity
        """
        # Simple heuristics for HyDE usage
        query_length = len(query.split())
        
        # Decide HyDE configuration based on complexity
        if query_complexity > 0.7 or query_length > 10:
            # Complex query: use multiple hypotheticals
            return {
                "use_hyde": True,
                "num_hypotheticals": 3,
                "hyde_weight": 0.7,
                "strategy": "weighted"
            }
        elif query_complexity > 0.4 or query_length > 5:
            # Medium complexity: use single hypothetical
            return {
                "use_hyde": True,
                "num_hypotheticals": 1,
                "hyde_weight": 0.5,
                "strategy": "weighted"
            }
        else:
            # Simple query: skip HyDE
            return {
                "use_hyde": False,
                "num_hypotheticals": 0,
                "hyde_weight": 0,
                "strategy": None
            }
    
    def analyze_hyde_effectiveness(
        self,
        hyde_results: List[Dict[str, Any]],
        original_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze the effectiveness of HyDE for a given query.
        
        Args:
            hyde_results: Results from HyDE search
            original_results: Results from original search
            
        Returns:
            Analysis metrics
        """
        # Calculate overlap
        hyde_ids = {r.get('doc_id', r['content'][:50]) for r in hyde_results[:10]}
        original_ids = {r.get('doc_id', r.get('content', '')[:50]) for r in original_results[:10]}
        overlap = hyde_ids.intersection(original_ids)
        
        # Calculate average scores
        hyde_avg_score = np.mean([r.get('score', 0) for r in hyde_results[:10]]) if hyde_results else 0
        original_avg_score = np.mean([r.get('score', 0) for r in original_results[:10]]) if original_results else 0
        
        # Determine effectiveness
        score_improvement = hyde_avg_score - original_avg_score
        diversity_score = 1 - (len(overlap) / 10)
        
        return {
            "overlap_ratio": len(overlap) / 10,
            "unique_hyde_docs": len(hyde_ids - original_ids),
            "unique_original_docs": len(original_ids - hyde_ids),
            "hyde_avg_score": hyde_avg_score,
            "original_avg_score": original_avg_score,
            "score_improvement": score_improvement,
            "diversity_score": diversity_score,
            "recommendation": "use_hyde" if score_improvement > 0.1 or diversity_score > 0.5 else "skip_hyde"
        }