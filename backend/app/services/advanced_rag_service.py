"""
Advanced RAG Service with sophisticated retrieval and processing techniques.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import Document
from langchain.chains import LLMChain
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from app.core.config import settings
from app.services.ai_providers import AIProviderFactory
from app.services.vector_store import VectorStoreService
import json
from sentence_transformers import CrossEncoder
import re
from collections import Counter

logger = logging.getLogger(__name__)


@dataclass
class RetrievalConfig:
    """Configuration for advanced retrieval."""
    use_hybrid_search: bool = True
    use_query_expansion: bool = True
    use_multi_query: bool = True
    use_reranking: bool = True
    use_contextual_compression: bool = True
    max_context_length: int = 12000
    top_k_initial: int = 20
    top_k_reranked: int = 10
    bm25_weight: float = 0.3
    semantic_weight: float = 0.7
    expansion_queries: int = 3
    min_relevance_score: float = 0.3


class AdvancedRAGService:
    """Advanced Retrieval-Augmented Generation service with state-of-the-art techniques."""
    
    def __init__(
        self, 
        cross_encoder_model: str = 'cross-encoder/ms-marco-MiniLM-L-12-v2',
        config: Optional[RetrievalConfig] = None
    ):
        provider = AIProviderFactory.get_provider()
        self.llm = provider.get_chat_model(
            temperature=0.1,
            max_tokens=8192
        )
        self.cross_encoder = CrossEncoder(cross_encoder_model)
        self.vector_store = VectorStoreService()
        self.config = config or RetrievalConfig()
        
        # Role-specific prompts with enhanced instructions
        self.role_prompts = {
            "doctor": {
                "system": """You are an AI assistant helping medical professionals analyze patient documents and medical literature.
                Your responses should be:
                1. Clinically accurate and evidence-based
                2. Include relevant medical terminology with explanations
                3. Highlight potential drug interactions, contraindications, and safety concerns
                4. Reference specific medical guidelines when applicable
                5. Include appropriate medical disclaimers
                
                IMPORTANT: This is for informational purposes only and not a substitute for professional medical advice, diagnosis, or treatment.""",
                "tone": "professional, clinical, precise, evidence-based"
            },
            "lawyer": {
                "system": """You are an AI assistant helping legal professionals analyze case files and legal documents.
                Your responses should be:
                1. Legally precise and analytically rigorous
                2. Reference specific statutes, regulations, or case law when applicable
                3. Identify potential legal issues and arguments
                4. Consider multiple interpretations where applicable
                5. Include appropriate legal disclaimers
                
                IMPORTANT: This is for informational purposes only and does not constitute legal advice.""",
                "tone": "formal, precise, analytical, authoritative"
            },
            "hr": {
                "system": """You are an AI assistant helping HR professionals analyze policies, employee documents, and compliance materials.
                Your responses should be:
                1. Focus on actionable insights and best practices
                2. Highlight compliance requirements and risks
                3. Reference relevant labor laws and regulations
                4. Consider both employer and employee perspectives
                5. Provide practical implementation guidance
                
                IMPORTANT: This is for informational purposes only. Consult with legal counsel for specific situations.""",
                "tone": "professional, clear, actionable, balanced"
            },
            "general": {
                "system": """You are an advanced AI assistant with expertise in comprehensive document analysis and information synthesis.
                Your goal is to provide the most accurate, thorough, and insightful answers possible.
                
                Your responses should:
                1. Thoroughly analyze all provided context
                2. Synthesize information from multiple sources
                3. Identify patterns, contradictions, and key insights
                4. Provide structured, well-organized answers
                5. Always cite sources using [Source N] format
                6. Acknowledge limitations and uncertainties
                7. Suggest areas for further investigation when appropriate""",
                "tone": "analytical, comprehensive, precise, insightful"
            }
        }

    async def _expand_query_multi(self, query: str) -> List[str]:
        """Generate multiple query expansions for better coverage."""
        prompt = """You are a query expansion expert. Generate {num} different reformulations of the following query.
        Each reformulation should:
        1. Maintain the original intent
        2. Use different terminology or phrasing
        3. Add relevant context or specifications
        4. Include related concepts and synonyms
        
        Original query: {query}
        
        Output format (one per line):
        1. [First reformulation]
        2. [Second reformulation]
        3. [Third reformulation]
        """
        
        response = await self.llm.apredict(
            prompt.format(num=self.config.expansion_queries, query=query)
        )
        
        # Parse the response to extract queries
        expanded_queries = [query]  # Include original
        lines = response.strip().split('\n')
        for line in lines:
            # Extract query from numbered format
            match = re.match(r'\d+\.\s*(.*)', line.strip())
            if match:
                expanded_queries.append(match.group(1))
        
        return expanded_queries[:self.config.expansion_queries + 1]

    async def _decompose_complex_query(self, query: str) -> List[str]:
        """Decompose complex queries into simpler sub-queries."""
        prompt = """Analyze the following query and break it down into simpler, atomic sub-queries if it contains multiple questions or complex concepts.
        If the query is already simple, return it as is.
        
        Query: {query}
        
        Output format (one per line):
        - [Sub-query 1]
        - [Sub-query 2]
        - [etc.]
        """
        
        response = await self.llm.apredict(prompt.format(query=query))
        
        sub_queries = []
        lines = response.strip().split('\n')
        for line in lines:
            match = re.match(r'-\s*(.*)', line.strip())
            if match:
                sub_queries.append(match.group(1))
        
        return sub_queries if sub_queries else [query]

    def _hybrid_search(
        self, 
        query: str, 
        documents: List[Document],
        dataset_names: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Perform hybrid search combining semantic and keyword search."""
        
        # Semantic search using vector store
        semantic_results_raw = self.vector_store.search(
            query=query,
            dataset_names=dataset_names,
            k=self.config.top_k_initial
        )
        
        # Convert semantic results to dict format
        semantic_results = []
        for doc, score in semantic_results_raw:
            semantic_results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": max(0, score)  # Ensure non-negative scores
            })
        
        # BM25 keyword search
        if documents:
            bm25_retriever = BM25Retriever.from_documents(documents)
            bm25_retriever.k = self.config.top_k_initial
            bm25_results = bm25_retriever.get_relevant_documents(query)
            
            # Convert BM25 results to dict format
            bm25_dict_results = []
            for doc in bm25_results:
                bm25_dict_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": 0.0  # BM25 doesn't provide scores directly
                })
            
            # Combine results with weighted scoring
            combined_results = self._combine_search_results(
                semantic_results, 
                bm25_dict_results
            )
        else:
            combined_results = semantic_results
        
        return combined_results

    def _combine_search_results(
        self,
        semantic_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Combine semantic and BM25 results with weighted scoring."""
        
        # Create a mapping of content to results
        result_map = {}
        
        # Add semantic results with weighted scores
        for result in semantic_results:
            key = result["content"][:200]  # Use first 200 chars as key
            if key not in result_map:
                result_map[key] = result.copy()
                result_map[key]["combined_score"] = result.get("score", 0) * self.config.semantic_weight
            else:
                result_map[key]["combined_score"] += result.get("score", 0) * self.config.semantic_weight
        
        # Add BM25 results with weighted scores
        for i, result in enumerate(bm25_results):
            key = result["content"][:200]
            # Calculate BM25 score based on rank
            bm25_score = 1.0 / (i + 1)  # Simple rank-based scoring
            
            if key not in result_map:
                result_map[key] = result.copy()
                result_map[key]["combined_score"] = bm25_score * self.config.bm25_weight
            else:
                result_map[key]["combined_score"] += bm25_score * self.config.bm25_weight
        
        # Sort by combined score
        combined_results = list(result_map.values())
        combined_results.sort(key=lambda x: x.get("combined_score", 0), reverse=True)
        
        return combined_results[:self.config.top_k_initial]

    def _rerank_with_crossencoder(
        self, 
        query: str, 
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Rerank results using CrossEncoder for better relevance."""
        if not results:
            return []
        
        # Prepare pairs for cross-encoder
        pairs = [[query, result["content"]] for result in results]
        
        # Get relevance scores
        scores = self.cross_encoder.predict(pairs)
        
        # Add rerank scores to results
        for result, score in zip(results, scores):
            result["rerank_score"] = float(score)
        
        # Filter by minimum relevance score
        filtered_results = [
            r for r in results 
            if r["rerank_score"] >= self.config.min_relevance_score
        ]
        
        # Sort by rerank score
        filtered_results.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        return filtered_results[:self.config.top_k_reranked]

    async def _contextual_compression(
        self,
        query: str,
        documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Compress document content to only relevant parts."""
        
        compression_prompt = """Given the following question and document excerpt, extract only the parts that are directly relevant to answering the question.
        If the document doesn't contain relevant information, return "NOT_RELEVANT".
        
        Question: {query}
        
        Document:
        {document}
        
        Relevant excerpt:"""
        
        compressed_docs = []
        for doc in documents:
            response = await self.llm.apredict(
                compression_prompt.format(
                    query=query,
                    document=doc["content"][:2000]  # Limit document size
                )
            )
            
            if response.strip() != "NOT_RELEVANT":
                doc_copy = doc.copy()
                doc_copy["compressed_content"] = response.strip()
                doc_copy["original_content"] = doc["content"]
                compressed_docs.append(doc_copy)
        
        return compressed_docs

    async def generate_answer(
        self,
        query: str,
        search_results: List[Dict[str, Any]],
        role: str = "general",
        dataset_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate an answer using advanced RAG techniques.
        
        Args:
            query: User's question
            search_results: Initial retrieved documents
            role: User role for response formatting
            dataset_names: Optional list of dataset names to search
            
        Returns:
            Generated answer with citations and metadata
        """
        try:
            # 1. Query expansion and decomposition
            expanded_queries = []
            if self.config.use_query_expansion:
                expanded_queries = await self._expand_query_multi(query)
                
            # Decompose complex queries
            sub_queries = await self._decompose_complex_query(query)
            
            # 2. Perform retrieval for all queries
            all_results = []
            
            # Search with expanded queries
            if self.config.use_multi_query and expanded_queries:
                for exp_query in expanded_queries:
                    results = self.vector_store.search(
                        query=exp_query,
                        dataset_names=dataset_names,
                        k=self.config.top_k_initial // len(expanded_queries)
                    )
                    # Convert tuple results to dict format
                    for doc, score in results:
                        all_results.append({
                            "content": doc.page_content,
                            "metadata": doc.metadata,
                            "score": max(0, score)  # Ensure non-negative scores
                        })
            else:
                all_results = search_results
            
            # Remove duplicates based on content
            seen = set()
            unique_results = []
            for result in all_results:
                content_key = result["content"][:200]
                if content_key not in seen:
                    seen.add(content_key)
                    unique_results.append(result)
            
            # 3. Rerank if enabled
            if self.config.use_reranking and unique_results:
                reranked_results = self._rerank_with_crossencoder(query, unique_results)
            else:
                reranked_results = unique_results[:self.config.top_k_reranked]
            
            # 4. Contextual compression if enabled
            if self.config.use_contextual_compression and reranked_results:
                compressed_results = await self._contextual_compression(query, reranked_results)
                # Use compressed content if available
                for i, result in enumerate(reranked_results):
                    for compressed in compressed_results:
                        if result["content"][:200] == compressed["content"][:200]:
                            reranked_results[i]["display_content"] = compressed.get("compressed_content", result["content"])
                            break
            
            if not reranked_results:
                return {
                    "answer": "No relevant documents found to answer your query. Please try rephrasing your question or check if the relevant documents have been uploaded.",
                    "citations": [],
                    "highlights": [],
                    "confidence": "low",
                    "metadata": {
                        "techniques_used": self._get_techniques_used(),
                        "queries_expanded": len(expanded_queries),
                        "documents_retrieved": 0
                    }
                }
            
            # 5. Prepare enhanced context
            context = self._prepare_enhanced_context(reranked_results)
            
            # 6. Generate response
            role_config = self.role_prompts.get(role, self.role_prompts["general"])
            prompt = self._create_advanced_prompt(role_config["system"])
            
            response = await self._generate_comprehensive_response(
                prompt, query, context, role_config["tone"], sub_queries
            )
            
            # 7. Parse and enhance response
            parsed_response = self._parse_enhanced_response(response, reranked_results)
            
            # Add role-specific disclaimer
            if role != "general":
                parsed_response["disclaimer"] = self._get_disclaimer(role)
            
            # Add metadata about the retrieval process
            parsed_response["metadata"] = {
                "techniques_used": self._get_techniques_used(),
                "queries_expanded": len(expanded_queries),
                "documents_retrieved": len(reranked_results),
                "sub_queries": sub_queries if len(sub_queries) > 1 else None
            }
            
            return parsed_response
            
        except Exception as e:
            logger.error(f"Error in advanced RAG generation: {str(e)}")
            return {
                "answer": "An error occurred while generating the answer. Please try again.",
                "error": str(e),
                "citations": [],
                "highlights": [],
                "confidence": "error"
            }

    def _prepare_enhanced_context(
        self, 
        results: List[Dict[str, Any]]
    ) -> str:
        """Prepare enhanced context with better structure and metadata."""
        context_parts = []
        total_size = 0
        
        for idx, result in enumerate(results, 1):
            metadata = result.get("metadata", {})
            source = metadata.get("filename", "Unknown")
            dataset = metadata.get("dataset_name", "Unknown")
            chunk_index = metadata.get("chunk_index", 0)
            
            # Use compressed content if available, otherwise original
            content = result.get("display_content", result.get("content", ""))
            
            # Add relevance score to context
            relevance = result.get("rerank_score", result.get("score", 0))
            
            result_text = f"""[Source {idx}]
File: {source}
Dataset: {dataset}
Chunk: {chunk_index}
Relevance: {relevance:.2f}
Content:
{content}
---"""
            
            if total_size + len(result_text) > self.config.max_context_length:
                break
            
            context_parts.append(result_text)
            total_size += len(result_text)
        
        return "\n".join(context_parts)

    def _create_advanced_prompt(self, system_message: str) -> ChatPromptTemplate:
        """Create an advanced prompt template with better instructions."""
        system_template = SystemMessagePromptTemplate.from_template(system_message)
        
        human_template = """You are analyzing documents to answer a question. Follow these guidelines:

1. COMPREHENSIVENESS: Provide a thorough answer that addresses all aspects of the question
2. ACCURACY: Base your answer strictly on the provided context
3. SYNTHESIS: Integrate information from multiple sources when relevant
4. CITATIONS: Always cite sources using [Source N] format
5. STRUCTURE: Organize your answer logically with clear sections if needed
6. UNCERTAINTY: Clearly indicate when information is incomplete or contradictory
7. INSIGHTS: Identify patterns, trends, and key insights from the documents

Context:
{context}

Main Question: {query}

{sub_queries_section}

Please provide your response in the following JSON format:
{{
    "answer": "A comprehensive, well-structured answer with [Source N] citations throughout. Use markdown formatting for better readability. Include multiple paragraphs if needed to fully address the question.",
    "highlights": [
        "Key insight or finding 1 with [Source N]",
        "Key insight or finding 2 with [Source N]",
        "Key insight or finding 3 with [Source N]"
    ],
    "confidence": "high/medium/low - based on the quality and relevance of available information",
    "suggested_followup": "A relevant follow-up question that would provide additional value",
    "gaps": "Any information gaps or areas where the documents don't provide complete answers (optional)",
    "contradictions": "Any contradictions found between sources (optional)"
}}

Tone: {tone}
"""
        
        human_message = HumanMessagePromptTemplate.from_template(human_template)
        
        return ChatPromptTemplate.from_messages([system_template, human_message])

    async def _generate_comprehensive_response(
        self,
        prompt: ChatPromptTemplate,
        query: str,
        context: str,
        tone: str,
        sub_queries: List[str]
    ) -> str:
        """Generate comprehensive response using enhanced prompt."""
        
        # Format sub-queries section if multiple queries
        sub_queries_section = ""
        if len(sub_queries) > 1:
            sub_queries_section = "Sub-questions to address:\n" + "\n".join(
                [f"- {sq}" for sq in sub_queries]
            )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        
        response = await chain.ainvoke({
            "query": query,
            "context": context,
            "tone": tone,
            "sub_queries_section": sub_queries_section
        })
        
        # Handle different response types properly
        if isinstance(response, dict):
            return response.get('text', str(response))
        elif isinstance(response, (list, tuple)):
            # If it's a tuple or list, convert to string
            return str(response[0]) if response else ""
        else:
            return str(response)

    def _parse_enhanced_response(
        self,
        response: str,
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Parse enhanced response with additional metadata."""
        try:
            parsed = json.loads(response)
            
            # Enhanced citation extraction
            citations = []
            for idx, result in enumerate(results, 1):
                source_pattern = f"\\[Source {idx}\\]"
                is_mentioned = bool(re.search(source_pattern, parsed.get("answer", "")))
                
                metadata = result.get("metadata", {})
                citations.append({
                    "source_number": idx,
                    "filename": metadata.get("filename", "Unknown"),
                    "dataset": metadata.get("dataset_name", "Unknown"),
                    "chunk_index": metadata.get("chunk_index", 0),
                    "snippet": result.get("display_content", result["content"])[:400] + "...",
                    "relevance_score": result.get("rerank_score", result.get("score", 0)),
                    "mentioned_in_answer": is_mentioned,
                    "full_content": result.get("content", "")  # Include full content for reference
                })
            
            # Sort citations by relevance
            citations.sort(key=lambda x: x["relevance_score"], reverse=True)
            
            return {
                "answer": parsed.get("answer", "Unable to generate answer"),
                "citations": citations,
                "highlights": parsed.get("highlights", []),
                "confidence": parsed.get("confidence", "medium"),
                "suggested_followup": parsed.get("suggested_followup"),
                "gaps": parsed.get("gaps"),
                "contradictions": parsed.get("contradictions")
            }
            
        except json.JSONDecodeError:
            # Fallback parsing
            citations = []
            for idx, result in enumerate(results, 1):
                metadata = result.get("metadata", {})
                citations.append({
                    "source_number": idx,
                    "filename": metadata.get("filename", "Unknown"),
                    "dataset": metadata.get("dataset_name", "Unknown"),
                    "chunk_index": metadata.get("chunk_index", 0),
                    "snippet": result["content"][:400] + "...",
                    "relevance_score": result.get("rerank_score", result.get("score", 0)),
                    "mentioned_in_answer": False
                })
            
            return {
                "answer": response,
                "citations": citations,
                "highlights": [],
                "confidence": "low"
            }

    def _get_techniques_used(self) -> List[str]:
        """Get list of techniques used based on configuration."""
        techniques = []
        if self.config.use_hybrid_search:
            techniques.append("Hybrid Search (Semantic + BM25)")
        if self.config.use_query_expansion:
            techniques.append("Query Expansion")
        if self.config.use_multi_query:
            techniques.append("Multi-Query Retrieval")
        if self.config.use_reranking:
            techniques.append("Cross-Encoder Reranking")
        if self.config.use_contextual_compression:
            techniques.append("Contextual Compression")
        return techniques

    def _get_disclaimer(self, role: str) -> str:
        """Get role-specific disclaimer."""
        disclaimers = {
            "doctor": "⚠️ Medical Disclaimer: This information is for educational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.",
            "lawyer": "⚠️ Legal Disclaimer: This information is for educational purposes only and does not constitute legal advice. For specific legal advice, please consult with a qualified attorney who can consider the particular circumstances of your case.",
            "hr": "⚠️ HR Disclaimer: This information is for general guidance only and should not be considered as specific HR or legal advice. Please consult with qualified HR professionals or legal counsel for decisions regarding specific situations."
        }
        return disclaimers.get(role, "")