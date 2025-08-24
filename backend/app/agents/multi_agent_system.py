import logging
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
from app.services.vector_store import VectorStoreService
from app.services.rag_service import RAGService
from app.services.advanced_rag_service import AdvancedRAGService, RetrievalConfig
from app.services.async_document_processor import AsyncDocumentProcessor

logger = logging.getLogger(__name__)


class AgentRole(Enum):
    """Defines different agent roles in the system."""
    INGESTION = "ingestion"
    RETRIEVAL = "retrieval"
    RANKING = "ranking"
    SUMMARIZATION = "summarization"
    CITATION = "citation"
    VALIDATION = "validation"
    ORCHESTRATOR = "orchestrator"


@dataclass
class AgentMessage:
    """Message structure for inter-agent communication."""
    sender: str
    receiver: str
    content: Any
    message_type: str
    timestamp: str
    priority: int = 0


class BaseAgent:
    """Base class for all agents in the multi-agent system."""
    
    def __init__(self, name: str, role: AgentRole):
        self.name = name
        self.role = role
        self.message_queue = asyncio.Queue()
        self.active_tasks = []
        
    async def process_message(self, message: AgentMessage) -> Any:
        """Process incoming messages."""
        raise NotImplementedError
    
    async def execute_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the agent's primary task."""
        raise NotImplementedError
    
    async def send_message(self, receiver: str, content: Any, message_type: str = "info"):
        """Send message to another agent."""
        from datetime import datetime
        message = AgentMessage(
            sender=self.name,
            receiver=receiver,
            content=content,
            message_type=message_type,
            timestamp=datetime.utcnow().isoformat()
        )
        return message


class DocumentIngestionAgent(BaseAgent):
    """Agent responsible for parallel document ingestion and processing."""
    
    def __init__(self):
        super().__init__("DocumentIngestionAgent", AgentRole.INGESTION)
        self.processor = AsyncDocumentProcessor(max_workers=4)
        self.vector_store = VectorStoreService()
    
    async def execute_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ingest documents in parallel with progress tracking.
        
        Args:
            task_data: Contains file_paths, dataset_name, metadata
            
        Returns:
            Ingestion results
        """
        try:
            file_paths = task_data.get("file_paths", [])
            dataset_name = task_data.get("dataset_name")
            metadata = task_data.get("metadata", {})
            
            logger.info(f"{self.name}: Starting parallel ingestion of {len(file_paths)} documents")
            
            # Process documents in parallel batches
            processed_docs = await self.processor.process_batch_async(
                file_paths, 
                dataset_name, 
                metadata,
                batch_size=5
            )
            
            # Add to vector store
            doc_ids = self.vector_store.add_documents(
                processed_docs, 
                dataset_name
            )
            
            result = {
                "status": "success",
                "agent": self.name,
                "dataset_name": dataset_name,
                "documents_processed": len(processed_docs),
                "chunks_created": sum(len(doc["chunks"]) for doc in processed_docs),
                "document_ids": doc_ids
            }
            
            logger.info(f"{self.name}: Ingestion completed - {result}")
            return result
            
        except Exception as e:
            logger.error(f"{self.name}: Ingestion failed - {str(e)}")
            return {
                "status": "error",
                "agent": self.name,
                "error": str(e)
            }


class ParallelRetrievalAgent(BaseAgent):
    """Agent that performs parallel retrieval across multiple strategies."""
    
    def __init__(self):
        super().__init__("ParallelRetrievalAgent", AgentRole.RETRIEVAL)
        self.vector_store = VectorStoreService()
    
    async def execute_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform parallel retrieval using multiple strategies.
        
        Args:
            task_data: Contains query, dataset_names, k
            
        Returns:
            Combined retrieval results
        """
        try:
            query = task_data.get("query")
            dataset_names = task_data.get("dataset_names")
            k = task_data.get("k", 10)
            
            logger.info(f"{self.name}: Starting parallel retrieval for query: {query[:50]}...")
            
            # Execute multiple retrieval strategies in parallel
            tasks = [
                self._semantic_search(query, dataset_names, k),
                self._keyword_search(query, dataset_names, k),
                self._hybrid_search(query, dataset_names, k)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Combine and deduplicate results
            combined_results = []
            seen_chunks = set()
            
            for strategy_results in results:
                if isinstance(strategy_results, Exception):
                    logger.error(f"{self.name}: Strategy failed - {str(strategy_results)}")
                    continue
                    
                for result in strategy_results:
                    chunk_id = f"{result.get('metadata', {}).get('document_id')}_{result.get('metadata', {}).get('chunk_index')}"
                    if chunk_id not in seen_chunks:
                        seen_chunks.add(chunk_id)
                        combined_results.append(result)
            
            return {
                "status": "success",
                "agent": self.name,
                "query": query,
                "results_count": len(combined_results),
                "results": combined_results
            }
            
        except Exception as e:
            logger.error(f"{self.name}: Retrieval failed - {str(e)}")
            return {
                "status": "error",
                "agent": self.name,
                "error": str(e)
            }
    
    async def _semantic_search(self, query: str, dataset_names: Optional[List[str]], k: int) -> List[Dict[str, Any]]:
        """Perform semantic similarity search."""
        results = self.vector_store.search(query, dataset_names, k)
        formatted_results = []
        
        for doc, score in results:
            formatted_results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": score,
                "strategy": "semantic"
            })
        
        return formatted_results
    
    async def _keyword_search(self, query: str, dataset_names: Optional[List[str]], k: int) -> List[Dict[str, Any]]:
        """Perform keyword-based search."""
        # Simple keyword matching implementation
        results = []
        semantic_results = self.vector_store.search(query, dataset_names, k * 2)
        
        keywords = query.lower().split()
        
        for doc, _ in semantic_results:
            content_lower = doc.page_content.lower()
            keyword_score = sum(1 for keyword in keywords if keyword in content_lower) / len(keywords)
            
            if keyword_score > 0:
                results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": keyword_score,
                    "strategy": "keyword"
                })
        
        return sorted(results, key=lambda x: x["score"], reverse=True)[:k]
    
    async def _hybrid_search(self, query: str, dataset_names: Optional[List[str]], k: int) -> List[Dict[str, Any]]:
        """Perform hybrid search combining multiple signals."""
        results = self.vector_store.hybrid_search(query, dataset_names, k)
        
        for result in results:
            result["strategy"] = "hybrid"
        
        return results


class RankingAgent(BaseAgent):
    """Agent responsible for re-ranking search results."""
    
    def __init__(self):
        super().__init__("RankingAgent", AgentRole.RANKING)
    
    async def execute_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Re-rank search results based on multiple factors.
        
        Args:
            task_data: Contains query, results
            
        Returns:
            Re-ranked results
        """
        try:
            query = task_data.get("query")
            results = task_data.get("results", [])
            
            logger.info(f"{self.name}: Re-ranking {len(results)} results")
            
            # Calculate composite scores
            ranked_results = []
            
            for result in results:
                # Calculate relevance factors
                semantic_score = result.get("score", 0)
                keyword_score = self._calculate_keyword_relevance(query, result["content"])
                position_score = 1.0 / (results.index(result) + 1)  # Higher score for earlier positions
                freshness_score = self._calculate_freshness(result.get("metadata", {}))
                
                # Weighted combination
                composite_score = (
                    semantic_score * 0.4 +
                    keyword_score * 0.3 +
                    position_score * 0.2 +
                    freshness_score * 0.1
                )
                
                result["composite_score"] = composite_score
                result["ranking_details"] = {
                    "semantic": semantic_score,
                    "keyword": keyword_score,
                    "position": position_score,
                    "freshness": freshness_score
                }
                
                ranked_results.append(result)
            
            # Sort by composite score
            ranked_results.sort(key=lambda x: x["composite_score"], reverse=True)
            
            return {
                "status": "success",
                "agent": self.name,
                "ranked_results": ranked_results[:10]  # Return top 10
            }
            
        except Exception as e:
            logger.error(f"{self.name}: Ranking failed - {str(e)}")
            return {
                "status": "error",
                "agent": self.name,
                "error": str(e)
            }
    
    def _calculate_keyword_relevance(self, query: str, content: str) -> float:
        """Calculate keyword relevance score."""
        query_terms = query.lower().split()
        content_lower = content.lower()
        
        if not query_terms:
            return 0.0
        
        matches = sum(1 for term in query_terms if term in content_lower)
        return matches / len(query_terms)
    
    def _calculate_freshness(self, metadata: Dict[str, Any]) -> float:
        """Calculate document freshness score."""
        from datetime import datetime
        
        processed_at = metadata.get("processed_at")
        if not processed_at:
            return 0.5
        
        try:
            processed_date = datetime.fromisoformat(processed_at)
            days_old = (datetime.utcnow() - processed_date).days
            
            # Exponential decay based on age
            return max(0, 1 - (days_old / 365))
        except:
            return 0.5


class SummarizationAgent(BaseAgent):
    """Agent responsible for generating summaries and answers using advanced RAG."""
    
    def __init__(self, use_advanced: bool = True):
        super().__init__("SummarizationAgent", AgentRole.SUMMARIZATION)
        
        # Configure advanced RAG for maximum accuracy
        if use_advanced:
            config = RetrievalConfig(
                use_hybrid_search=True,
                use_query_expansion=True,
                use_multi_query=True,
                use_reranking=True,
                use_contextual_compression=True,
                max_context_length=15000,  # Larger context for better accuracy
                top_k_initial=30,  # Get more initial results
                top_k_reranked=15,  # Keep more after reranking
                min_relevance_score=0.2  # Lower threshold to include more context
            )
            self.rag_service = AdvancedRAGService(config=config)
            logger.info("Using Advanced RAG Service with full accuracy features")
        else:
            self.rag_service = RAGService()
            logger.info("Using Standard RAG Service")
    
    async def execute_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate answer with citations using advanced RAG techniques.
        
        Args:
            task_data: Contains query, results, role, dataset_names
            
        Returns:
            Generated answer with enhanced metadata
        """
        try:
            query = task_data.get("query")
            results = task_data.get("results", [])
            role = task_data.get("role", "general")
            dataset_names = task_data.get("dataset_names")
            
            logger.info(f"{self.name}: Generating answer using Advanced RAG for role: {role}")
            
            # Format results for RAG service
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "content": result.get("content"),
                    "metadata": result.get("metadata"),
                    "score": result.get("composite_score", result.get("score", 0)),
                    "full_content": result.get("content")
                })
            
            # Use advanced RAG if available
            if isinstance(self.rag_service, AdvancedRAGService):
                answer = await self.rag_service.generate_answer(
                    query, 
                    formatted_results, 
                    role,
                    dataset_names
                )
            else:
                answer = await self.rag_service.generate_answer(
                    query, 
                    formatted_results, 
                    role
                )
            
            return {
                "status": "success",
                "agent": self.name,
                "answer": answer
            }
            
        except Exception as e:
            logger.error(f"{self.name}: Summarization failed - {str(e)}")
            return {
                "status": "error",
                "agent": self.name,
                "error": str(e)
            }


class CitationAgent(BaseAgent):
    """Agent responsible for extracting and validating citations."""
    
    def __init__(self):
        super().__init__("CitationAgent", AgentRole.CITATION)
    
    async def execute_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract and validate citations from answer.
        
        Args:
            task_data: Contains answer, results
            
        Returns:
            Enhanced citations
        """
        try:
            answer = task_data.get("answer", {})
            results = task_data.get("results", [])
            
            logger.info(f"{self.name}: Extracting and validating citations")
            
            # Extract citations from answer
            citations = answer.get("citations", [])
            
            # Enhance citations with additional metadata
            enhanced_citations = []
            
            for citation in citations:
                source_num = citation.get("source_number")
                if source_num and source_num <= len(results):
                    result = results[source_num - 1]
                    
                    enhanced_citation = {
                        **citation,
                        "full_metadata": result.get("metadata", {}),
                        "relevance_score": result.get("composite_score", result.get("score", 0)),
                        "chunk_content": result.get("content", "")[:500],
                        "validated": True
                    }
                    
                    enhanced_citations.append(enhanced_citation)
            
            return {
                "status": "success",
                "agent": self.name,
                "citations": enhanced_citations
            }
            
        except Exception as e:
            logger.error(f"{self.name}: Citation extraction failed - {str(e)}")
            return {
                "status": "error",
                "agent": self.name,
                "error": str(e)
            }


class ValidationAgent(BaseAgent):
    """Agent responsible for validating the complete response."""
    
    def __init__(self):
        super().__init__("ValidationAgent", AgentRole.VALIDATION)
    
    async def execute_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate the complete response for quality and accuracy.
        
        Args:
            task_data: Contains answer, citations, query
            
        Returns:
            Validation results
        """
        try:
            answer = task_data.get("answer", {})
            citations = task_data.get("citations", [])
            query = task_data.get("query")
            
            logger.info(f"{self.name}: Validating response quality")
            
            validation_results = {
                "has_answer": bool(answer.get("answer")),
                "has_citations": len(citations) > 0,
                "citation_coverage": len(citations) / max(1, len(answer.get("answer", "").split("[Source"))) if answer.get("answer") else 0,
                "confidence_score": answer.get("confidence", "medium"),
                "query_addressed": self._check_query_addressed(query, answer.get("answer", ""))
            }
            
            # Calculate overall quality score
            quality_score = sum([
                validation_results["has_answer"] * 0.3,
                validation_results["has_citations"] * 0.2,
                validation_results["citation_coverage"] * 0.2,
                (1.0 if validation_results["confidence_score"] == "high" else 0.5) * 0.2,
                validation_results["query_addressed"] * 0.1
            ])
            
            validation_results["quality_score"] = quality_score
            validation_results["passed"] = quality_score > 0.6
            
            return {
                "status": "success",
                "agent": self.name,
                "validation": validation_results
            }
            
        except Exception as e:
            logger.error(f"{self.name}: Validation failed - {str(e)}")
            return {
                "status": "error",
                "agent": self.name,
                "error": str(e)
            }
    
    def _check_query_addressed(self, query: str, answer: str) -> float:
        """Check if the query was adequately addressed."""
        if not query or not answer:
            return 0.0
        
        # Simple check: presence of query keywords in answer
        query_terms = set(query.lower().split())
        answer_terms = set(answer.lower().split())
        
        if not query_terms:
            return 1.0
        
        overlap = len(query_terms.intersection(answer_terms))
        return overlap / len(query_terms)


class MultiAgentOrchestrator:
    """Orchestrates the multi-agent system for document processing and querying."""
    
    def __init__(self):
        self.agents = {
            AgentRole.INGESTION: DocumentIngestionAgent(),
            AgentRole.RETRIEVAL: ParallelRetrievalAgent(),
            AgentRole.RANKING: RankingAgent(),
            AgentRole.SUMMARIZATION: SummarizationAgent(),
            AgentRole.CITATION: CitationAgent(),
            AgentRole.VALIDATION: ValidationAgent()
        }
        self.execution_history = []
    
    async def process_query_multi_agent(
        self,
        query: str,
        dataset_names: Optional[List[str]] = None,
        role: str = "general",
        k: int = 10
    ) -> Dict[str, Any]:
        """
        Process a query using multiple specialized agents in parallel where possible.
        
        Args:
            query: User query
            dataset_names: Optional dataset filter
            role: User role
            k: Number of results to retrieve
            
        Returns:
            Complete response with answer, citations, and execution details
        """
        try:
            logger.info(f"Orchestrator: Starting multi-agent query processing")
            
            # Phase 1: Parallel Retrieval
            retrieval_task = {
                "query": query,
                "dataset_names": dataset_names,
                "k": k * 2  # Retrieve more for ranking
            }
            
            retrieval_result = await self.agents[AgentRole.RETRIEVAL].execute_task(retrieval_task)
            
            if retrieval_result["status"] != "success" or not retrieval_result.get("results"):
                return {
                    "status": "no_results",
                    "message": "No relevant documents found for your query.",
                    "query": query,
                    "execution_details": {"retrieval": retrieval_result}
                }
            
            # Phase 2: Parallel Ranking and Initial Summarization
            ranking_task = {
                "query": query,
                "results": retrieval_result["results"]
            }
            
            ranking_future = asyncio.create_task(
                self.agents[AgentRole.RANKING].execute_task(ranking_task)
            )
            
            # Wait for ranking to complete
            ranking_result = await ranking_future
            
            # Phase 3: Generate Answer with top-ranked results
            summarization_task = {
                "query": query,
                "results": ranking_result.get("ranked_results", retrieval_result["results"][:k]),
                "role": role,
                "dataset_names": dataset_names  # Pass dataset names for advanced RAG
            }
            
            answer_result = await self.agents[AgentRole.SUMMARIZATION].execute_task(summarization_task)
            
            # Phase 4: Parallel Citation and Validation
            citation_task = {
                "answer": answer_result.get("answer", {}),
                "results": ranking_result.get("ranked_results", retrieval_result["results"][:k])
            }
            
            validation_task = {
                "answer": answer_result.get("answer", {}),
                "citations": answer_result.get("answer", {}).get("citations", []),
                "query": query
            }
            
            # Execute citation and validation in parallel
            citation_future = asyncio.create_task(
                self.agents[AgentRole.CITATION].execute_task(citation_task)
            )
            validation_future = asyncio.create_task(
                self.agents[AgentRole.VALIDATION].execute_task(validation_task)
            )
            
            citation_result = await citation_future
            validation_result = await validation_future
            
            # Compile final response
            final_response = {
                "status": "success",
                "query": query,
                "answer": answer_result.get("answer", {}).get("answer", ""),
                "citations": citation_result.get("citations", []),
                "highlights": answer_result.get("answer", {}).get("highlights", []),
                "confidence": answer_result.get("answer", {}).get("confidence", "medium"),
                "role": role,
                "sources_count": len(ranking_result.get("ranked_results", [])),
                "validation": validation_result.get("validation", {}),
                "execution_details": {
                    "retrieval": {
                        "agent": "ParallelRetrievalAgent",
                        "results_count": retrieval_result.get("results_count", 0)
                    },
                    "ranking": {
                        "agent": "RankingAgent",
                        "ranked_count": len(ranking_result.get("ranked_results", []))
                    },
                    "summarization": {
                        "agent": "SummarizationAgent",
                        "status": answer_result.get("status")
                    },
                    "citation": {
                        "agent": "CitationAgent",
                        "citations_count": len(citation_result.get("citations", []))
                    },
                    "validation": {
                        "agent": "ValidationAgent",
                        "quality_score": validation_result.get("validation", {}).get("quality_score", 0)
                    }
                }
            }
            
            # Store execution history
            self.execution_history.append({
                "query": query,
                "timestamp": datetime.utcnow().isoformat(),
                "agents_used": list(self.agents.keys()),
                "success": True
            })
            
            logger.info(f"Orchestrator: Query processing completed successfully")
            return final_response
            
        except Exception as e:
            logger.error(f"Orchestrator: Query processing failed - {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "query": query
            }
    
    async def ingest_documents_multi_agent(
        self,
        file_paths: List[str],
        dataset_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Ingest documents using the ingestion agent.
        
        Args:
            file_paths: List of file paths
            dataset_name: Dataset name
            metadata: Additional metadata
            
        Returns:
            Ingestion results
        """
        try:
            logger.info(f"Orchestrator: Starting document ingestion")
            
            ingestion_task = {
                "file_paths": file_paths,
                "dataset_name": dataset_name,
                "metadata": metadata
            }
            
            result = await self.agents[AgentRole.INGESTION].execute_task(ingestion_task)
            
            # Store execution history
            self.execution_history.append({
                "operation": "ingestion",
                "dataset_name": dataset_name,
                "files_count": len(file_paths),
                "timestamp": datetime.utcnow().isoformat(),
                "success": result.get("status") == "success"
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Orchestrator: Document ingestion failed - {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Get the execution history of the orchestrator."""
        return self.execution_history


# Import datetime for timestamps
from datetime import datetime