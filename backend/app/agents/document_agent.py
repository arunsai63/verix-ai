import logging
from typing import Dict, Any, List, Optional
from app.services.document_processor import DocumentProcessor
from app.services.vector_store import VectorStoreService
from app.services.rag_service import RAGService

logger = logging.getLogger(__name__)


class DocumentIngestionAgent:
    """Agent responsible for document ingestion and processing."""
    
    def __init__(self):
        self.name = "DocumentIngestionAgent"
        self.description = "Handles document upload, conversion, and storage"
        self.processor = DocumentProcessor()
        self.vector_store = VectorStoreService()
    
    async def ingest_documents(
        self,
        file_paths: List[str],
        dataset_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Main task for ingesting documents into the system.
        
        Args:
            file_paths: List of file paths to ingest
            dataset_name: Name of the dataset
            metadata: Additional metadata
            
        Returns:
            Ingestion results
        """
        try:
            # Log task start
            logger.info(f"Starting document ingestion: {len(file_paths)} documents into {dataset_name}")
            
            processed_docs = self.processor.process_batch(
                file_paths, dataset_name, metadata
            )
            
            doc_ids = self.vector_store.add_documents(
                processed_docs, dataset_name
            )
            
            result = {
                "status": "success",
                "dataset_name": dataset_name,
                "documents_processed": len(processed_docs),
                "chunks_created": sum(len(doc["chunks"]) for doc in processed_docs),
                "document_ids": doc_ids
            }
            
            logger.info(f"Document ingestion completed: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Document ingestion failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }


class RetrievalAgent:
    """Agent responsible for document retrieval and search."""
    
    def __init__(self):
        self.name = "RetrievalAgent"
        self.description = "Handles document search and retrieval"
        self.vector_store = VectorStoreService()
    
    async def search_documents(
        self,
        query: str,
        dataset_names: Optional[List[str]] = None,
        k: int = 10
    ) -> Dict[str, Any]:
        """
        Search for relevant documents.
        
        Args:
            query: Search query
            dataset_names: Optional dataset filter
            k: Number of results
            
        Returns:
            Search results
        """
        try:
            # Log task start
            logger.info(f"Starting document search: {query[:50]}...")
            
            results = self.vector_store.hybrid_search(
                query, dataset_names, k=k
            )
            
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "content": result["content"][:500],
                    "score": result["score"],
                    "source": result["metadata"].get("filename"),
                    "dataset": result["metadata"].get("dataset_name"),
                    "chunk_index": result["metadata"].get("chunk_index"),
                    "full_content": result["content"],
                    "metadata": result["metadata"]
                })
            
            response = {
                "status": "success",
                "query": query,
                "results_count": len(formatted_results),
                "results": formatted_results
            }
            
            logger.info(f"Document search completed: {response['results_count']} results")
            return response
            
        except Exception as e:
            logger.error(f"Document search failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }


class SummarizationAgent:
    """Agent responsible for summarizing and formatting responses."""
    
    def __init__(self):
        self.name = "SummarizationAgent"
        self.description = "Handles response generation and summarization"
        self.rag_service = None
    
    def initialize_rag(self, rag_service):
        """Initialize with RAG service."""
        self.rag_service = rag_service
    
    async def generate_answer(
        self,
        query: str,
        search_results: List[Dict[str, Any]],
        role: str = "general"
    ) -> Dict[str, Any]:
        """
        Generate an answer with citations based on search results.
        
        Args:
            query: Original query
            search_results: Retrieved documents
            role: User role for formatting
            
        Returns:
            Generated answer with citations
        """
        try:
            if not self.rag_service:
                from app.services.rag_service import RAGService
                self.rag_service = RAGService()
            
            # Log task start
            logger.info(f"Starting answer generation for {role}: {query[:50]}...")
            
            answer = await self.rag_service.generate_answer(
                query, search_results, role
            )
            
            logger.info(f"Answer generation completed")
            return answer
            
        except Exception as e:
            logger.error(f"Answer generation failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }


class DocumentOrchestrator:
    """Orchestrates multiple agents for complete document processing pipeline."""
    
    def __init__(self):
        self.ingestion_agent = DocumentIngestionAgent()
        self.retrieval_agent = RetrievalAgent()
        self.summarization_agent = SummarizationAgent()
    
    async def process_query(
        self,
        query: str,
        dataset_names: Optional[List[str]] = None,
        role: str = "general",
        k: int = 10
    ) -> Dict[str, Any]:
        """
        Process a complete query through the pipeline.
        
        Args:
            query: User query
            dataset_names: Optional dataset filter
            role: User role
            k: Number of results to retrieve
            
        Returns:
            Complete response with answer and citations
        """
        try:
            search_results = await self.retrieval_agent.search_documents(
                query, dataset_names, k
            )
            
            if search_results["status"] != "success" or not search_results.get("results"):
                return {
                    "status": "no_results",
                    "message": "No relevant documents found for your query.",
                    "query": query
                }
            
            answer = await self.summarization_agent.generate_answer(
                query, search_results["results"], role
            )
            
            return {
                "status": "success",
                "query": query,
                "answer": answer.get("answer"),
                "citations": answer.get("citations"),
                "highlights": answer.get("highlights"),
                "confidence": answer.get("confidence"),
                "role": role,
                "sources_count": len(search_results["results"])
            }
            
        except Exception as e:
            logger.error(f"Query processing failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "query": query
            }
    
    async def ingest_dataset(
        self,
        file_paths: List[str],
        dataset_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Ingest a new dataset.
        
        Args:
            file_paths: List of file paths
            dataset_name: Dataset name
            metadata: Additional metadata
            
        Returns:
            Ingestion results
        """
        return await self.ingestion_agent.ingest_documents(
            file_paths, dataset_name, metadata
        )