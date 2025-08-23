import os
from typing import List, Dict, Any, Optional, Tuple
import logging
from pathlib import Path
import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from app.core.config import settings
from app.services.ai_providers import AIProviderFactory

logger = logging.getLogger(__name__)


class VectorStoreService:
    """Manages vector storage and retrieval using ChromaDB and LangChain."""
    
    def __init__(self):
        provider = AIProviderFactory.get_provider(settings.embedding_provider)
        self.embeddings = provider.get_embeddings_model()
        
        persist_directory = Path(settings.chroma_persist_directory)
        persist_directory.mkdir(parents=True, exist_ok=True)
        
        self.chroma_client = chromadb.PersistentClient(
            path=str(persist_directory),
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        self.collection_name = settings.chroma_collection_name
        self.vector_store = None
        self._initialize_vector_store()
    
    def _initialize_vector_store(self):
        """Initialize or load existing vector store."""
        try:
            self.vector_store = Chroma(
                client=self.chroma_client,
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=settings.chroma_persist_directory
            )
            logger.info(f"Vector store initialized with collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            raise
    
    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        dataset_name: str
    ) -> List[str]:
        """
        Add processed documents to the vector store.
        
        Args:
            documents: List of processed documents with chunks
            dataset_name: Name of the dataset
            
        Returns:
            List of document IDs added to the store
        """
        try:
            langchain_docs = []
            doc_ids = []
            
            for doc in documents:
                for chunk in doc.get("chunks", []):
                    metadata = {
                        **chunk["metadata"],
                        "dataset_name": dataset_name,
                        "document_id": doc["metadata"]["file_hash"]
                    }
                    
                    langchain_doc = Document(
                        page_content=chunk["content"],
                        metadata=metadata
                    )
                    langchain_docs.append(langchain_doc)
            
            if langchain_docs:
                ids = self.vector_store.add_documents(langchain_docs)
                doc_ids.extend(ids)
                logger.info(f"Added {len(langchain_docs)} chunks to vector store")
            
            return doc_ids
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            raise
    
    def search(
        self,
        query: str,
        dataset_names: Optional[List[str]] = None,
        k: int = 10,
        score_threshold: float = 0.3
    ) -> List[Tuple[Document, float]]:
        """
        Search for relevant documents based on query.
        
        Args:
            query: Search query
            dataset_names: Optional list of dataset names to search within
            k: Number of results to return
            score_threshold: Minimum relevance score threshold
            
        Returns:
            List of (Document, relevance_score) tuples
        """
        try:
            filter_dict = {}
            if dataset_names:
                filter_dict["dataset_name"] = {"$in": dataset_names}
            
            # Try without score threshold first to debug
            results = self.vector_store.similarity_search_with_relevance_scores(
                query,
                k=k,
                filter=filter_dict if filter_dict else None
            )
            
            logger.info(f"Search returned {len(results)} results for query: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Error searching vector store: {str(e)}")
            raise
    
    def hybrid_search(
        self,
        query: str,
        dataset_names: Optional[List[str]] = None,
        k: int = 10,
        rerank: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining semantic and keyword matching.
        
        Args:
            query: Search query
            dataset_names: Optional list of dataset names to search within
            k: Number of results to return
            rerank: Whether to rerank results
            
        Returns:
            List of search results with metadata and scores
        """
        try:
            semantic_results = self.search(query, dataset_names, k=k*2)
            
            formatted_results = []
            seen_chunks = set()
            
            for doc, score in semantic_results:
                chunk_id = f"{doc.metadata.get('document_id')}_{doc.metadata.get('chunk_index')}"
                if chunk_id not in seen_chunks:
                    seen_chunks.add(chunk_id)
                    
                    result = {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": score,
                        "search_type": "semantic"
                    }
                    
                    if query.lower() in doc.page_content.lower():
                        result["score"] *= 1.2
                        result["search_type"] = "hybrid"
                    
                    formatted_results.append(result)
            
            if rerank:
                formatted_results.sort(key=lambda x: x["score"], reverse=True)
            
            return formatted_results[:k]
            
        except Exception as e:
            logger.error(f"Error performing hybrid search: {str(e)}")
            raise
    
    def get_document_by_id(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific document by its ID.
        
        Args:
            document_id: The document hash/ID
            
        Returns:
            Document data if found, None otherwise
        """
        try:
            results = self.vector_store.get(
                where={"document_id": document_id}
            )
            
            if results and results["documents"]:
                return {
                    "content": results["documents"][0],
                    "metadata": results["metadatas"][0]
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving document {document_id}: {str(e)}")
            return None
    
    def delete_dataset(self, dataset_name: str) -> bool:
        """
        Delete all documents from a specific dataset.
        
        Args:
            dataset_name: Name of the dataset to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.vector_store.delete(
                where={"dataset_name": dataset_name}
            )
            logger.info(f"Deleted dataset: {dataset_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting dataset {dataset_name}: {str(e)}")
            return False
    
    def get_dataset_stats(self, dataset_name: str) -> Dict[str, Any]:
        """
        Get statistics for a specific dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary with dataset statistics
        """
        try:
            results = self.vector_store.get(
                where={"dataset_name": dataset_name}
            )
            
            if not results or not results["documents"]:
                return {
                    "dataset_name": dataset_name,
                    "total_chunks": 0,
                    "total_documents": 0
                }
            
            unique_docs = set()
            for metadata in results["metadatas"]:
                unique_docs.add(metadata.get("document_id"))
            
            return {
                "dataset_name": dataset_name,
                "total_chunks": len(results["documents"]),
                "total_documents": len(unique_docs),
                "file_types": list(set(m.get("file_extension") for m in results["metadatas"]))
            }
            
        except Exception as e:
            logger.error(f"Error getting dataset stats: {str(e)}")
            return {
                "dataset_name": dataset_name,
                "total_chunks": 0,
                "total_documents": 0,
                "error": str(e)
            }