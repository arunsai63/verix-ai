"""
Chat with Document Service

This module provides interactive chat functionality with documents including:
- Conversation memory and context management
- Multi-document chat capabilities
- CSV data analysis through natural language
- Session management and persistence
- Follow-up question handling
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import json
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
import redis
import pickle
import pandas as pd

from ..services.vector_store import VectorStoreService
from ..services.csv_analytics_service import CSVAnalyticsService, AnalyticsRequest
from ..services.ai_providers import get_llm_provider
from ..services.rag_service import RAGService
from ..core.config import settings

logger = logging.getLogger(__name__)


class MessageRole(str, Enum):
    """Message roles in conversation"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass
class ChatMessage:
    """Individual chat message"""
    role: MessageRole
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    citations: Optional[List[Dict[str, Any]]] = None
    analytics: Optional[Dict[str, Any]] = None


@dataclass
class ChatSession:
    """Chat session with conversation history"""
    session_id: str
    dataset_names: List[str]
    messages: List[ChatMessage] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    context_window: int = 10  # Number of previous messages to consider
    
    def add_message(self, message: ChatMessage):
        """Add message to conversation history"""
        self.messages.append(message)
        self.updated_at = datetime.now()
    
    def get_context(self, num_messages: Optional[int] = None) -> List[ChatMessage]:
        """Get recent conversation context"""
        num = num_messages or self.context_window
        return self.messages[-num:] if len(self.messages) > num else self.messages
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary"""
        return {
            "session_id": self.session_id,
            "dataset_names": self.dataset_names,
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat(),
                    "citations": msg.citations,
                    "analytics": msg.analytics
                }
                for msg in self.messages
            ],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata
        }


class ChatService:
    """Interactive chat service for documents"""
    
    def __init__(self):
        self.vector_store = VectorStoreService()
        self.csv_analytics = CSVAnalyticsService()
        self.llm_provider = get_llm_provider()
        self.rag_service = RAGService()
        
        # Initialize Redis for session storage if available
        try:
            self.redis_client = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                decode_responses=False
            )
            self.redis_client.ping()
            self.use_redis = True
            logger.info("Redis connected for chat session storage")
        except:
            self.redis_client = None
            self.use_redis = False
            self.sessions = {}  # In-memory fallback
            logger.info("Using in-memory storage for chat sessions")
    
    async def create_session(
        self,
        dataset_names: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> ChatSession:
        """Create new chat session"""
        session_id = str(uuid.uuid4())
        
        # Add system message with context
        system_message = ChatMessage(
            role=MessageRole.SYSTEM,
            content=f"You are an intelligent assistant helping analyze documents from datasets: {', '.join(dataset_names)}. "
                   f"You have access to document content, can perform data analysis on CSV files, and maintain conversation context. "
                   f"Always provide accurate, relevant responses based on the document content."
        )
        
        session = ChatSession(
            session_id=session_id,
            dataset_names=dataset_names,
            messages=[system_message],
            metadata=metadata or {}
        )
        
        # Store session
        await self._store_session(session)
        
        logger.info(f"Created chat session {session_id} for datasets: {dataset_names}")
        return session
    
    async def chat(
        self,
        session_id: str,
        message: str,
        stream: bool = False
    ) -> ChatMessage:
        """Process chat message and generate response"""
        
        # Retrieve session
        session = await self._get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        # Add user message
        user_message = ChatMessage(
            role=MessageRole.USER,
            content=message
        )
        session.add_message(user_message)
        
        try:
            # Determine if this is a data analysis query
            is_data_query = await self._is_data_analysis_query(message, session.dataset_names)
            
            if is_data_query:
                # Handle CSV/data analytics
                response = await self._handle_data_query(message, session)
            else:
                # Handle document chat
                response = await self._handle_document_query(message, session)
            
            # Add assistant response to session
            session.add_message(response)
            
            # Store updated session
            await self._store_session(session)
            
            return response
            
        except Exception as e:
            logger.error(f"Error in chat: {str(e)}")
            error_response = ChatMessage(
                role=MessageRole.ASSISTANT,
                content=f"I encountered an error processing your request: {str(e)}. Please try rephrasing your question."
            )
            session.add_message(error_response)
            await self._store_session(session)
            return error_response
    
    async def _handle_document_query(
        self,
        message: str,
        session: ChatSession
    ) -> ChatMessage:
        """Handle document-based queries"""
        
        # Build context from conversation history
        context = self._build_conversation_context(session)
        
        # Enhanced query with context
        enhanced_query = f"{context}\n\nCurrent question: {message}"
        
        # Search for relevant documents
        search_results_tuples = self.vector_store.search(
            query=enhanced_query,
            dataset_names=session.dataset_names,
            k=10
        )
        
        if not search_results_tuples:
            return ChatMessage(
                role=MessageRole.ASSISTANT,
                content="I couldn't find relevant information in the documents to answer your question. "
                       "Could you please rephrase or provide more context?"
            )
        
        # Convert tuples to dictionaries
        search_results = []
        for doc, score in search_results_tuples:
            search_results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": score
            })
        
        # Generate response with citations
        response_text = await self._generate_contextual_response(
            message,
            search_results,
            session.get_context(5)
        )
        
        # Extract citations
        citations = self._extract_citations_from_results(search_results[:3])
        
        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content=response_text,
            citations=citations,
            metadata={"search_results": len(search_results)}
        )
    
    async def _handle_data_query(
        self,
        message: str,
        session: ChatSession
    ) -> ChatMessage:
        """Handle CSV data analysis queries"""
        
        # Find CSV files in datasets
        csv_files = await self._find_csv_files(session.dataset_names)
        
        if not csv_files:
            return ChatMessage(
                role=MessageRole.ASSISTANT,
                content="No CSV files found in the specified datasets. This query requires data files for analysis."
            )
        
        # Load and analyze CSV data
        try:
            # Load first CSV file (extend to handle multiple files)
            df = pd.read_csv(csv_files[0])
            
            # Create analytics request
            analytics_request = AnalyticsRequest(
                data=df,
                query=message,
                visualize=True
            )
            
            # Perform analysis
            analytics_response = await self.csv_analytics.analyze_csv(analytics_request)
            
            # Format response
            response_text = self._format_analytics_response(analytics_response)
            
            return ChatMessage(
                role=MessageRole.ASSISTANT,
                content=response_text,
                analytics={
                    "results": analytics_response.analysis_results,
                    "visualizations": analytics_response.visualizations,
                    "statistics": analytics_response.statistics
                },
                metadata={
                    "data_shape": df.shape,
                    "processing_time": analytics_response.processing_time
                }
            )
            
        except Exception as e:
            logger.error(f"Error in data analysis: {str(e)}")
            return ChatMessage(
                role=MessageRole.ASSISTANT,
                content=f"I encountered an error analyzing the data: {str(e)}. "
                       f"Please ensure the data is properly formatted and try again."
            )
    
    async def _generate_contextual_response(
        self,
        query: str,
        search_results: List[Dict[str, Any]],
        conversation_history: List[ChatMessage]
    ) -> str:
        """Generate response considering conversation context"""
        
        # Format conversation history
        history_text = "\n".join([
            f"{msg.role}: {msg.content}"
            for msg in conversation_history[-3:]  # Last 3 messages
        ])
        
        # Format search results
        context_text = "\n\n".join([
            f"[Source {i+1}]: {result.get('content', '')}"
            for i, result in enumerate(search_results[:5])
        ])
        
        prompt = f"""
        Based on the conversation history and document context, answer the user's question.
        
        Conversation History:
        {history_text}
        
        Document Context:
        {context_text}
        
        Current Question: {query}
        
        Instructions:
        1. Consider the conversation history for context
        2. Use information from the documents to answer
        3. Be specific and cite sources when possible
        4. If following up on a previous topic, maintain continuity
        5. If you cannot answer based on the documents, say so clearly
        
        Response:
        """
        
        response = await self.llm_provider.generate(
            prompt=prompt,
            temperature=0.7,
            max_tokens=1000
        )
        
        return response
    
    def _build_conversation_context(self, session: ChatSession) -> str:
        """Build context from conversation history"""
        recent_messages = session.get_context(5)
        
        if len(recent_messages) <= 2:  # Only system message and current
            return ""
        
        context_parts = []
        for msg in recent_messages[1:-1]:  # Skip system and current message
            if msg.role == MessageRole.USER:
                context_parts.append(f"Previous question: {msg.content}")
            elif msg.role == MessageRole.ASSISTANT:
                # Include brief summary of previous answers
                summary = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
                context_parts.append(f"Previous answer summary: {summary}")
        
        return "Conversation context:\n" + "\n".join(context_parts)
    
    async def _is_data_analysis_query(self, message: str, dataset_names: List[str]) -> bool:
        """Determine if query is for data analysis"""
        
        # Keywords indicating data analysis
        data_keywords = [
            "average", "mean", "sum", "total", "count", "statistics",
            "correlation", "trend", "distribution", "compare", "analyze",
            "chart", "graph", "plot", "visualize", "show me",
            "group by", "filter", "aggregate", "calculate"
        ]
        
        message_lower = message.lower()
        
        # Check for data analysis keywords
        has_data_keyword = any(keyword in message_lower for keyword in data_keywords)
        
        if not has_data_keyword:
            return False
        
        # Check if datasets contain CSV files
        csv_files = await self._find_csv_files(dataset_names)
        return len(csv_files) > 0
    
    async def _find_csv_files(self, dataset_names: List[str]) -> List[str]:
        """Find CSV files in specified datasets"""
        csv_files = []
        
        for dataset_name in dataset_names:
            # This would typically check the actual file system or database
            # For now, return mock data
            import os
            dataset_path = f"datasets/{dataset_name}"
            if os.path.exists(dataset_path):
                for file in os.listdir(dataset_path):
                    if file.endswith('.csv'):
                        csv_files.append(os.path.join(dataset_path, file))
        
        return csv_files
    
    def _format_analytics_response(self, analytics_response) -> str:
        """Format analytics response for chat"""
        
        parts = [analytics_response.natural_language_summary]
        
        # Add key statistics if available
        if analytics_response.statistics:
            stats = analytics_response.statistics
            if "numeric_summary" in stats:
                parts.append("\nKey Statistics:")
                summary = stats["numeric_summary"].get("summary", {})
                for col, values in list(summary.items())[:3]:  # Top 3 columns
                    if isinstance(values, dict) and "mean" in values:
                        parts.append(f"- {col}: Mean={values['mean']:.2f}, Min={values['min']:.2f}, Max={values['max']:.2f}")
        
        # Mention visualizations
        if analytics_response.visualizations:
            parts.append(f"\nI've generated {len(analytics_response.visualizations)} visualization(s) to help illustrate the findings.")
        
        return "\n".join(parts)
    
    def _extract_citations_from_results(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract citations from search results"""
        citations = []
        
        for i, result in enumerate(search_results):
            citation = {
                "index": i + 1,
                "source": result.get("metadata", {}).get("source", "Unknown"),
                "page": result.get("metadata", {}).get("page"),
                "excerpt": result.get("content", "")[:200] + "...",
                "relevance_score": result.get("score", 0)
            }
            citations.append(citation)
        
        return citations
    
    async def get_session_history(self, session_id: str) -> Optional[ChatSession]:
        """Retrieve full session history"""
        return await self._get_session(session_id)
    
    async def list_sessions(
        self,
        dataset_name: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """List available chat sessions"""
        sessions = []
        
        if self.use_redis:
            # Get sessions from Redis
            pattern = f"chat_session:*"
            keys = self.redis_client.keys(pattern)
            
            for key in keys[:limit]:
                session_data = self.redis_client.get(key)
                if session_data:
                    session = pickle.loads(session_data)
                    if not dataset_name or dataset_name in session.dataset_names:
                        sessions.append({
                            "session_id": session.session_id,
                            "dataset_names": session.dataset_names,
                            "message_count": len(session.messages),
                            "created_at": session.created_at.isoformat(),
                            "updated_at": session.updated_at.isoformat()
                        })
        else:
            # Get from in-memory storage
            for session_id, session in list(self.sessions.items())[:limit]:
                if not dataset_name or dataset_name in session.dataset_names:
                    sessions.append({
                        "session_id": session.session_id,
                        "dataset_names": session.dataset_names,
                        "message_count": len(session.messages),
                        "created_at": session.created_at.isoformat(),
                        "updated_at": session.updated_at.isoformat()
                    })
        
        return sessions
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete a chat session"""
        if self.use_redis:
            key = f"chat_session:{session_id}"
            return bool(self.redis_client.delete(key))
        else:
            if session_id in self.sessions:
                del self.sessions[session_id]
                return True
            return False
    
    async def _store_session(self, session: ChatSession):
        """Store session in Redis or memory"""
        if self.use_redis:
            key = f"chat_session:{session.session_id}"
            # Set expiration to 24 hours
            self.redis_client.setex(
                key,
                timedelta(hours=24),
                pickle.dumps(session)
            )
        else:
            self.sessions[session.session_id] = session
    
    async def _get_session(self, session_id: str) -> Optional[ChatSession]:
        """Retrieve session from storage"""
        if self.use_redis:
            key = f"chat_session:{session_id}"
            session_data = self.redis_client.get(key)
            if session_data:
                return pickle.loads(session_data)
        else:
            return self.sessions.get(session_id)
        
        return None
    
    async def export_session(
        self,
        session_id: str,
        format: str = "json"
    ) -> Union[str, Dict[str, Any]]:
        """Export chat session in various formats"""
        
        session = await self._get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        if format == "json":
            return json.dumps(session.to_dict(), indent=2)
        
        elif format == "markdown":
            md_lines = [
                f"# Chat Session: {session.session_id}",
                f"**Datasets:** {', '.join(session.dataset_names)}",
                f"**Created:** {session.created_at.isoformat()}",
                f"**Updated:** {session.updated_at.isoformat()}",
                "",
                "## Conversation",
                ""
            ]
            
            for msg in session.messages:
                if msg.role != MessageRole.SYSTEM:
                    role_label = "User" if msg.role == MessageRole.USER else "Assistant"
                    md_lines.append(f"### {role_label}")
                    md_lines.append(msg.content)
                    
                    if msg.citations:
                        md_lines.append("\n**Sources:**")
                        for citation in msg.citations:
                            md_lines.append(f"- {citation['source']} (relevance: {citation['relevance_score']:.2f})")
                    
                    md_lines.append("")
            
            return "\n".join(md_lines)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")