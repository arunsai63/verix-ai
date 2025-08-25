"""
Chat API Routes
"""

import logging
from typing import Optional
from fastapi import APIRouter, HTTPException

from app.schemas.requests import ChatSessionRequest, ChatMessageRequest
from app.schemas.responses import (
    ChatSessionResponse, 
    ChatMessageResponse,
    ChatHistoryResponse
)
from app.services.chat_service import ChatService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/chat", tags=["chat"])

# Initialize service
chat_service = ChatService()


@router.post("/session", response_model=ChatSessionResponse)
async def create_chat_session(request: ChatSessionRequest):
    """
    Create a new chat session for document interaction.
    
    Args:
        request: Chat session request with dataset names
    """
    try:
        session = await chat_service.create_session(
            dataset_names=request.dataset_names,
            metadata=request.metadata
        )
        
        return ChatSessionResponse(
            status="success",
            session_id=session.session_id,
            dataset_names=session.dataset_names,
            created_at=session.created_at.isoformat()
        )
        
    except Exception as e:
        logger.error(f"Chat session creation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/message", response_model=ChatMessageResponse)
async def send_chat_message(request: ChatMessageRequest):
    """
    Send a message in a chat session and get response.
    
    Args:
        request: Chat message request
    """
    try:
        response = await chat_service.chat(
            session_id=request.session_id,
            message=request.message,
            stream=request.stream
        )
        
        return ChatMessageResponse(
            status="success",
            session_id=request.session_id,
            message=response.content,
            citations=response.citations,
            analytics=response.analytics,
            metadata=response.metadata,
            timestamp=response.timestamp.isoformat()
        )
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Chat message error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/session/{session_id}", response_model=ChatHistoryResponse)
async def get_chat_history(session_id: str):
    """Get full chat history for a session."""
    try:
        session = await chat_service.get_session_history(session_id)
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return ChatHistoryResponse(
            status="success",
            session_id=session.session_id,
            dataset_names=session.dataset_names,
            messages=[
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat(),
                    "citations": msg.citations,
                    "analytics": msg.analytics
                }
                for msg in session.messages
            ],
            created_at=session.created_at.isoformat(),
            updated_at=session.updated_at.isoformat(),
            total_messages=len(session.messages)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting chat history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions")
async def list_chat_sessions(
    dataset_name: Optional[str] = None,
    limit: int = 10
):
    """List available chat sessions."""
    try:
        sessions = await chat_service.list_sessions(
            dataset_name=dataset_name,
            limit=limit
        )
        return {
            "status": "success",
            "sessions": sessions,
            "total": len(sessions)
        }
        
    except Exception as e:
        logger.error(f"Error listing sessions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/session/{session_id}")
async def delete_chat_session(session_id: str):
    """Delete a chat session."""
    try:
        success = await chat_service.delete_session(session_id)
        
        if success:
            return {"status": "success", "message": f"Session {session_id} deleted"}
        else:
            raise HTTPException(status_code=404, detail="Session not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/session/{session_id}/export")
async def export_chat_session(
    session_id: str,
    format: str = "json"
):
    """Export chat session in various formats."""
    try:
        if format not in ["json", "markdown"]:
            raise HTTPException(status_code=400, detail="Format must be 'json' or 'markdown'")
        
        export_data = await chat_service.export_session(session_id, format)
        
        return {
            "status": "success",
            "format": format,
            "data": export_data
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error exporting session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))