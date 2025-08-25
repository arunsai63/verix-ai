"""
API Routes Package

This package contains all API route modules organized by functionality.
"""

from .system import router as system_router
from .documents import router as documents_router
from .datasets import router as datasets_router
from .jobs import router as jobs_router
from .summarization import router as summarization_router
from .chat import router as chat_router
from .analytics import router as analytics_router

__all__ = [
    "system_router",
    "documents_router",
    "datasets_router",
    "jobs_router",
    "summarization_router",
    "chat_router",
    "analytics_router"
]