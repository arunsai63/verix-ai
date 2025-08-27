"""
API Routes Package

This package contains all API route modules organized by functionality.
"""

from .system import router as system_router
from .documents import router as documents_router
from .datasets import router as datasets_router
from .jobs import router as jobs_router

__all__ = [
    "system_router",
    "documents_router",
    "datasets_router",
    "jobs_router"
]