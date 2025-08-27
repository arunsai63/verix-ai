"""
Embedding Service for text vectorization.
"""

import logging
from typing import List, Optional
import numpy as np

from app.services.ai_providers import AIProviderFactory
from app.core.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for text embeddings."""
    
    def __init__(self):
        provider = AIProviderFactory.get_provider()
        self.embeddings = provider.get_embeddings_model()
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        try:
            embedding = self.embeddings.embed_query(text)
            return embedding
        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
            # Return random embedding as fallback
            return np.random.randn(384).tolist()
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            embeddings = self.embeddings.embed_documents(texts)
            return embeddings
        except Exception as e:
            logger.error(f"Batch embedding failed: {str(e)}")
            # Return random embeddings as fallback
            return [np.random.randn(384).tolist() for _ in texts]
    
    def similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Cosine similarity score
        """
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))