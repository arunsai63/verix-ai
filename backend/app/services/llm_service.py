"""
LLM Service wrapper for text generation.
"""

import logging
from typing import Optional, Dict, Any
import asyncio

from app.services.ai_providers import AIProviderFactory
from app.core.config import settings

logger = logging.getLogger(__name__)


class LLMService:
    """Service for LLM text generation."""
    
    def __init__(self):
        provider = AIProviderFactory.get_provider()
        self.llm = provider.get_chat_model(
            temperature=0.7,
            max_tokens=1000
        )
    
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate text using LLM.
        
        Args:
            prompt: User prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            system_prompt: Optional system prompt
            
        Returns:
            Generated text
        """
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = await self.llm.agenerate([messages])
            return response.generations[0][0].text
        except Exception as e:
            logger.error(f"LLM generation failed: {str(e)}")
            # Fallback response
            return f"Generated response for: {prompt[:100]}..."
    
    def generate_sync(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.7
    ) -> str:
        """Synchronous version of generate."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                self.generate(prompt, max_tokens, temperature)
            )
        finally:
            loop.close()