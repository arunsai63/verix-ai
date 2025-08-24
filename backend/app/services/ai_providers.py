import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import json
import httpx
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
# from langchain_anthropic import ChatAnthropic
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import LLMChain
from langchain.embeddings.base import Embeddings
from app.core.config import settings

logger = logging.getLogger(__name__)


class CleanOllamaEmbeddings(Embeddings):
    """Custom Ollama embeddings that avoids passing invalid parameters."""
    
    def __init__(self, model: str, base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of documents."""
        embeddings = []
        
        with httpx.Client() as client:
            for text in texts:
                try:
                    response = client.post(
                        f"{self.base_url}/api/embeddings",
                        json={"model": self.model, "prompt": text},
                        timeout=60.0
                    )
                    response.raise_for_status()
                    
                    result = response.json()
                    if "embedding" in result:
                        embeddings.append(result["embedding"])
                    else:
                        logger.error(f"No embedding in response: {result}")
                        # Return zero vector as fallback
                        embeddings.append([0.0] * 768)  # Default dimension
                        
                except Exception as e:
                    logger.error(f"Error generating embedding for text: {str(e)}")
                    # Return zero vector as fallback
                    embeddings.append([0.0] * 768)
        
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query."""
        try:
            with httpx.Client() as client:
                response = client.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": self.model, "prompt": text},
                    timeout=60.0
                )
                response.raise_for_status()
                
                result = response.json()
                if "embedding" in result:
                    return result["embedding"]
                else:
                    logger.error(f"No embedding in response: {result}")
                    return [0.0] * 768  # Default dimension
                    
        except Exception as e:
            logger.error(f"Error generating query embedding: {str(e)}")
            return [0.0] * 768


class BaseLLMProvider(ABC):
    """Base class for LLM providers."""
    
    @abstractmethod
    def get_chat_model(self, **kwargs):
        """Get the chat model instance."""
        pass
    
    @abstractmethod
    def get_embeddings_model(self, **kwargs):
        """Get the embeddings model instance."""
        pass
    
    @abstractmethod
    def validate_config(self) -> bool:
        """Validate the provider configuration."""
        pass


class OllamaProvider(BaseLLMProvider):
    """Ollama provider for local LLM and embeddings."""
    
    def __init__(self):
        self.base_url = settings.ollama_base_url
        self.chat_model = settings.ollama_chat_model
        self.embedding_model = settings.ollama_embedding_model
    
    def get_chat_model(self, **kwargs):
        """Get Ollama chat model."""
        default_kwargs = {
            "model": self.chat_model,
            "base_url": self.base_url,
            "temperature": kwargs.get("temperature", 0.3),
            "num_predict": kwargs.get("max_tokens", 2000),
        }
        return ChatOllama(**default_kwargs)
    
    def get_embeddings_model(self, **kwargs):
        """Get Ollama embeddings model."""
        return CleanOllamaEmbeddings(
            model=self.embedding_model,
            base_url=self.base_url
        )
    
    def validate_config(self) -> bool:
        """Check if Ollama is running and accessible."""
        try:
            with httpx.Client() as client:
                response = client.get(f"{self.base_url}/api/tags")
                if response.status_code == 200:
                    models = response.json().get("models", [])
                    model_names = [m["name"] for m in models]
                    
                    # Check chat model (handle with/without tags)
                    chat_model_found = any(
                        name == self.chat_model or name.startswith(f"{self.chat_model}:")
                        for name in model_names
                    )
                    if not chat_model_found:
                        logger.warning(f"Chat model {self.chat_model} not found in Ollama. Available: {model_names}")
                        return False
                    
                    # Check embedding model (handle with/without tags)
                    embedding_model_found = any(
                        name == self.embedding_model or name.startswith(f"{self.embedding_model}:")
                        for name in model_names
                    )
                    if not embedding_model_found:
                        logger.warning(f"Embedding model {self.embedding_model} not found in Ollama. Available: {model_names}")
                        return False
                    
                    return True
                return False
        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {str(e)}")
            return False


class OpenAIProvider(BaseLLMProvider):
    """OpenAI provider for GPT models and embeddings."""
    
    def __init__(self):
        self.api_key = settings.openai_api_key
        self.chat_model = settings.openai_chat_model
        self.embedding_model = settings.openai_embedding_model
    
    def get_chat_model(self, **kwargs):
        """Get OpenAI chat model."""
        default_kwargs = {
            "api_key": self.api_key,
            "model": self.chat_model,
            "temperature": kwargs.get("temperature", 0.3),
            "max_tokens": kwargs.get("max_tokens", 2000),
        }
        return ChatOpenAI(**default_kwargs)
    
    def get_embeddings_model(self, **kwargs):
        """Get OpenAI embeddings model."""
        return OpenAIEmbeddings(
            api_key=self.api_key,
            model=self.embedding_model
        )
    
    def validate_config(self) -> bool:
        """Validate OpenAI API key."""
        if not self.api_key or self.api_key == "your-openai-api-key-here":
            logger.error("OpenAI API key not configured")
            return False
        return True


class ClaudeProvider(BaseLLMProvider):
    """Anthropic Claude provider."""
    
    def __init__(self):
        self.api_key = settings.claude_api_key
        self.chat_model = settings.claude_chat_model
    
    def get_chat_model(self, **kwargs):
        """Get Claude chat model."""
        default_kwargs = {
            "anthropic_api_key": self.api_key,
            "model": self.chat_model,
            "temperature": kwargs.get("temperature", 0.3),
            "max_tokens": kwargs.get("max_tokens", 2000),
        }
        return ChatAnthropic(**default_kwargs)
    
    def get_embeddings_model(self, **kwargs):
        """Claude doesn't provide embeddings, fallback to Ollama or OpenAI."""
        if settings.embedding_provider == "openai" and settings.openai_api_key:
            return OpenAIEmbeddings(
                api_key=settings.openai_api_key,
                model=settings.openai_embedding_model
            )
        else:
            return CleanOllamaEmbeddings(
                model=settings.ollama_embedding_model,
                base_url=settings.ollama_base_url
            )
    
    def validate_config(self) -> bool:
        """Validate Claude API key."""
        if not self.api_key or self.api_key == "your-claude-api-key-here":
            logger.error("Claude API key not configured")
            return False
        return True


class AIProviderFactory:
    """Factory class to get the appropriate AI provider."""
    
    _providers = {
        "ollama": OllamaProvider,
        "openai": OpenAIProvider,
        "claude": ClaudeProvider,
    }
    
    @classmethod
    def get_provider(cls, provider_name: Optional[str] = None) -> BaseLLMProvider:
        """
        Get AI provider instance.
        
        Args:
            provider_name: Name of the provider (ollama, openai, claude)
                         If None, uses the default from settings
        
        Returns:
            Provider instance
        """
        provider_name = provider_name or settings.llm_provider
        provider_name = provider_name.lower()
        
        if provider_name not in cls._providers:
            logger.warning(f"Unknown provider {provider_name}, falling back to ollama")
            provider_name = "ollama"
        
        provider_class = cls._providers[provider_name]
        provider = provider_class()
        
        if not provider.validate_config():
            logger.warning(f"Provider {provider_name} validation failed, falling back to ollama")
            if provider_name != "ollama":
                provider = OllamaProvider()
                if not provider.validate_config():
                    raise ValueError("No valid AI provider configuration found")
        
        logger.info(f"Using AI provider: {provider_name}")
        return provider
    
    @classmethod
    def list_providers(cls) -> List[str]:
        """List available providers."""
        return list(cls._providers.keys())