from fastapi import APIRouter, HTTPException, Body
from typing import Dict, Any, Optional
from pydantic import BaseModel
from app.services.ai_providers import AIProviderFactory
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/providers", tags=["AI Providers"])


class ProviderStatus(BaseModel):
    name: str
    available: bool
    is_active: bool
    error: Optional[str] = None


class ProviderSwitch(BaseModel):
    llm_provider: Optional[str] = None
    embedding_provider: Optional[str] = None


@router.get("/status", response_model=Dict[str, Any])
async def get_providers_status():
    """Get the status of all available AI providers."""
    providers_status = []
    
    for provider_name in AIProviderFactory.list_providers():
        try:
            provider = AIProviderFactory.get_provider(provider_name)
            available = provider.validate_config()
            
            providers_status.append(ProviderStatus(
                name=provider_name,
                available=available,
                is_active=(provider_name == settings.llm_provider),
                error=None if available else f"{provider_name} configuration not valid"
            ))
        except Exception as e:
            providers_status.append(ProviderStatus(
                name=provider_name,
                available=False,
                is_active=False,
                error=str(e)
            ))
    
    return {
        "current_llm_provider": settings.llm_provider,
        "current_embedding_provider": settings.embedding_provider,
        "providers": [p.dict() for p in providers_status]
    }


@router.post("/switch")
async def switch_provider(provider_config: ProviderSwitch):
    """
    Switch to a different AI provider dynamically.
    Note: This change is temporary and will reset on restart.
    Update .env file for permanent changes.
    """
    try:
        if provider_config.llm_provider:
            # Validate the provider exists and is configured
            provider = AIProviderFactory.get_provider(provider_config.llm_provider)
            if not provider.validate_config():
                raise HTTPException(
                    status_code=400,
                    detail=f"Provider {provider_config.llm_provider} is not properly configured"
                )
            settings.llm_provider = provider_config.llm_provider
            
        if provider_config.embedding_provider:
            # Validate the embedding provider
            provider = AIProviderFactory.get_provider(provider_config.embedding_provider)
            if not provider.validate_config():
                raise HTTPException(
                    status_code=400,
                    detail=f"Provider {provider_config.embedding_provider} is not properly configured"
                )
            settings.embedding_provider = provider_config.embedding_provider
        
        return {
            "message": "Provider switched successfully",
            "llm_provider": settings.llm_provider,
            "embedding_provider": settings.embedding_provider
        }
        
    except Exception as e:
        logger.error(f"Error switching provider: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{provider_name}")
async def get_provider_models(provider_name: str):
    """Get available models for a specific provider."""
    try:
        provider = AIProviderFactory.get_provider(provider_name)
        
        if provider_name == "ollama":
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{settings.ollama_base_url}/api/tags")
                if response.status_code == 200:
                    models = response.json().get("models", [])
                    return {
                        "provider": provider_name,
                        "models": [m["name"] for m in models]
                    }
        elif provider_name == "openai":
            return {
                "provider": provider_name,
                "models": ["gpt-4-turbo-preview", "gpt-4", "gpt-3.5-turbo"],
                "embedding_models": ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"]
            }
        elif provider_name == "claude":
            return {
                "provider": provider_name,
                "models": ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"]
            }
        
        return {"provider": provider_name, "models": []}
        
    except Exception as e:
        logger.error(f"Error getting models for {provider_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))