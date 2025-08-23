#!/usr/bin/env python
"""Test script for AI provider functionality."""

import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.services.ai_providers import AIProviderFactory
from app.core.config import settings


async def test_providers():
    """Test all configured AI providers."""
    print("Testing AI Provider Configuration")
    print("=" * 50)
    
    # Test current provider
    print(f"\nCurrent LLM Provider: {settings.llm_provider}")
    print(f"Current Embedding Provider: {settings.embedding_provider}")
    
    # List all providers
    providers = AIProviderFactory.list_providers()
    print(f"\nAvailable providers: {providers}")
    
    # Test each provider
    for provider_name in providers:
        print(f"\n--- Testing {provider_name} provider ---")
        try:
            provider = AIProviderFactory.get_provider(provider_name)
            is_valid = provider.validate_config()
            print(f"Configuration valid: {is_valid}")
            
            if is_valid:
                # Test chat model
                try:
                    chat_model = provider.get_chat_model()
                    print(f"Chat model initialized: {type(chat_model).__name__}")
                except Exception as e:
                    print(f"Chat model error: {e}")
                
                # Test embeddings model
                try:
                    embeddings_model = provider.get_embeddings_model()
                    print(f"Embeddings model initialized: {type(embeddings_model).__name__}")
                except Exception as e:
                    print(f"Embeddings model error: {e}")
                    
        except Exception as e:
            print(f"Provider error: {e}")
    
    # Test actual generation with default provider
    print("\n--- Testing generation with default provider ---")
    try:
        provider = AIProviderFactory.get_provider()
        
        # Test chat
        chat_model = provider.get_chat_model()
        response = await chat_model.ainvoke("Say 'Hello, VerixAI is working!' in one sentence.")
        print(f"Chat response: {response.content}")
        
        # Test embeddings
        embeddings_model = provider.get_embeddings_model()
        embedding = await embeddings_model.aembed_query("test document")
        print(f"Embedding generated: {len(embedding)} dimensions")
        
    except Exception as e:
        print(f"Generation error: {e}")
        print("\nMake sure Ollama is running and models are downloaded:")
        print("  1. Start Ollama: ollama serve")
        print("  2. Download models: ./scripts/setup_ollama.sh")


if __name__ == "__main__":
    asyncio.run(test_providers())