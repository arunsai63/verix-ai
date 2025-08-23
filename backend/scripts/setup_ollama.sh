#!/bin/bash

echo "Setting up Ollama models for VerixAI..."

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "Error: Ollama is not running. Please start Ollama first with 'ollama serve' or 'docker-compose up ollama'"
    exit 1
fi

echo "Ollama is running. Downloading required models..."

# Download chat model
echo "Pulling chat model: llama3.2..."
ollama pull llama3.2

# Download embedding model
echo "Pulling embedding model: nomic-embed-text..."
ollama pull nomic-embed-text

# List installed models
echo ""
echo "Installed models:"
ollama list

echo ""
echo "Setup complete! You can now use Ollama with VerixAI."
echo ""
echo "To use different models, update these environment variables:"
echo "  OLLAMA_CHAT_MODEL=<model-name>"
echo "  OLLAMA_EMBEDDING_MODEL=<model-name>"
echo ""
echo "Popular alternatives:"
echo "  Chat: llama3.1, mistral, mixtral, qwen2.5"
echo "  Embeddings: mxbai-embed-large, all-minilm"