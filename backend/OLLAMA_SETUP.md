# Ollama Setup Guide for VerixAI

## Installation

### macOS
```bash
brew install ollama
```

### Linux
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### Windows
Download from: https://ollama.com/download/windows

## Starting Ollama

Start the Ollama service:
```bash
ollama serve
```

## Downloading Required Models

### Chat Model (llama3.2)
```bash
ollama pull llama3.2
```

### Embedding Model (nomic-embed-text)
```bash
ollama pull nomic-embed-text
```

### Alternative Models

You can use other models by changing the environment variables:

#### Chat Models:
- `llama3.2` (default, 3B parameters, fast)
- `llama3.1` (8B parameters, more capable)
- `mistral` (7B parameters, good balance)
- `mixtral` (8x7B parameters, very capable but slower)
- `qwen2.5` (various sizes available)

#### Embedding Models:
- `nomic-embed-text` (default, optimized for embeddings)
- `mxbai-embed-large` (larger model, better quality)
- `all-minilm` (smaller, faster)

## Configuration

Update your `.env` file:

```env
# Use Ollama as the default provider
LLM_PROVIDER=ollama
EMBEDDING_PROVIDER=ollama

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_CHAT_MODEL=llama3.2
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
```

## Switching Between Providers

You can easily switch between providers by changing the `LLM_PROVIDER` and `EMBEDDING_PROVIDER` variables:

### Use OpenAI:
```env
LLM_PROVIDER=openai
EMBEDDING_PROVIDER=openai
OPENAI_API_KEY=your-api-key-here
```

### Use Claude:
```env
LLM_PROVIDER=claude
EMBEDDING_PROVIDER=ollama  # Claude doesn't provide embeddings
CLAUDE_API_KEY=your-api-key-here
```

### Mixed Configuration:
```env
LLM_PROVIDER=claude
EMBEDDING_PROVIDER=openai
```

## Verifying Installation

1. Check if Ollama is running:
```bash
curl http://localhost:11434/api/tags
```

2. Test a model:
```bash
ollama run llama3.2 "Hello, how are you?"
```

## Performance Tips

1. **Model Selection**: Smaller models like `llama3.2` (3B) are faster but less capable than larger ones like `mixtral` (8x7B).

2. **Hardware Requirements**:
   - Minimum: 8GB RAM for small models
   - Recommended: 16GB+ RAM for larger models
   - GPU acceleration supported on NVIDIA, AMD, and Apple Silicon

3. **Context Length**: Ollama models support various context lengths. Adjust based on your document size needs.

## Troubleshooting

### Ollama not responding
```bash
# Restart Ollama
killall ollama
ollama serve
```

### Model not found
```bash
# List available models
ollama list

# Pull the required model
ollama pull <model-name>
```

### Memory issues
- Use smaller models
- Reduce context length in your application
- Ensure adequate system RAM

## Docker Setup

If using Docker, add Ollama service to `docker-compose.yml`:

```yaml
services:
  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

volumes:
  ollama_data:
```

Then update your backend service to use the Ollama container:
```env
OLLAMA_BASE_URL=http://ollama:11434
```