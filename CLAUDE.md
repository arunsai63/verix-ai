# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
Currently Project is in development only so no need of production optimisation or scaling or performance improvements.

## Project Overview

VerixAI is a focused document analysis platform using FastAPI (backend) and React (frontend) designed for querying large document collections with AI-generated responses backed by precise citations and advanced retrieval algorithms.

## Commands

### Docker
```bash
docker-compose up --build -d                 # Start all services
docker-compose logs -f backend               # View backend logs
docker-compose down                          # Stop all services
```

## how to test
- using docker compose restart the service you made changes to and verify the changes are reflected in the application.
- first if its builing and running or not, and then if the API's are working are not using curl may be
- then "docker-compose exec backend sh -c "cd /app && PYTHONPATH=/app poetry run pytest tests/" for changes in backend
- npm run build for changes in frontend

## Architecture

### Multi-Agent System (backend/app/agents/)
The system uses specialized agents for document processing and query handling:
- **DocumentIngestionAgent**: Parallel document processing
- **ParallelRetrievalAgent**: Concurrent semantic/keyword search
- **RankingAgent**: Result scoring and ranking
- **AnswerGenerationAgent**: Role-aware answer generation
- **CitationAgent**: Source citation extraction
- **ValidationAgent**: Response quality verification

### Core Services (backend/app/services/)
- **document_processor.py**: Async document processing with worker pools
- **retrieval_service.py**: Hybrid search implementation
- **llm_service.py**: LLM provider abstraction (Ollama/OpenAI/Claude)
- **dataset_service.py**: Dataset management and storage

### API Structure (backend/app/api/)
- **routes/**: FastAPI route handlers
- **providers/**: LLM provider implementations
- All endpoints return standardized responses via schemas

### Frontend Components (frontend/src/)
- **components/**: Reusable UI components using Radix UI
- **pages/**: Route-level components (Query, Upload, Datasets)
- **services/api.ts**: Axios-based API client
- **types/**: TypeScript interfaces matching backend schemas

## Key Configuration

### Environment Variables
- `LLM_PROVIDER`: ollama|openai|claude (default: ollama)
- `EMBEDDING_PROVIDER`: ollama|openai (default: ollama)
- `MULTI_AGENT_ENABLED`: true|false (enables multi-agent system)
- `ASYNC_PROCESSING_ENABLED`: true|false (enables parallel processing)
- `MAX_PARALLEL_DOCUMENTS`: Number of concurrent document processors (default: 5)

### LLM Models
- **Ollama**: llama3.2:1b (generation), nomic-embed-text (embeddings)
- **OpenAI**: gpt-4 (generation), text-embedding-3-small (embeddings)
- **Claude**: claude-3-opus-20240229 (generation only)

## Development Patterns

### Backend Patterns
- Use Pydantic schemas for all API contracts
- Implement async/await for I/O operations
- Follow dependency injection pattern with FastAPI
- Store embeddings in ChromaDB collections
- Use Poetry for dependency management

### Frontend Patterns
- Components use TypeScript with strict typing
- API calls via centralized service layer
- Tailwind CSS for styling with custom components
- React 19 with functional components and hooks
- Vite for build tooling (not Create React App)

### Testing Approach
- Backend: pytest with async fixtures, mock LLM providers
- Frontend: Vitest with React Testing Library
- Test files adjacent to source files (*.test.py, *.test.tsx)
- docker-compose exec backend sh -c "cd /app && PYTHONPATH=/app poetry run pytest tests/ --cov=app --cov-report=term-missing"

## Important Files
- `backend/app/core/config.py`: System configuration and settings
- `backend/app/services/document_processor.py`: Core document processing logic
- `frontend/src/services/api.ts`: Frontend API integration
- `docker-compose.yml`: Multi-service orchestration

things to note:
github cli is installed and working, you can use it. use profile arunsai63 (my github username is arunsai63)
