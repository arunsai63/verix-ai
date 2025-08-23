# VerixAI Technical Documentation

## Architecture Overview

VerixAI is a document analysis platform built with a microservices architecture using the following stack:

### Technology Stack

#### Backend
- **Framework**: FastAPI (Python 3.10+)
- **Document Processing**: MarkItDown
- **Vector Database**: ChromaDB
- **LLM Integration**: LangChain + OpenAI
- **Agent Framework**: Strands Agents SDK
- **Database**: PostgreSQL
- **Cache**: Redis
- **Testing**: Pytest

#### Frontend
- **Framework**: React 18 with TypeScript
- **UI Library**: Material-UI (MUI)
- **HTTP Client**: Axios
- **File Upload**: react-dropzone
- **Markdown Rendering**: react-markdown
- **Testing**: Jest + React Testing Library

#### Infrastructure
- **Containerization**: Docker
- **Orchestration**: Docker Compose
- **Reverse Proxy**: Nginx
- **CI/CD**: GitHub Actions (optional)

## System Architecture

```
┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │
│  React Frontend │────▶│  Nginx Proxy    │
│                 │     │                 │
└─────────────────┘     └────────┬────────┘
                                 │
                                 ▼
                    ┌────────────────────┐
                    │                    │
                    │  FastAPI Backend   │
                    │                    │
                    └──┬──────┬──────┬──┘
                       │      │      │
                ┌──────▼──┐ ┌─▼──┐ ┌─▼────────┐
                │ChromaDB │ │Redis│ │PostgreSQL│
                └─────────┘ └────┘ └──────────┘
```

## Core Components

### 1. Document Processor (`app/services/document_processor.py`)

Handles document conversion and chunking:

```python
class DocumentProcessor:
    - process_file(): Convert single file to markdown
    - process_batch(): Process multiple files
    - _chunk_document(): Split into embeddings-ready chunks
    - extract_citations(): Extract references from content
```

**Key Features:**
- Supports PDF, DOCX, PPTX, HTML, TXT, MD, CSV, XLSX
- Automatic metadata extraction
- Smart chunking with overlap
- Citation detection

### 2. Vector Store Service (`app/services/vector_store.py`)

Manages embeddings and similarity search:

```python
class VectorStoreService:
    - add_documents(): Store document embeddings
    - search(): Semantic similarity search
    - hybrid_search(): Combined semantic + keyword search
    - delete_dataset(): Remove dataset from store
```

**Implementation Details:**
- Uses OpenAI embeddings (text-embedding-3-small)
- ChromaDB for vector storage
- Supports filtering by dataset
- Relevance score thresholding

### 3. RAG Service (`app/services/rag_service.py`)

Generates answers using retrieval-augmented generation:

```python
class RAGService:
    - generate_answer(): Create cited response
    - _prepare_context(): Format retrieved chunks
    - _parse_response(): Extract citations
    - generate_summary(): Summarize documents
```

**Role-Specific Prompts:**
- Doctor: Medical terminology, health disclaimers
- Lawyer: Legal language, legal disclaimers
- HR: Compliance focus, actionable insights
- General: Clear, informative responses

### 4. Agent System (`app/agents/document_agent.py`)

Orchestrates document processing pipeline:

```python
class DocumentOrchestrator:
    - DocumentIngestionAgent: File processing
    - RetrievalAgent: Document search
    - SummarizationAgent: Answer generation
    - process_query(): Complete query pipeline
```

## API Endpoints

### Document Management

#### POST `/api/upload`
Upload and process documents
```json
{
  "files": [File],
  "dataset_name": "string",
  "metadata": {}
}
```

#### GET `/api/datasets`
List all datasets
```json
[
  {
    "name": "string",
    "document_count": 0,
    "created_at": 0,
    "size_bytes": 0
  }
]
```

#### DELETE `/api/datasets/{dataset_name}`
Delete a dataset

#### GET `/api/datasets/{dataset_name}/stats`
Get dataset statistics

### Query Endpoints

#### POST `/api/query`
Query documents with AI
```json
{
  "query": "string",
  "dataset_names": ["string"],
  "role": "general|doctor|lawyer|hr",
  "max_results": 10
}
```

Response:
```json
{
  "status": "success",
  "answer": "string",
  "citations": [...],
  "highlights": [...],
  "confidence": "high|medium|low"
}
```

## Data Flow

### Document Ingestion Pipeline

1. **Upload**: Files uploaded via multipart form
2. **Storage**: Files saved to dataset directory
3. **Conversion**: MarkItDown converts to markdown
4. **Chunking**: Content split into overlapping chunks
5. **Embedding**: Chunks embedded via OpenAI
6. **Storage**: Embeddings stored in ChromaDB

### Query Processing Pipeline

1. **Query Receipt**: User query with role/dataset filters
2. **Embedding**: Query converted to vector
3. **Retrieval**: Similarity search in ChromaDB
4. **Context Prep**: Top chunks formatted
5. **LLM Generation**: OpenAI generates answer
6. **Citation Extraction**: Sources linked to response
7. **Response**: Formatted answer with citations

## Database Schema

### PostgreSQL Tables (if using full DB)

```sql
-- Datasets table
CREATE TABLE datasets (
    id UUID PRIMARY KEY,
    name VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    metadata JSONB
);

-- Documents table
CREATE TABLE documents (
    id UUID PRIMARY KEY,
    dataset_id UUID REFERENCES datasets(id),
    filename VARCHAR(255),
    file_hash VARCHAR(64),
    processed_at TIMESTAMP,
    metadata JSONB
);

-- Queries table (for analytics)
CREATE TABLE queries (
    id UUID PRIMARY KEY,
    query_text TEXT,
    role VARCHAR(50),
    dataset_names TEXT[],
    response_time_ms INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);
```

## Performance Optimization

### Caching Strategy
- Redis for query result caching
- 15-minute TTL for repeated queries
- Dataset metadata caching

### Chunking Strategy
- Default chunk size: 1000 characters
- Overlap: 200 characters
- Preserves semantic boundaries

### Embedding Optimization
- Batch processing for multiple documents
- Parallel chunk embedding
- Incremental indexing

## Security Considerations

### API Security
- CORS configuration
- Rate limiting (TODO)
- API key authentication (TODO)
- Input validation

### Data Security
- File type validation
- Size limits (100MB default)
- Sanitized filenames
- SQL injection prevention

### Secrets Management
- Environment variables for sensitive data
- Never commit .env files
- Rotate keys regularly

## Testing Strategy

### Backend Tests
```bash
cd backend
pytest tests/ -v --cov=app
```

**Test Coverage:**
- Unit tests for services
- Integration tests for API
- Mock tests for external services

### Frontend Tests
```bash
cd frontend
npm test -- --coverage
```

**Test Coverage:**
- Component rendering tests
- User interaction tests
- API integration tests

## Monitoring & Logging

### Logging Configuration
```python
logging.basicConfig(
    level=settings.log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### Health Checks
- `/health` endpoint for backend
- Database connectivity check
- Vector store availability

### Metrics to Monitor
- Query response time
- Document processing time
- Error rates
- Memory usage
- Disk usage

## Development Workflow

### Local Development
```bash
# Backend
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload

# Frontend
cd frontend
npm install
npm start
```

### Code Style
- Python: Black formatter, Flake8 linting
- TypeScript: ESLint, Prettier
- Pre-commit hooks for consistency

## Troubleshooting Guide

### Common Issues

1. **Import Errors**
```bash
# Ensure all packages installed
pip install -r requirements.txt --force-reinstall
```

2. **Vector Store Connection**
```bash
# Check ChromaDB is running
docker ps | grep chromadb
```

3. **OpenAI API Errors**
- Check API key validity
- Monitor rate limits
- Verify network connectivity

## Future Enhancements

### Planned Features
- Multi-language support
- Real-time collaboration
- Advanced analytics dashboard
- Custom embedding models
- Fine-tuned LLMs

### Scalability Improvements
- Kubernetes deployment
- Distributed vector store
- Message queue integration
- Caching layer expansion

## Contributing

### Code Standards
- Type hints for Python
- TypeScript strict mode
- Comprehensive docstrings
- Test coverage >80%

### Pull Request Process
1. Create feature branch
2. Write tests
3. Update documentation
4. Submit PR with description
5. Pass CI/CD checks