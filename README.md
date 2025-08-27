# VerixAI: Intelligent Document Analysis with Citations

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/arunsai63/verix-ai)
[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![React Version](https://img.shields.io/badge/react-18-blue.svg)](https://reactjs.org/)

**VerixAI is a production-ready document analysis platform focused on one core mission: helping knowledge workers query large document collections efficiently and receive AI-generated answers with precise, verifiable citations.**

Designed for professionals handling extensive document repositories—whether medical records, legal cases, or corporate policies—VerixAI provides accurate, context-aware responses backed by advanced retrieval and ranking algorithms.

<img width="1541" height="770" alt="image" src="https://github.com/user-attachments/assets/b58ea83a-fb0f-4e24-af46-fd4a8179f4dd" />

---

## ✨ Key Features

- **Multi-Format Document Ingestion**: Process PDF, DOCX, PPTX, HTML, TXT, MD, XLSX, and JSON files with high-fidelity extraction.
- **Advanced RAG Pipeline**: Sophisticated Retrieval-Augmented Generation with hybrid search (semantic + keyword) and advanced ranking algorithms.
- **Precise Citations**: Every answer includes clear citations with source documents, chunk indices, and confidence scores for full verifiability.
- **Role-Aware Responses**: Tailored responses for different professional contexts (Doctor, Lawyer, HR) with appropriate disclaimers and terminology.
- **Dataset Management**: Organize documents into isolated, searchable collections for improved query accuracy and context control.
- **Multi-Agent Architecture**: Specialized AI agents for document ingestion, parallel retrieval, ranking, citation validation, and quality control.
- **Parallel Processing**: Asynchronous document processing with configurable worker pools for handling large-scale document collections.
- **Scalable Infrastructure**: Modern async stack (FastAPI, React) containerized for easy deployment and horizontal scaling.

---

## 🚀 Live Demo & Screenshots

**✨ [Check out the live demo here!](https://arunsai63.github.io/verix-ai/) ✨**

<img width="1538" height="771" alt="image" src="https://github.com/user-attachments/assets/a534bb5b-b078-4af9-a38b-0f2d0b727189" />


**Query Interface**
<img width="1534" height="763" alt="image" src="https://github.com/user-attachments/assets/3a510c9e-3221-4bde-86b9-2a956189c536" />


**Upload and Dataset Management**
<img width="1232" height="415" alt="image" src="https://github.com/user-attachments/assets/9ea08801-0366-40e8-bbdb-b528857b91da" />

---

## 🗺️ Roadmap

### Recently Completed ✅
- [x] **Multi-Agent Architecture**: Specialized agents for document processing and retrieval
- [x] **Advanced Retrieval System**: Hybrid search with semantic and keyword matching
- [x] **Parallel Document Processing**: Asynchronous processing with worker pools
- [x] **Multi-LLM Support**: Support for Ollama (local models), OpenAI, and Claude
- [x] **Citation Validation**: Automated validation of source citations

### Phase 1: Core Enhancements (Q1 2025)
- [ ] **Advanced Document Processing**: OCR support for scanned documents
- [ ] **Real-time Collaboration**: Multiple users working on same dataset
- [ ] **Export Functionality**: Export Q&A sessions as reports (PDF/Word)
- [ ] **Voice Input/Output**: Speech-to-text queries and text-to-speech responses

### Phase 2: Intelligence Features (Q2 2025)
- [ ] **Document Comparison**: Compare and contrast multiple documents
- [ ] **Knowledge Graph**: Visual representation of document relationships
- [ ] **Custom Embeddings**: Fine-tune embeddings for specific domains
- [ ] **Advanced Analytics Dashboard**: Usage metrics and insights

### Phase 3: Enterprise Features (Q3 2025)
- [ ] **SSO Integration**: SAML/OAuth support for enterprise authentication
- [ ] **Audit Logging**: Complete audit trail for compliance
- [ ] **API Rate Limiting**: Advanced rate limiting and usage analytics
- [ ] **Multi-tenancy**: Support for multiple isolated organizations

### Phase 4: Advanced Analytics (Q4 2025)
- [ ] **Sentiment Analysis**: Analyze document sentiment and tone
- [ ] **Entity Recognition**: Extract and link named entities
- [ ] **Time-series Analysis**: Track changes across document versions
- [ ] **Custom Models**: Support for domain-specific fine-tuned models

### Community Contributions Welcome!
We encourage contributions in these areas:
- Additional file format support (epub, rtf, etc.)
- Language translations and internationalization
- Performance optimizations
- Bug fixes and documentation improvements

---

## 🛠️ Tech Stack & Architecture

VerixAI is built with a modern, microservices-oriented architecture.

| Component | Technology |
| :--- | :--- |
| **Backend** | FastAPI, Python 3.10+, LangChain, Uvicorn |
| **Frontend** | React 18, TypeScript, Material-UI (MUI), Axios |
| **Vector Database** | ChromaDB |
| **LLM & Embeddings** | OpenAI (GPT-4, text-embedding-3-small) |
| **Document Processing** | MarkItDown |
| **Infrastructure** | Docker, Docker Compose, Nginx |
| **Optional** | PostgreSQL (for metadata) |

### System Architecture

```
┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │
│  React Frontend │────▶│  Nginx Proxy    │
│ (Port 3000)     │     │ (Production)    │
│                 │     │                 │
└─────────────────┘     └────────┬────────┘
                                 │
                                 ▼
                    ┌────────────────────┐
                    │                    │
                    │  FastAPI Backend   │
                    │   (Port 8000)      │
                    └──┬──────┬──────┬──┘
                       │             │
                ┌──────▼──┐        ┌─▼────────┐
                │ChromaDB │        │PostgreSQL│
                └─────────┘        └──────────┘
```

---

## 🏁 Getting Started

The easiest way to get VerixAI running is with Docker and Docker Compose.

### Prerequisites

- Docker and Docker Compose installed
- An OpenAI API Key
- Git

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/arunsai63/verix-ai.git
    cd verix-ai
    ```

2.  **Set up your environment variables:**
    -   Copy the example environment file:
        ```bash
        cp .env.example .env
        ```
    -   Edit the `.env` file and add your OpenAI API key:
        ```
        OPENAI_API_KEY=your-actual-api-key-here
        # Generate a secure random key for JWTs
        SECRET_KEY=a_very_secure_random_string_of_at_least_32_characters
        ```

3.  **Launch the application with Docker Compose:**
    ```bash
    docker-compose up --build -d
    ```
    This command builds the images and starts the frontend, backend, and ChromaDB services in detached mode.

4.  **Access VerixAI:**
    -   **Frontend Application**: [http://localhost:3000](http://localhost:3000)
    -   **Backend API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 📖 Usage

1.  **Upload Documents**: Navigate to the "Upload" tab, select or drag-and-drop your files, assign them to a new or existing dataset, and click "Upload".
2.  **Query Documents**: Go to the "Query" tab, type your question, select the dataset(s) to search, choose a professional role, and get your cited answer.
3.  **Manage Datasets**: View, inspect, and delete your document collections from the "Datasets" tab.

### Example Queries

-   **General**: `"Summarize the key findings from the Q4 reports."`
-   **Doctor**: `"What are the patient's pre-existing conditions and current medications?"`
-   **Lawyer**: `"Find all precedents related to intellectual property theft in the provided case files."`
-   **HR**: `"What is the company's official policy on remote work and what are the eligibility criteria?"`

### 🆕 New Features

#### Document Summarization
Navigate to the "Summarize" tab to generate various types of summaries:
- **Summary Types**: Executive, Key Points, Chapter-wise, Technical, Bullet Points, Abstract
- **Length Options**: Brief (1-2 paragraphs), Standard (1 page), Detailed (2-3 pages)
- **Custom Instructions**: Add specific guidance for the summary generation

#### Interactive Chat
Use the "Chat" tab for conversational document analysis:
- Create chat sessions with one or multiple datasets
- Ask follow-up questions with maintained context
- View citations and sources for each response
- Export conversation history as JSON or Markdown

#### CSV Analytics
Upload CSV files and analyze them using natural language:
- **Example Queries**:
  - "What is the average sales by region?"
  - "Show me the trend of revenue over time"
  - "Find correlations between variables"
- **Automatic Visualizations**: Line charts, bar graphs, heatmaps, scatter plots
- **Statistical Analysis**: Descriptive statistics, correlations, distributions
- **Data Export**: Download results as JSON, CSV, or HTML reports

---
