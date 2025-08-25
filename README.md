# VerixAI: Intelligent Document Analysis with Citations

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/arunsai63/verix-ai)
[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![React Version](https://img.shields.io/badge/react-18-blue.svg)](https://reactjs.org/)

**VerixAI is a production-ready document analysis platform that empowers knowledge workers to query large document collections and receive AI-generated answers with precise, verifiable citations.**

Whether you're a doctor reviewing patient histories, a lawyer researching case law, or an HR professional managing policies, VerixAI adapts to your domain, providing accurate and context-aware responses.

<img width="1541" height="770" alt="image" src="https://github.com/user-attachments/assets/b58ea83a-fb0f-4e24-af46-fd4a8179f4dd" />

---

## ✨ Key Features

- **Multi-Format Document Ingestion**: Process a wide range of file types including PDF, DOCX, PPTX, HTML, TXT, MD, CSV, and XLSX.
- **Intelligent RAG Pipeline**: Utilizes a sophisticated Retrieval-Augmented Generation (RAG) pipeline with hybrid search (semantic + keyword) for accurate information retrieval.
- **Cited & Verifiable Answers**: Every AI-generated answer is backed by clear citations from the source documents, complete with confidence scores.
- **Role-Aware Responses**: The system adapts its language and focus based on the selected user role (e.g., Doctor, Lawyer, HR), providing tailored insights and disclaimers.
- **Dataset Management**: Easily organize documents into distinct, searchable collections for better context isolation and management.
- **Scalable & Asynchronous**: Built with a modern, asynchronous stack (FastAPI, React) and containerized for easy deployment and scaling.

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

### Phase 1: Core Enhancements (Q1 2025)
- [ ] **Multi-LLM Support**: Add support for Claude, Gemini, and local models (Ollama)
- [ ] **Advanced Document Processing**: OCR support for scanned documents
- [ ] **Real-time Collaboration**: Multiple users working on same dataset
- [ ] **Export Functionality**: Export Q&A sessions as reports (PDF/Word)

### Phase 2: Intelligence Features (Q2 2025)
- [ ] **Document Comparison**: Compare and contrast multiple documents
- [ ] **Auto-summarization**: Automatic executive summaries for uploaded documents
- [ ] **Knowledge Graph**: Visual representation of document relationships
- [ ] **Custom Embeddings**: Fine-tune embeddings for specific domains

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

---
