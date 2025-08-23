# verix-ai

Product Name: VerixAI

Context & Goal

Build an AI assistant that helps doctors, lawyers, HR professionals, and similar knowledge workers quickly analyze large volumes of documents and data and return concise, relevant answers with citations. The AI should decide the best technical approach and code organization on its own.

What the Assistant Should Do
	•	Ingest & normalize many file types (PDF, DOCX, PPTX, HTML, email exports, etc.).
	•	Search & retrieve the most relevant information for a user’s query across large datasets.
	•	Summarize & explain findings clearly, with short highlights and citations/snippets.
	•	Support roles (doctor/lawyer/HR) to adapt tone, disclaimers, and result formatting.
	•	Provide an optional web dashboard for uploading documents, managing datasets, running queries, and viewing results.

Must-Use Tools (assistant chooses how to integrate them)
	•	Document conversion: MarkItDown – convert diverse files to clean Markdown for downstream analysis.
https://github.com/microsoft/markitdown
	•	LLM orchestration & retrieval: LangChain – embeddings, retrieval-augmented generation (RAG), and tool/chain calling.
https://python.langchain.com/
	•	Agent framework: Strands Agents SDK – to coordinate tasks like ingest, retrieve, summarize, and respond.
Overview: https://strandsagents.com/latest/
SDK (Python): https://github.com/strands-agents/sdk-python

(You may add other standard components—vector databases, auth, simple APIs/UI—but keep choices and structure at your discretion.)

Target Users & Scenarios
	•	Lawyer: “From thousands of case files, find top precedents for breach of contract in healthcare and summarize why each is relevant, with citations.”
	•	Doctor: “From uploaded patient PDFs, extract key history, medications, and flag potential contraindications (informational only).”
	•	HR: “Compare 2022 vs 2024 Travel Policy; summarize differences and list action items with cited sections.”

Functional Requirements
	1.	Document Ingestion
	•	Batch uploads; show progress/status.
	•	Convert to Markdown (use MarkItDown), chunk if needed, and store with source metadata (filename, page, date, dataset name).
	2.	Search & Answering
	•	Natural-language queries across one or more datasets.
	•	Retrieval-augmented responses with ranked sources and page-level citations.
	•	Short answer + bullet highlights + “open the source” links/snippets.
	3.	Role Awareness
	•	Output style and disclaimers adapt to doctor/lawyer/HR contexts.
	•	Examples in the UI to guide good queries.
	4.	Dashboard (optional but preferred)
	•	Upload & manage datasets; view doc counts and last updated.
	•	Run queries, view results, copy/export answers, and inspect citations.
	•	Basic settings (model choice/API key, dataset visibility).

Non-Functional Requirements
	•	Privacy first: All processing should respect confidentiality; clearly note where data is stored.
	•	Citations by default: If confidence is low or sources are weak, say so and suggest refinements.
	•	Graceful failure: If nothing relevant is found, return a helpful “no strong matches” message.
	•	Performance: Handle large corpora; allow incremental ingestion and re-indexing.
	•	Accessibility & UX: Clear typography, keyboard navigation, and helpful loading/error messages.

Output You Should Produce
	•	Working Python code that implements the ingestion → retrieval → summarized-answer flow using the tools above.
	•	(Preferred) A simple web dashboard for upload, querying, and viewing results.
	•	Run instructions (how to start the backend and the dashboard).
	•	Example requests for each role and sample responses with citations.
	•	Notes on how to switch/extend the vector store or model later.

Guardrails & Disclaimers
	•	Always add role-specific disclaimers (e.g., “This is informational and not legal/medical advice.”).
	•	Avoid hallucinations; never fabricate citations.
	•	Clearly separate facts from interpretations.
	•	Respect PII: redact or minimize sensitive data in examples and logs.

Nice-to-Have (if time permits)
	•	Query history, saved reports, and export to PDF/Markdown.
	•	Dataset-level permissions or private/team workspaces.
	•	Basic analytics (top queries, coverage gaps).
	•	Multi-language document support and queries.
	
Set up project structure and configuration  
Create .gitignore file
Initialize Python backend with dependencies
Implement document ingestion module with MarkItDown
Set up vector database and embeddings with LangChain
Implement Strands Agents SDK integration
Create retrieval and RAG pipeline
Build role-aware response formatting
Develop FastAPI backend with endpoints
Create React frontend dashboard
Implement file upload and dataset management UI
Build query interface and results display
Add citation and source viewing features
Write comprehensive tests for backend
Write tests for frontend components
Create Docker configuration
Add example queries and sample data
Write deployment and usage documentation


create a docs folder to document everything
- techincal documentation
- user documentation
- llm document(for LLM's to get complete project understanding clearly)


prefer to use any component library in react and use tailwind css for styling


