export interface Dataset {
  name: string;
  document_count: number;
  created_at: number;
  size_bytes: number;
  last_updated?: Date;
  metadata?: Record<string, any>;
}

export interface Citation {
  source_number: number;
  filename: string;
  dataset: string;
  chunk_index: number;
  snippet: string;
  relevance_score: number;
}

export interface QueryResult {
  status: string;
  query: string;
  answer: string;
  citations: Citation[];
  highlights: string[];
  confidence: string;
  role: string;
  sources_count: number;
  disclaimer?: string;
  suggested_followup?: string;
}

export interface UploadResponse {
  status: string;
  dataset_name: string;
  documents_processed: number;
  chunks_created: number;
  message: string;
  errors?: Array<{ file: string; error: string }>;
}

export type UserRole = 'general' | 'doctor' | 'lawyer' | 'hr';