import axios from 'axios';
import { Dataset, QueryResult, UploadResponse } from '../types';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

const api = {
  async uploadDocuments(
    files: File[],
    datasetName: string,
    metadata?: Record<string, any>,
    useCelery: boolean = true
  ): Promise<{ data: any }> {
    const formData = new FormData();
    files.forEach((file) => {
      formData.append('files', file);
    });
    formData.append('dataset_name', datasetName);
    if (metadata) {
      formData.append('metadata', JSON.stringify(metadata));
    }
    formData.append('use_celery', String(useCelery));

    return apiClient.post('/api/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
  },

  async getJobStatus(jobId: string): Promise<{ data: any }> {
    return apiClient.get(`/api/jobs/${jobId}/status`);
  },

  async getAllJobs(limit: number = 100, offset: number = 0, status?: string): Promise<{ data: { jobs: any[]; total: number } }> {
    const params = new URLSearchParams();
    params.append('limit', String(limit));
    params.append('offset', String(offset));
    if (status) {
      params.append('status', status);
    }
    return apiClient.get(`/api/jobs?${params.toString()}`);
  },

  async getActiveJobs(): Promise<{ data: { active_jobs: any[]; total: number } }> {
    return apiClient.get('/api/jobs/active');
  },

  async cancelJob(jobId: string): Promise<{ data: any }> {
    return apiClient.delete(`/api/jobs/${jobId}`);
  },

  async getJobMetrics(): Promise<{ data: any }> {
    return apiClient.get('/api/jobs/metrics');
  },

  async queryDocuments(
    query: string,
    datasetNames?: string[],
    role: string = 'general',
    maxResults: number = 10
  ): Promise<{ data: QueryResult }> {
    return apiClient.post('/api/query', {
      query,
      dataset_names: datasetNames,
      role,
      max_results: maxResults,
    });
  },

  async getDatasets(): Promise<{ data: Dataset[] }> {
    return apiClient.get('/api/datasets');
  },

  async deleteDataset(datasetName: string): Promise<void> {
    return apiClient.delete(`/api/datasets/${datasetName}`);
  },

  async getDatasetStats(datasetName: string): Promise<{ data: any }> {
    return apiClient.get(`/api/datasets/${datasetName}/stats`);
  },

  // Summarization endpoints
  async summarizeDocument(
    datasetName?: string,
    documentName?: string,
    content?: string,
    summaryType: string = 'executive',
    length: string = 'standard',
    customInstructions?: string,
    includeCitations: boolean = true
  ): Promise<{ data: any }> {
    return apiClient.post('/api/summarize', {
      dataset_name: datasetName,
      document_name: documentName,
      content,
      summary_type: summaryType,
      length,
      custom_instructions: customInstructions,
      include_citations: includeCitations,
    });
  },

  async getSummaryTypes(): Promise<{ data: any }> {
    return apiClient.get('/api/summarize/types');
  },

  // Chat endpoints
  async createChatSession(
    datasetNames: string[],
    metadata?: Record<string, any>
  ): Promise<{ data: any }> {
    return apiClient.post('/api/chat/session', {
      dataset_names: datasetNames,
      metadata,
    });
  },

  async sendChatMessage(
    sessionId: string,
    message: string,
    stream: boolean = false
  ): Promise<{ data: any }> {
    return apiClient.post('/api/chat/message', {
      session_id: sessionId,
      message,
      stream,
    });
  },

  async getChatHistory(sessionId: string): Promise<{ data: any }> {
    return apiClient.get(`/api/chat/session/${sessionId}`);
  },

  async listChatSessions(
    datasetName?: string,
    limit: number = 10
  ): Promise<{ data: any }> {
    const params = new URLSearchParams();
    if (datasetName) params.append('dataset_name', datasetName);
    params.append('limit', String(limit));
    return apiClient.get(`/api/chat/sessions?${params.toString()}`);
  },

  async deleteChatSession(sessionId: string): Promise<void> {
    return apiClient.delete(`/api/chat/session/${sessionId}`);
  },

  async exportChatSession(
    sessionId: string,
    format: string = 'json'
  ): Promise<{ data: any }> {
    return apiClient.get(`/api/chat/session/${sessionId}/export?format=${format}`);
  },

  // CSV Analytics endpoints
  async analyzeCSV(
    datasetName: string,
    query: string,
    fileName?: string,
    visualize: boolean = true,
    filters?: Record<string, any>,
    exportFormat?: string
  ): Promise<{ data: any }> {
    return apiClient.post('/api/analytics/csv', {
      dataset_name: datasetName,
      file_name: fileName,
      query,
      visualize,
      filters,
      export_format: exportFormat,
    });
  },

  async getAnalysisTypes(): Promise<{ data: any }> {
    return apiClient.get('/api/analytics/analysis-types');
  },
};

export default api;