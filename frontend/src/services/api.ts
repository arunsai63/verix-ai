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
};

export default api;