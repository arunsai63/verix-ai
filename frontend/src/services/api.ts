import axios from 'axios';
import { Dataset, QueryResult, UploadResponse } from '../types';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

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
    metadata?: Record<string, any>
  ): Promise<{ data: UploadResponse }> {
    const formData = new FormData();
    files.forEach((file) => {
      formData.append('files', file);
    });
    formData.append('dataset_name', datasetName);
    if (metadata) {
      formData.append('metadata', JSON.stringify(metadata));
    }

    return apiClient.post('/api/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
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