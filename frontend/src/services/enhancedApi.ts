import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export interface EnhancedUploadOptions {
  chunkingStrategy?: 'semantic' | 'hierarchical' | 'dynamic' | 'auto' | 'hybrid';
  maxChunkSize?: number;
  minChunkSize?: number;
  enableHierarchical?: boolean;
  optimizeChunking?: boolean;
}

export interface EnhancedQueryOptions {
  retrievalStrategy?: 'hybrid' | 'semantic' | 'keyword' | 'auto';
  enableReranking?: boolean;
  enableQueryExpansion?: boolean;
  enableHyDE?: boolean;
  topK?: number;
  filters?: Record<string, any>;
}

const enhancedApi = {
  /**
   * Upload documents with enhanced chunking strategies
   */
  async uploadDocuments(
    files: File[],
    datasetName: string,
    options: EnhancedUploadOptions = {}
  ): Promise<{ data: any }> {
    const formData = new FormData();
    files.forEach((file) => {
      formData.append('files', file);
    });
    formData.append('dataset_name', datasetName);
    formData.append('chunking_strategy', options.chunkingStrategy || 'auto');
    formData.append('max_chunk_size', String(options.maxChunkSize || 1500));
    formData.append('min_chunk_size', String(options.minChunkSize || 100));
    formData.append('enable_hierarchical', String(options.enableHierarchical !== false));
    formData.append('optimize_chunking', String(options.optimizeChunking || false));

    return apiClient.post('/api/enhanced/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
  },

  /**
   * Query documents with enhanced retrieval mechanisms
   */
  async queryDocuments(
    query: string,
    datasetName: string,
    role: string = 'general',
    options: EnhancedQueryOptions = {}
  ): Promise<{ data: any }> {
    return apiClient.post('/api/enhanced/query', {
      query,
      dataset_name: datasetName,
      role,
      top_k: options.topK || 10,
      retrieval_strategy: options.retrievalStrategy || 'hybrid',
      enable_reranking: options.enableReranking !== false,
      enable_query_expansion: options.enableQueryExpansion !== false,
      enable_hyde: options.enableHyDE !== false,
      filters: options.filters,
    });
  },

  /**
   * Multi-hop query for complex questions
   */
  async multiHopQuery(
    query: string,
    datasetName: string,
    maxHops: number = 3,
    topKPerHop: number = 5
  ): Promise<{ data: any }> {
    return apiClient.post('/api/enhanced/query/multi-hop', {
      query,
      dataset_name: datasetName,
      max_hops: maxHops,
      top_k_per_hop: topKPerHop,
    });
  },

  /**
   * Get available chunking strategies
   */
  async getChunkingStrategies(): Promise<{ data: any }> {
    return apiClient.get('/api/enhanced/chunking/strategies');
  },

  /**
   * Get available retrieval strategies
   */
  async getRetrievalStrategies(): Promise<{ data: any }> {
    return apiClient.get('/api/enhanced/retrieval/strategies');
  },

  /**
   * Analyze chunk quality for a dataset
   */
  async analyzeChunks(
    datasetName: string,
    sampleSize: number = 100
  ): Promise<{ data: any }> {
    const formData = new FormData();
    formData.append('dataset_name', datasetName);
    formData.append('sample_size', String(sampleSize));
    
    return apiClient.post('/api/enhanced/analyze/chunks', formData);
  },

  /**
   * Check enhanced services health
   */
  async checkHealth(): Promise<{ data: any }> {
    return apiClient.get('/api/enhanced/health');
  },
};

export default enhancedApi;