// API service for communicating with the Sentinel X backend

const API_BASE_URL = 'http://localhost:8000';

export interface LogAnalysisRequest {
  logs: string;
  use_neural_network?: boolean;
  analysis_type?: string;
}

export interface URLAnalysisRequest {
  url: string;
  crawl_enabled?: boolean;
  max_pages?: number;
  crawl_delay?: number;
  use_neural_network?: boolean;
}

export interface LineDetail {
  line_number: number;
  content: string;
  score: number;
  flags: string[];
}

export interface AnalysisResponse {
  timestamp: string;
  verdict: string;
  method_used: string;
  justification: string;
  ann_prob?: number;
  dfa_detail: Record<string, number[]>;
  features: Record<string, any>;
  plain_explanation: string;
  line_details: LineDetail[];
}

export interface RecentAnalysis {
  id: string;
  timestamp: string;
  verdict: string;
  method_used: string;
  line_count: number;
}

export interface ModelInfo {
  filename: string;
  size: number;
  modified: string;
}

class APIService {
  private async request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
    const url = `${API_BASE_URL}${endpoint}`;
    
    const config: RequestInit = {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    };

    try {
      const response = await fetch(url, config);
      
      if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: 'Network error' }));
        throw new Error(error.detail || `HTTP ${response.status}`);
      }
      
      return await response.json();
    } catch (error) {
      if (error instanceof Error) {
        throw error;
      }
      throw new Error('Unknown error occurred');
    }
  }

  async healthCheck(): Promise<{ status: string; timestamp: string }> {
    return this.request('/health');
  }

  async analyzeLogs(request: LogAnalysisRequest): Promise<AnalysisResponse> {
    return this.request('/analyze/logs', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  async analyzeURL(request: URLAnalysisRequest): Promise<AnalysisResponse> {
    return this.request('/analyze/url', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  async uploadModel(file: File): Promise<{ message: string }> {
    const formData = new FormData();
    formData.append('file', file);
    
    return this.request('/upload-model', {
      method: 'POST',
      headers: {}, // Don't set Content-Type, let browser set it for FormData
      body: formData,
    });
  }

  async getModels(): Promise<{ models: ModelInfo[] }> {
    return this.request('/models');
  }

  async getRecentAnalyses(): Promise<{ analyses: RecentAnalysis[] }> {
    return this.request('/recent-analyses');
  }

  async getServerInfo(): Promise<{ message: string; version: string; xgboost_available: boolean }> {
    return this.request('/');
  }
}

export const apiService = new APIService();