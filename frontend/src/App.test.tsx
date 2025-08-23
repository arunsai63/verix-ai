import React from 'react';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import App from './App';
import api from './services/api';

jest.mock('./services/api');

describe('App Component', () => {
  beforeEach(() => {
    (api.getDatasets as jest.Mock).mockResolvedValue({ data: [] });
  });

  test('renders app title', () => {
    render(<App />);
    const titleElement = screen.getByText(/VerixAI - Document Analysis Assistant/i);
    expect(titleElement).toBeInTheDocument();
  });

  test('renders all three tabs', () => {
    render(<App />);
    expect(screen.getByText(/Upload Documents/i)).toBeInTheDocument();
    expect(screen.getByText(/Query Documents/i)).toBeInTheDocument();
    expect(screen.getByText(/Manage Datasets/i)).toBeInTheDocument();
  });

  test('loads datasets on mount', async () => {
    const mockDatasets = [
      {
        name: 'test-dataset',
        document_count: 5,
        created_at: Date.now(),
        size_bytes: 1024
      }
    ];
    
    (api.getDatasets as jest.Mock).mockResolvedValue({ data: mockDatasets });
    
    render(<App />);
    
    await waitFor(() => {
      expect(api.getDatasets).toHaveBeenCalled();
    });
  });

  test('switches between tabs', () => {
    render(<App />);
    
    const queryTab = screen.getByRole('tab', { name: /Query Documents/i });
    fireEvent.click(queryTab);
    
    const datasetsTab = screen.getByRole('tab', { name: /Manage Datasets/i });
    fireEvent.click(datasetsTab);
  });

  test('shows notification on dataset load error', async () => {
    (api.getDatasets as jest.Mock).mockRejectedValue(new Error('Failed to load'));
    
    render(<App />);
    
    await waitFor(() => {
      expect(screen.getByText(/Failed to load datasets/i)).toBeInTheDocument();
    });
  });
});
