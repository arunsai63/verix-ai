import React, { useState, useCallback } from 'react';
import {
  Box,
  Button,
  TextField,
  Typography,
  LinearProgress,
  Alert,
  Card,
  CardContent,
  Chip,
  Stack,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
} from '@mui/material';
import { CloudUpload, Delete } from '@mui/icons-material';
import { useDropzone } from 'react-dropzone';
import api from '../services/api';
import { Dataset } from '../types';

interface UploadSectionProps {
  onUploadComplete: () => void;
  datasets: Dataset[];
}

const UploadSection: React.FC<UploadSectionProps> = ({ onUploadComplete, datasets }) => {
  const [files, setFiles] = useState<File[]>([]);
  const [datasetName, setDatasetName] = useState('');
  const [useExistingDataset, setUseExistingDataset] = useState(false);
  const [selectedDataset, setSelectedDataset] = useState('');
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [uploadProgress, setUploadProgress] = useState(0);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    setFiles((prev) => [...prev, ...acceptedFiles]);
    setError(null);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
      'application/vnd.openxmlformats-officedocument.presentationml.presentation': ['.pptx'],
      'text/html': ['.html'],
      'text/plain': ['.txt'],
      'text/markdown': ['.md'],
      'text/csv': ['.csv'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
    },
    maxSize: 100 * 1024 * 1024, // 100MB
  });

  const removeFile = (index: number) => {
    setFiles((prev) => prev.filter((_, i) => i !== index));
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
  };

  const handleUpload = async () => {
    if (files.length === 0) {
      setError('Please select at least one file to upload');
      return;
    }

    const targetDataset = useExistingDataset ? selectedDataset : datasetName;
    if (!targetDataset) {
      setError('Please enter a dataset name or select an existing one');
      return;
    }

    setUploading(true);
    setError(null);
    setUploadProgress(0);

    try {
      const progressInterval = setInterval(() => {
        setUploadProgress((prev) => Math.min(prev + 10, 90));
      }, 500);

      await api.uploadDocuments(files, targetDataset);

      clearInterval(progressInterval);
      setUploadProgress(100);

      setTimeout(() => {
        setFiles([]);
        setDatasetName('');
        setSelectedDataset('');
        setUploadProgress(0);
        onUploadComplete();
      }, 1000);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Upload failed. Please try again.');
    } finally {
      setUploading(false);
    }
  };

  return (
    <Box>
      <Typography variant="h5" gutterBottom>
        Upload Documents
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        Upload PDF, Word, PowerPoint, HTML, or text files to create or add to a dataset.
      </Typography>

      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Box
            {...getRootProps()}
            sx={{
              border: '2px dashed',
              borderColor: isDragActive ? 'primary.main' : 'grey.300',
              borderRadius: 2,
              p: 4,
              textAlign: 'center',
              cursor: 'pointer',
              backgroundColor: isDragActive ? 'action.hover' : 'background.paper',
              transition: 'all 0.3s',
              '&:hover': {
                borderColor: 'primary.main',
                backgroundColor: 'action.hover',
              },
            }}
          >
            <input {...getInputProps()} />
            <CloudUpload sx={{ fontSize: 48, color: 'text.secondary', mb: 2 }} />
            <Typography variant="h6" gutterBottom>
              {isDragActive ? 'Drop files here' : 'Drag & drop files here'}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              or click to browse files
            </Typography>
            <Typography variant="caption" display="block" sx={{ mt: 1 }}>
              Supported: PDF, DOCX, PPTX, HTML, TXT, MD, CSV, XLSX (Max: 100MB per file)
            </Typography>
          </Box>
        </CardContent>
      </Card>

      {files.length > 0 && (
        <Card sx={{ mb: 3 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Selected Files ({files.length})
            </Typography>
            <Stack spacing={1}>
              {files.map((file, index) => (
                <Box
                  key={index}
                  sx={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    p: 1,
                    borderRadius: 1,
                    backgroundColor: 'grey.50',
                  }}
                >
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Typography variant="body2">{file.name}</Typography>
                    <Chip label={formatFileSize(file.size)} size="small" />
                  </Box>
                  <Button
                    size="small"
                    color="error"
                    startIcon={<Delete />}
                    onClick={() => removeFile(index)}
                    disabled={uploading}
                  >
                    Remove
                  </Button>
                </Box>
              ))}
            </Stack>
          </CardContent>
        </Card>
      )}

      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Dataset Configuration
          </Typography>
          
          <FormControl fullWidth sx={{ mb: 2 }}>
            <InputLabel>Dataset Option</InputLabel>
            <Select
              value={useExistingDataset ? 'existing' : 'new'}
              onChange={(e) => setUseExistingDataset(e.target.value === 'existing')}
              disabled={uploading}
            >
              <MenuItem value="new">Create New Dataset</MenuItem>
              <MenuItem value="existing">Add to Existing Dataset</MenuItem>
            </Select>
          </FormControl>

          {useExistingDataset ? (
            <FormControl fullWidth sx={{ mb: 2 }}>
              <InputLabel>Select Dataset</InputLabel>
              <Select
                value={selectedDataset}
                onChange={(e) => setSelectedDataset(e.target.value)}
                disabled={uploading || datasets.length === 0}
              >
                {datasets.map((dataset) => (
                  <MenuItem key={dataset.name} value={dataset.name}>
                    {dataset.name} ({dataset.document_count} documents)
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          ) : (
            <TextField
              fullWidth
              label="New Dataset Name"
              value={datasetName}
              onChange={(e) => setDatasetName(e.target.value)}
              disabled={uploading}
              placeholder="e.g., medical-records-2024"
              sx={{ mb: 2 }}
            />
          )}

          {error && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {error}
            </Alert>
          )}

          {uploading && (
            <Box sx={{ mb: 2 }}>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                Uploading and processing documents...
              </Typography>
              <LinearProgress variant="determinate" value={uploadProgress} />
            </Box>
          )}

          <Button
            fullWidth
            variant="contained"
            size="large"
            startIcon={<CloudUpload />}
            onClick={handleUpload}
            disabled={uploading || files.length === 0}
          >
            {uploading ? 'Uploading...' : 'Upload Documents'}
          </Button>
        </CardContent>
      </Card>
    </Box>
  );
};

export default UploadSection;