import React, { useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useDropzone } from 'react-dropzone';
import {
  Upload,
  File,
  X,
  CheckCircle,
  AlertCircle,
  FileText,
  FileSpreadsheet,
  FileImage,
  FileCode,
  Folder,
  Plus,
  Sparkles,
  Cloud,
  ArrowUpCircle
} from 'lucide-react';
import api from '../services/api';
import { Dataset } from '../types';
import Card from './ui/Card';
import Button from './ui/Button';
import Input from './ui/Input';
import Badge from './ui/Badge';
import Progress from './ui/Progress';

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
  const [uploadSuccess, setUploadSuccess] = useState(false);

  const onDrop = useCallback((acceptedFiles: File[], rejectedFiles: any[]) => {
    setFiles((prev) => [...prev, ...acceptedFiles]);
    setError(null);
    
    if (rejectedFiles.length > 0) {
      const errors = rejectedFiles.map(f => 
        f.errors[0]?.code === 'file-too-large' 
          ? `${f.file.name} exceeds 100MB limit`
          : `${f.file.name} is not a supported file type`
      );
      setError(errors.join(', '));
    }
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

  const getFileIcon = (fileName: string) => {
    const ext = fileName.split('.').pop()?.toLowerCase();
    switch (ext) {
      case 'pdf':
        return <FileText className="w-5 h-5 text-error-500" />;
      case 'docx':
      case 'doc':
        return <FileText className="w-5 h-5 text-primary-500" />;
      case 'xlsx':
      case 'xls':
      case 'csv':
        return <FileSpreadsheet className="w-5 h-5 text-success-500" />;
      case 'pptx':
      case 'ppt':
        return <FileImage className="w-5 h-5 text-warning-500" />;
      case 'html':
      case 'md':
        return <FileCode className="w-5 h-5 text-accent-500" />;
      default:
        return <File className="w-5 h-5 text-neutral-500" />;
    }
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
    setUploadSuccess(false);

    try {
      const progressInterval = setInterval(() => {
        setUploadProgress((prev) => Math.min(prev + 10, 90));
      }, 500);

      await api.uploadDocuments(files, targetDataset);

      clearInterval(progressInterval);
      setUploadProgress(100);
      setUploadSuccess(true);

      setTimeout(() => {
        setFiles([]);
        setDatasetName('');
        setSelectedDataset('');
        setUploadProgress(0);
        setUploadSuccess(false);
        onUploadComplete();
      }, 2000);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Upload failed. Please try again.');
      setUploadProgress(0);
    } finally {
      setUploading(false);
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="max-w-4xl mx-auto space-y-6"
    >
      {/* Header */}
      <div>
        <h1 className="text-3xl font-display font-bold text-neutral-900 dark:text-neutral-100 mb-2">
          Upload Documents
        </h1>
        <p className="text-neutral-600 dark:text-neutral-400">
          Transform your documents into searchable knowledge with AI-powered analysis
        </p>
      </div>

      {/* Upload Area */}
      <Card variant="default">
        <div className="p-8">
          <div
            {...getRootProps()}
            className={`
              relative border-2 border-dashed rounded-2xl p-12 text-center cursor-pointer
              transition-all duration-300 group
              ${isDragActive 
                ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/20' 
                : 'border-neutral-300 dark:border-neutral-700 hover:border-primary-400 hover:bg-primary-50/50 dark:hover:bg-primary-900/10'
              }
            `}
          >
            <input {...getInputProps()} />
            
            {/* Animated background pattern */}
            <div className="absolute inset-0 opacity-5">
              <div className="absolute inset-0 bg-gradient-mesh animate-gradient"></div>
            </div>
            
            <motion.div
              animate={{ 
                y: isDragActive ? -10 : 0,
                scale: isDragActive ? 1.1 : 1
              }}
              transition={{ type: 'spring', stiffness: 300 }}
              className="relative z-10"
            >
              <div className="w-20 h-20 mx-auto mb-6 bg-gradient-to-br from-primary-400 to-accent-400 rounded-2xl flex items-center justify-center group-hover:scale-110 transition-transform">
                <Cloud className="w-10 h-10 text-white" />
              </div>
              
              <h3 className="text-xl font-semibold mb-2 text-neutral-900 dark:text-neutral-100">
                {isDragActive ? 'Drop your files here' : 'Drag & drop files or click to browse'}
              </h3>
              
              <p className="text-neutral-600 dark:text-neutral-400 mb-4">
                Support for PDF, DOCX, PPTX, HTML, TXT, MD, CSV, XLSX
              </p>
              
              <div className="flex items-center justify-center space-x-6 text-sm text-neutral-500">
                <div className="flex items-center space-x-2">
                  <CheckCircle className="w-4 h-4 text-success-500" />
                  <span>Max 100MB per file</span>
                </div>
                <div className="flex items-center space-x-2">
                  <CheckCircle className="w-4 h-4 text-success-500" />
                  <span>Multiple files supported</span>
                </div>
              </div>
            </motion.div>
          </div>
        </div>
      </Card>

      {/* Selected Files */}
      <AnimatePresence>
        {files.length > 0 && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
          >
            <Card variant="default">
              <div className="p-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">
                    Selected Files
                  </h3>
                  <Badge variant="primary" size="md">
                    {files.length} {files.length === 1 ? 'file' : 'files'}
                  </Badge>
                </div>
                
                <div className="space-y-2 max-h-64 overflow-y-auto custom-scrollbar">
                  {files.map((file, index) => (
                    <motion.div
                      key={`${file.name}-${index}`}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: index * 0.05 }}
                      className="flex items-center justify-between p-3 rounded-xl bg-neutral-50 dark:bg-neutral-800 hover:bg-neutral-100 dark:hover:bg-neutral-700 transition-colors"
                    >
                      <div className="flex items-center space-x-3">
                        {getFileIcon(file.name)}
                        <div>
                          <p className="font-medium text-neutral-900 dark:text-neutral-100 text-sm">
                            {file.name}
                          </p>
                          <p className="text-xs text-neutral-500">
                            {formatFileSize(file.size)}
                          </p>
                        </div>
                      </div>
                      
                      <button
                        onClick={() => removeFile(index)}
                        disabled={uploading}
                        className="p-1.5 rounded-lg hover:bg-error-100 dark:hover:bg-error-900/30 text-neutral-400 hover:text-error-600 dark:hover:text-error-400 transition-colors disabled:opacity-50"
                      >
                        <X className="w-4 h-4" />
                      </button>
                    </motion.div>
                  ))}
                </div>
              </div>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Dataset Configuration */}
      <Card variant="default">
        <div className="p-6">
          <h3 className="text-lg font-semibold mb-4 text-neutral-900 dark:text-neutral-100">
            Dataset Configuration
          </h3>
          
          <div className="space-y-4">
            {/* Dataset Option Toggle */}
            <div className="flex space-x-2 p-1 bg-neutral-800 rounded-xl">
              <button
                onClick={() => setUseExistingDataset(false)}
                disabled={uploading}
                className={`
                  flex-1 py-2.5 px-4 rounded-lg font-medium transition-all duration-200
                  ${!useExistingDataset 
                    ? 'bg-neutral-700 text-primary-400 shadow-sm' 
                    : 'text-neutral-400 hover:text-neutral-200'
                  }
                `}
              >
                <div className="flex items-center justify-center space-x-2">
                  <Plus className="w-4 h-4" />
                  <span>Create New Dataset</span>
                </div>
              </button>
              
              <button
                onClick={() => setUseExistingDataset(true)}
                disabled={uploading || datasets.length === 0}
                className={`
                  flex-1 py-2.5 px-4 rounded-lg font-medium transition-all duration-200
                  ${useExistingDataset 
                    ? 'bg-neutral-700 text-primary-400 shadow-sm' 
                    : 'text-neutral-400 hover:text-neutral-200'
                  }
                  ${datasets.length === 0 ? 'opacity-50 cursor-not-allowed' : ''}
                `}
              >
                <div className="flex items-center justify-center space-x-2">
                  <Folder className="w-4 h-4" />
                  <span>Add to Existing</span>
                </div>
              </button>
            </div>

            {/* Dataset Input/Select */}
            <AnimatePresence mode="wait">
              {useExistingDataset ? (
                <motion.div
                  key="existing"
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                >
                  <select
                    value={selectedDataset}
                    onChange={(e) => setSelectedDataset(e.target.value)}
                    disabled={uploading || datasets.length === 0}
                    className="w-full px-4 py-3 rounded-xl border border-neutral-700 bg-neutral-800 text-neutral-100 font-medium focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-all"
                  >
                    <option value="">Select a dataset...</option>
                    {datasets.map((dataset) => (
                      <option key={dataset.name} value={dataset.name}>
                        {dataset.name} ({dataset.document_count} documents)
                      </option>
                    ))}
                  </select>
                </motion.div>
              ) : (
                <motion.div
                  key="new"
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                >
                  <Input
                    label="Dataset Name"
                    placeholder="e.g., research-papers-2024"
                    value={datasetName}
                    onChange={(e) => setDatasetName(e.target.value)}
                    disabled={uploading}
                    leftIcon={<Sparkles className="w-4 h-4" />}
                  />
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </Card>

      {/* Error Alert */}
      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
          >
            <Card variant="outlined" className="border-error-200 dark:border-error-800 bg-error-50 dark:bg-error-900/20">
              <div className="p-4 flex items-start space-x-3">
                <AlertCircle className="w-5 h-5 text-error-600 dark:text-error-400 flex-shrink-0 mt-0.5" />
                <div className="flex-1">
                  <p className="text-error-800 dark:text-error-300 font-medium">Upload Error</p>
                  <p className="text-error-600 dark:text-error-400 text-sm mt-1">{error}</p>
                </div>
              </div>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Upload Progress */}
      <AnimatePresence>
        {uploading && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
          >
            <Card variant="gradient">
              <div className="p-6">
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center space-x-3">
                    <div className="w-10 h-10 bg-primary-100 dark:bg-primary-900/30 rounded-xl flex items-center justify-center">
                      <ArrowUpCircle className="w-6 h-6 text-primary-600 dark:text-primary-400 animate-pulse" />
                    </div>
                    <div>
                      <p className="font-medium text-neutral-900 dark:text-neutral-100">
                        Uploading and processing documents...
                      </p>
                      <p className="text-sm text-neutral-600 dark:text-neutral-400">
                        This may take a few moments
                      </p>
                    </div>
                  </div>
                  <Badge variant="primary" size="sm">
                    {uploadProgress}%
                  </Badge>
                </div>
                <Progress
                  value={uploadProgress}
                  variant="gradient"
                  size="lg"
                  animated
                />
              </div>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Success Message */}
      <AnimatePresence>
        {uploadSuccess && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
          >
            <Card variant="gradient" className="bg-gradient-to-r from-success-50 to-success-100 dark:from-success-900/20 dark:to-success-800/20 border-success-200 dark:border-success-800">
              <div className="p-6 text-center">
                <motion.div
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  transition={{ type: 'spring', stiffness: 500, damping: 30 }}
                  className="w-16 h-16 mx-auto mb-4 bg-success-500 rounded-full flex items-center justify-center"
                >
                  <CheckCircle className="w-10 h-10 text-white" />
                </motion.div>
                <h3 className="text-xl font-semibold text-success-900 dark:text-success-100 mb-2">
                  Upload Successful!
                </h3>
                <p className="text-success-700 dark:text-success-300">
                  Your documents have been processed and are ready for analysis
                </p>
              </div>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Upload Button */}
      <div className="flex justify-end">
        <Button
          variant="primary"
          size="lg"
          onClick={handleUpload}
          disabled={uploading || files.length === 0}
          isLoading={uploading}
          leftIcon={!uploading && <Upload className="w-5 h-5" />}
          className="min-w-[200px]"
        >
          {uploading ? 'Uploading...' : `Upload ${files.length} ${files.length === 1 ? 'File' : 'Files'}`}
        </Button>
      </div>
    </motion.div>
  );
};

export default UploadSection;