import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Trash2,
  RefreshCw,
  Folder,
  FileText,
  HardDrive,
  Info,
  Clock,
  Database,
  AlertTriangle,
} from 'lucide-react';
import { Dataset } from '../types';
import api from '../services/api';
import Card from './ui/Card';
import Button from './ui/Button';
import Badge from './ui/Badge';
import Dialog from './ui/Dialog';
import Progress from './ui/Progress';

interface DatasetsSectionProps {
  datasets: Dataset[];
  onDatasetDeleted: () => void;
  onRefresh: () => void;
}

const DatasetsSection: React.FC<DatasetsSectionProps> = ({
  datasets,
  onDatasetDeleted,
  onRefresh,
}) => {
  const [deleteDialog, setDeleteDialog] = useState<{
    open: boolean;
    dataset: Dataset | null;
  }>({
    open: false,
    dataset: null,
  });
  const [deleting, setDeleting] = useState(false);
  const [loadingStats, setLoadingStats] = useState<string | null>(null);
  const [datasetStats, setDatasetStats] = useState<Record<string, any>>({});

  const formatBytes = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + ' ' + sizes[i];
  };

  const formatDate = (timestamp: number) => {
    return new Date(timestamp * 1000).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  const handleDeleteClick = (dataset: Dataset) => {
    setDeleteDialog({ open: true, dataset });
  };

  const handleDeleteConfirm = async () => {
    if (!deleteDialog.dataset) return;

    setDeleting(true);
    try {
      await api.deleteDataset(deleteDialog.dataset.name);
      onDatasetDeleted();
      setDeleteDialog({ open: false, dataset: null });
    } catch (error) {
      console.error('Failed to delete dataset:', error);
    } finally {
      setDeleting(false);
    }
  };

  const handleDeleteCancel = () => {
    setDeleteDialog({ open: false, dataset: null });
  };

  const loadDatasetStats = async (datasetName: string) => {
    if (datasetStats[datasetName]) return;

    setLoadingStats(datasetName);
    try {
      const response = await api.getDatasetStats(datasetName);
      setDatasetStats((prev) => ({
        ...prev,
        [datasetName]: response.data,
      }));
    } catch (error) {
      console.error('Failed to load dataset stats:', error);
    } finally {
      setLoadingStats(null);
    }
  };

  if (datasets.length === 0) {
    return (
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="max-w-4xl mx-auto"
      >
        <Card variant="default">
          <div className="p-12 text-center">
            <div className="w-20 h-20 mx-auto mb-6 bg-neutral-100 dark:bg-neutral-800 rounded-2xl flex items-center justify-center">
              <Database className="w-10 h-10 text-neutral-400" />
            </div>
            <h3 className="text-xl font-semibold text-neutral-900 dark:text-neutral-100 mb-2">
              No Datasets Yet
            </h3>
            <p className="text-neutral-600 dark:text-neutral-400 mb-6">
              Upload documents to create your first dataset
            </p>
            <Button
              variant="outline"
              leftIcon={<RefreshCw className="w-4 h-4" />}
              onClick={onRefresh}
            >
              Refresh
            </Button>
          </div>
        </Card>
      </motion.div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="max-w-6xl mx-auto space-y-6"
    >
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-display font-bold text-neutral-900 dark:text-neutral-100 mb-2">
            Manage Datasets
          </h1>
          <p className="text-neutral-600 dark:text-neutral-400">
            {datasets.length} {datasets.length === 1 ? 'dataset' : 'datasets'} available
          </p>
        </div>
        <Button
          variant="outline"
          leftIcon={<RefreshCw className="w-4 h-4" />}
          onClick={onRefresh}
        >
          Refresh
        </Button>
      </div>

      {/* Dataset Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {datasets.map((dataset) => (
          <motion.div
            key={dataset.name}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
          >
            <Card variant="default" className="h-full">
              <div className="p-6 flex flex-col h-full">
                {/* Dataset Header */}
                <div className="flex items-start justify-between mb-4">
                  <div className="flex items-center space-x-3">
                    <div className="w-10 h-10 bg-primary-100 dark:bg-primary-900/30 rounded-xl flex items-center justify-center">
                      <Folder className="w-6 h-6 text-primary-600 dark:text-primary-400" />
                    </div>
                    <div>
                      <h3 className="font-semibold text-neutral-900 dark:text-neutral-100 truncate max-w-[180px]">
                        {dataset.name}
                      </h3>
                      <p className="text-xs text-neutral-500 dark:text-neutral-400 mt-0.5">
                        <Clock className="w-3 h-3 inline mr-1" />
                        {formatDate(dataset.created_at)}
                      </p>
                    </div>
                  </div>
                </div>

                {/* Dataset Info */}
                <div className="space-y-3 mb-4 flex-grow">
                  <div className="flex items-center justify-between py-2 px-3 rounded-lg bg-neutral-50 dark:bg-neutral-800">
                    <div className="flex items-center space-x-2">
                      <FileText className="w-4 h-4 text-neutral-500" />
                      <span className="text-sm text-neutral-600 dark:text-neutral-400">
                        Documents
                      </span>
                    </div>
                    <span className="text-sm font-medium text-neutral-900 dark:text-neutral-100">
                      {dataset.document_count}
                    </span>
                  </div>
                  
                  <div className="flex items-center justify-between py-2 px-3 rounded-lg bg-neutral-50 dark:bg-neutral-800">
                    <div className="flex items-center space-x-2">
                      <HardDrive className="w-4 h-4 text-neutral-500" />
                      <span className="text-sm text-neutral-600 dark:text-neutral-400">
                        Size
                      </span>
                    </div>
                    <span className="text-sm font-medium text-neutral-900 dark:text-neutral-100">
                      {formatBytes(dataset.size_bytes)}
                    </span>
                  </div>
                </div>

                {/* Loading Stats */}
                {loadingStats === dataset.name && (
                  <div className="mb-4">
                    <Progress value={50} size="sm" animated />
                  </div>
                )}

                {/* Dataset Stats */}
                {datasetStats[dataset.name] && (
                  <div className="mb-4 p-3 rounded-lg bg-primary-50 dark:bg-primary-900/20 border border-primary-200 dark:border-primary-800">
                    <div className="flex flex-wrap gap-2">
                      <Badge variant="primary" size="sm">
                        {datasetStats[dataset.name].total_chunks} chunks
                      </Badge>
                      {datasetStats[dataset.name].file_types?.map((type: string) => (
                        <Badge key={type} variant="outline" size="sm">
                          {type}
                        </Badge>
                      ))}
                    </div>
                  </div>
                )}

                {/* Actions */}
                <div className="flex items-center justify-between pt-4 border-t border-neutral-200 dark:border-neutral-700">
                  <button
                    onClick={() => loadDatasetStats(dataset.name)}
                    disabled={loadingStats === dataset.name || !!datasetStats[dataset.name]}
                    className="p-2 rounded-lg hover:bg-neutral-100 dark:hover:bg-neutral-800 text-neutral-500 hover:text-neutral-700 dark:hover:text-neutral-300 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    <Info className="w-4 h-4" />
                  </button>
                  <Button
                    variant="ghost"
                    size="sm"
                    leftIcon={<Trash2 className="w-4 h-4" />}
                    onClick={() => handleDeleteClick(dataset)}
                    className="text-error-600 hover:text-error-700 dark:text-error-400 dark:hover:text-error-300"
                  >
                    Delete
                  </Button>
                </div>
              </div>
            </Card>
          </motion.div>
        ))}
      </div>

      {/* Delete Dialog */}
      <Dialog
        open={deleteDialog.open}
        onClose={handleDeleteCancel}
        title="Delete Dataset"
        footer={
          <div className="flex justify-end space-x-3">
            <Button
              variant="outline"
              onClick={handleDeleteCancel}
              disabled={deleting}
            >
              Cancel
            </Button>
            <Button
              variant="primary"
              onClick={handleDeleteConfirm}
              disabled={deleting}
              isLoading={deleting}
              leftIcon={!deleting && <Trash2 className="w-4 h-4" />}
              className="bg-error-600 hover:bg-error-700"
            >
              {deleting ? 'Deleting...' : 'Delete'}
            </Button>
          </div>
        }
      >
        <div className="space-y-4">
          <div className="flex items-start space-x-3">
            <div className="w-10 h-10 bg-error-100 dark:bg-error-900/30 rounded-xl flex items-center justify-center flex-shrink-0">
              <AlertTriangle className="w-6 h-6 text-error-600 dark:text-error-400" />
            </div>
            <div>
              <p className="text-neutral-700 dark:text-neutral-300">
                Are you sure you want to delete the dataset <strong>"{deleteDialog.dataset?.name}"</strong>?
              </p>
              <p className="text-sm text-neutral-600 dark:text-neutral-400 mt-2">
                This will permanently remove all {deleteDialog.dataset?.document_count} documents and cannot be undone.
              </p>
            </div>
          </div>
        </div>
      </Dialog>
    </motion.div>
  );
};

export default DatasetsSection;