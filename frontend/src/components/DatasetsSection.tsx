import React, { useState } from 'react';
import {
  Box,
  Button,
  Typography,
  Card,
  CardContent,
  CardActions,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  DialogContentText,
  Chip,
  IconButton,
  LinearProgress,
  CircularProgress,
  Grid,
} from '@mui/material';
import {
  Delete,
  Refresh,
  Folder,
  Description,
  Storage as StorageIcon,
  Info,
} from '@mui/icons-material';
import { Dataset } from '../types';
import api from '../services/api';

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
      <Box textAlign="center" py={8}>
        <StorageIcon sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
        <Typography variant="h6" color="text.secondary" gutterBottom>
          No Datasets Yet
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
          Upload documents to create your first dataset
        </Typography>
        <Button variant="outlined" startIcon={<Refresh />} onClick={onRefresh}>
          Refresh
        </Button>
      </Box>
    );
  }

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h5">
          Manage Datasets ({datasets.length})
        </Typography>
        <Button variant="outlined" startIcon={<Refresh />} onClick={onRefresh}>
          Refresh
        </Button>
      </Box>

      <Grid container spacing={3}>
        {datasets.map((dataset) => (
          <Grid item xs={12} md={6} lg={4} key={dataset.name}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <Folder sx={{ mr: 1, color: 'primary.main' }} />
                  <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
                    {dataset.name}
                  </Typography>
                </Box>

                <Box sx={{ mb: 2 }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                    <Description sx={{ fontSize: 16, mr: 1, color: 'text.secondary' }} />
                    <Typography variant="body2" color="text.secondary">
                      {dataset.document_count} documents
                    </Typography>
                  </Box>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                    <StorageIcon sx={{ fontSize: 16, mr: 1, color: 'text.secondary' }} />
                    <Typography variant="body2" color="text.secondary">
                      {formatBytes(dataset.size_bytes)}
                    </Typography>
                  </Box>
                  <Typography variant="caption" color="text.secondary">
                    Created: {formatDate(dataset.created_at)}
                  </Typography>
                </Box>

                {loadingStats === dataset.name && <LinearProgress sx={{ mb: 2 }} />}

                {datasetStats[dataset.name] && (
                  <Box sx={{ mb: 2 }}>
                    <Chip
                      label={`${datasetStats[dataset.name].total_chunks} chunks`}
                      size="small"
                      sx={{ mr: 1 }}
                    />
                    {datasetStats[dataset.name].file_types?.map((type: string) => (
                      <Chip key={type} label={type} size="small" variant="outlined" sx={{ mr: 0.5, mb: 0.5 }} />
                    ))}
                  </Box>
                )}
              </CardContent>
              <CardActions>
                <IconButton
                  size="small"
                  onClick={() => loadDatasetStats(dataset.name)}
                  disabled={loadingStats === dataset.name || !!datasetStats[dataset.name]}
                >
                  <Info />
                </IconButton>
                <Box sx={{ flexGrow: 1 }} />
                <Button
                  size="small"
                  color="error"
                  startIcon={<Delete />}
                  onClick={() => handleDeleteClick(dataset)}
                >
                  Delete
                </Button>
              </CardActions>
            </Card>
          </Grid>
        ))}
      </Grid>

      <Dialog open={deleteDialog.open} onClose={handleDeleteCancel}>
        <DialogTitle>Delete Dataset</DialogTitle>
        <DialogContent>
          <DialogContentText>
            Are you sure you want to delete the dataset "{deleteDialog.dataset?.name}"? This will
            permanently remove all {deleteDialog.dataset?.document_count} documents and cannot be
            undone.
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleDeleteCancel} disabled={deleting}>
            Cancel
          </Button>
          <Button
            onClick={handleDeleteConfirm}
            color="error"
            variant="contained"
            disabled={deleting}
            startIcon={deleting ? <CircularProgress size={20} /> : <Delete />}
          >
            {deleting ? 'Deleting...' : 'Delete'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default DatasetsSection;