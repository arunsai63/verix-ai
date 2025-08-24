import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Activity,
  CheckCircle,
  AlertCircle,
  Clock,
  XCircle,
  RefreshCw,
  Trash2,
  FileText,
  TrendingUp,
  Loader
} from 'lucide-react';
import api from '../services/api';
import Card from './ui/Card';
import Button from './ui/Button';
import Badge from './ui/Badge';
import Progress from './ui/Progress';

interface Job {
  id: string;
  status: string;
  dataset_name: string;
  total_files: number;
  documents_processed: number;
  documents_failed: number;
  chunks_created: number;
  progress: number;
  created_at: string;
  updated_at: string;
  started_at?: string;
  completed_at?: string;
  error?: string;
  estimated_time_remaining?: number;
}

const JobMonitor: React.FC = () => {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [activeJobs, setActiveJobs] = useState<Job[]>([]);
  const [metrics, setMetrics] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [selectedJob, setSelectedJob] = useState<Job | null>(null);
  const [jobDetails, setJobDetails] = useState<any>(null);

  useEffect(() => {
    loadJobs();
    loadMetrics();
    const interval = setInterval(() => {
      loadActiveJobs();
    }, 3000);
    return () => clearInterval(interval);
  }, []);

  const loadJobs = async () => {
    try {
      setLoading(true);
      const response = await api.getAllJobs(50, 0);
      setJobs(response.data.jobs);
    } catch (error) {
      console.error('Error loading jobs:', error);
    } finally {
      setLoading(false);
    }
  };

  const loadActiveJobs = async () => {
    try {
      const response = await api.getActiveJobs();
      setActiveJobs(response.data.active_jobs);
    } catch (error) {
      console.error('Error loading active jobs:', error);
    }
  };

  const loadMetrics = async () => {
    try {
      const response = await api.getJobMetrics();
      setMetrics(response.data);
    } catch (error) {
      console.error('Error loading metrics:', error);
    }
  };

  const handleRefresh = async () => {
    setRefreshing(true);
    await Promise.all([loadJobs(), loadActiveJobs(), loadMetrics()]);
    setRefreshing(false);
  };

  const handleCancelJob = async (jobId: string) => {
    try {
      await api.cancelJob(jobId);
      await loadJobs();
      await loadActiveJobs();
    } catch (error) {
      console.error('Error cancelling job:', error);
    }
  };

  const loadJobDetails = async (job: Job) => {
    try {
      const response = await api.getJobStatus(job.id);
      setJobDetails(response.data);
      setSelectedJob(job);
    } catch (error) {
      console.error('Error loading job details:', error);
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="w-5 h-5 text-success-500" />;
      case 'processing':
      case 'queued':
        return <Loader className="w-5 h-5 text-primary-500 animate-spin" />;
      case 'failed':
        return <XCircle className="w-5 h-5 text-error-500" />;
      case 'timeout':
        return <Clock className="w-5 h-5 text-warning-500" />;
      case 'cancelled':
        return <AlertCircle className="w-5 h-5 text-neutral-500" />;
      default:
        return <Activity className="w-5 h-5 text-neutral-400" />;
    }
  };

  const getStatusBadge = (status: string) => {
    const variants: Record<string, 'success' | 'primary' | 'error' | 'warning' | 'secondary'> = {
      completed: 'success',
      completed_with_errors: 'warning',
      processing: 'primary',
      queued: 'secondary',
      failed: 'error',
      timeout: 'warning',
      cancelled: 'secondary'
    };
    return (
      <Badge variant={variants[status] || 'secondary'} size="sm">
        {status.replace('_', ' ')}
      </Badge>
    );
  };

  const formatTime = (timestamp: string) => {
    return new Date(timestamp).toLocaleString();
  };

  const formatDuration = (seconds: number) => {
    if (seconds < 60) return `${Math.round(seconds)}s`;
    const minutes = Math.floor(seconds / 60);
    if (minutes < 60) return `${minutes}m ${seconds % 60}s`;
    const hours = Math.floor(minutes / 60);
    return `${hours}h ${minutes % 60}m`;
  };

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-display font-bold text-neutral-900 dark:text-neutral-100 mb-2">
            Job Monitor
          </h1>
          <p className="text-neutral-600 dark:text-neutral-400">
            Track and manage document processing jobs
          </p>
        </div>
        <Button
          variant="secondary"
          size="md"
          onClick={handleRefresh}
          leftIcon={<RefreshCw className={`w-4 h-4 ${refreshing ? 'animate-spin' : ''}`} />}
          disabled={refreshing}
        >
          Refresh
        </Button>
      </div>

      {/* Metrics */}
      {metrics && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <Card variant="gradient">
            <div className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-neutral-600 dark:text-neutral-400">Total Jobs</p>
                  <p className="text-2xl font-bold text-neutral-900 dark:text-neutral-100">
                    {metrics.total_jobs}
                  </p>
                </div>
                <FileText className="w-8 h-8 text-primary-500 opacity-50" />
              </div>
            </div>
          </Card>

          <Card variant="gradient">
            <div className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-neutral-600 dark:text-neutral-400">Active Jobs</p>
                  <p className="text-2xl font-bold text-neutral-900 dark:text-neutral-100">
                    {metrics.active_jobs}
                  </p>
                </div>
                <Activity className="w-8 h-8 text-success-500 opacity-50" />
              </div>
            </div>
          </Card>

          <Card variant="gradient">
            <div className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-neutral-600 dark:text-neutral-400">Success Rate</p>
                  <p className="text-2xl font-bold text-neutral-900 dark:text-neutral-100">
                    {metrics.status_distribution?.completed 
                      ? Math.round((metrics.status_distribution.completed / metrics.total_jobs) * 100)
                      : 0}%
                  </p>
                </div>
                <TrendingUp className="w-8 h-8 text-accent-500 opacity-50" />
              </div>
            </div>
          </Card>

          <Card variant="gradient">
            <div className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-neutral-600 dark:text-neutral-400">Failed Jobs</p>
                  <p className="text-2xl font-bold text-neutral-900 dark:text-neutral-100">
                    {metrics.status_distribution?.failed || 0}
                  </p>
                </div>
                <AlertCircle className="w-8 h-8 text-error-500 opacity-50" />
              </div>
            </div>
          </Card>
        </div>
      )}

      {/* Active Jobs */}
      {activeJobs.length > 0 && (
        <Card variant="default">
          <div className="p-6">
            <h2 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 mb-4">
              Active Jobs
            </h2>
            <div className="space-y-3">
              {activeJobs.map((job) => (
                <motion.div
                  key={job.id}
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="p-4 rounded-xl bg-neutral-50 dark:bg-neutral-800"
                >
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center space-x-3">
                      {getStatusIcon(job.status)}
                      <div>
                        <p className="font-medium text-neutral-900 dark:text-neutral-100">
                          {job.dataset_name}
                        </p>
                        <p className="text-sm text-neutral-600 dark:text-neutral-400">
                          {job.documents_processed}/{job.total_files} files processed
                        </p>
                      </div>
                    </div>
                    <div className="flex items-center space-x-2">
                      {job.estimated_time_remaining && (
                        <Badge variant="secondary" size="sm">
                          ~{formatDuration(job.estimated_time_remaining)} remaining
                        </Badge>
                      )}
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => handleCancelJob(job.id)}
                      >
                        Cancel
                      </Button>
                    </div>
                  </div>
                  <Progress value={job.progress} variant="gradient" size="md" animated />
                </motion.div>
              ))}
            </div>
          </div>
        </Card>
      )}

      {/* All Jobs */}
      <Card variant="default">
        <div className="p-6">
          <h2 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 mb-4">
            Recent Jobs
          </h2>
          
          {loading ? (
            <div className="flex items-center justify-center py-12">
              <Loader className="w-8 h-8 text-primary-500 animate-spin" />
            </div>
          ) : jobs.length === 0 ? (
            <div className="text-center py-12">
              <FileText className="w-12 h-12 text-neutral-400 mx-auto mb-3" />
              <p className="text-neutral-600 dark:text-neutral-400">No jobs found</p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-neutral-200 dark:border-neutral-700">
                    <th className="text-left py-3 px-4 text-sm font-medium text-neutral-600 dark:text-neutral-400">
                      Dataset
                    </th>
                    <th className="text-left py-3 px-4 text-sm font-medium text-neutral-600 dark:text-neutral-400">
                      Status
                    </th>
                    <th className="text-left py-3 px-4 text-sm font-medium text-neutral-600 dark:text-neutral-400">
                      Progress
                    </th>
                    <th className="text-left py-3 px-4 text-sm font-medium text-neutral-600 dark:text-neutral-400">
                      Files
                    </th>
                    <th className="text-left py-3 px-4 text-sm font-medium text-neutral-600 dark:text-neutral-400">
                      Chunks
                    </th>
                    <th className="text-left py-3 px-4 text-sm font-medium text-neutral-600 dark:text-neutral-400">
                      Created
                    </th>
                    <th className="text-left py-3 px-4 text-sm font-medium text-neutral-600 dark:text-neutral-400">
                      Actions
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {jobs.map((job) => (
                    <tr
                      key={job.id}
                      className="border-b border-neutral-100 dark:border-neutral-800 hover:bg-neutral-50 dark:hover:bg-neutral-800/50 transition-colors"
                    >
                      <td className="py-3 px-4">
                        <p className="font-medium text-neutral-900 dark:text-neutral-100">
                          {job.dataset_name}
                        </p>
                        <p className="text-xs text-neutral-500 font-mono">
                          {job.id.slice(0, 8)}...
                        </p>
                      </td>
                      <td className="py-3 px-4">
                        {getStatusBadge(job.status)}
                      </td>
                      <td className="py-3 px-4">
                        <div className="w-24">
                          <Progress value={job.progress} variant="primary" size="sm" />
                        </div>
                      </td>
                      <td className="py-3 px-4">
                        <p className="text-sm text-neutral-700 dark:text-neutral-300">
                          {job.documents_processed}/{job.total_files}
                        </p>
                        {job.documents_failed > 0 && (
                          <p className="text-xs text-error-600 dark:text-error-400">
                            {job.documents_failed} failed
                          </p>
                        )}
                      </td>
                      <td className="py-3 px-4">
                        <p className="text-sm text-neutral-700 dark:text-neutral-300">
                          {job.chunks_created}
                        </p>
                      </td>
                      <td className="py-3 px-4">
                        <p className="text-sm text-neutral-700 dark:text-neutral-300">
                          {new Date(job.created_at).toLocaleDateString()}
                        </p>
                        <p className="text-xs text-neutral-500">
                          {new Date(job.created_at).toLocaleTimeString()}
                        </p>
                      </td>
                      <td className="py-3 px-4">
                        <div className="flex items-center space-x-1">
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => loadJobDetails(job)}
                          >
                            Details
                          </Button>
                          {job.status === 'processing' && (
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => handleCancelJob(job.id)}
                            >
                              Cancel
                            </Button>
                          )}
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </Card>

      {/* Job Details Modal */}
      <AnimatePresence>
        {selectedJob && jobDetails && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4"
            onClick={() => setSelectedJob(null)}
          >
            <motion.div
              initial={{ scale: 0.95 }}
              animate={{ scale: 1 }}
              exit={{ scale: 0.95 }}
              className="bg-white dark:bg-neutral-900 rounded-2xl max-w-3xl w-full max-h-[80vh] overflow-auto"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="p-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-xl font-semibold text-neutral-900 dark:text-neutral-100">
                    Job Details
                  </h3>
                  <button
                    onClick={() => setSelectedJob(null)}
                    className="p-2 rounded-lg hover:bg-neutral-100 dark:hover:bg-neutral-800 transition-colors"
                  >
                    <XCircle className="w-5 h-5 text-neutral-500" />
                  </button>
                </div>

                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <p className="text-sm text-neutral-600 dark:text-neutral-400">Job ID</p>
                      <p className="font-mono text-sm text-neutral-900 dark:text-neutral-100">
                        {jobDetails.id}
                      </p>
                    </div>
                    <div>
                      <p className="text-sm text-neutral-600 dark:text-neutral-400">Status</p>
                      {getStatusBadge(jobDetails.status)}
                    </div>
                    <div>
                      <p className="text-sm text-neutral-600 dark:text-neutral-400">Dataset</p>
                      <p className="text-neutral-900 dark:text-neutral-100">
                        {jobDetails.dataset_name}
                      </p>
                    </div>
                    <div>
                      <p className="text-sm text-neutral-600 dark:text-neutral-400">Progress</p>
                      <Progress value={jobDetails.progress} variant="primary" size="sm" />
                    </div>
                  </div>

                  {jobDetails.error && (
                    <div className="p-4 rounded-lg bg-error-50 dark:bg-error-900/20 border border-error-200 dark:border-error-800">
                      <p className="text-sm font-medium text-error-800 dark:text-error-300 mb-1">
                        Error
                      </p>
                      <p className="text-sm text-error-700 dark:text-error-400">
                        {jobDetails.error}
                      </p>
                    </div>
                  )}

                  {jobDetails.files && jobDetails.files.length > 0 && (
                    <div>
                      <p className="text-sm font-medium text-neutral-900 dark:text-neutral-100 mb-2">
                        Files
                      </p>
                      <div className="space-y-1 max-h-48 overflow-y-auto">
                        {jobDetails.files.map((file: any) => (
                          <div
                            key={file.name}
                            className="flex items-center justify-between p-2 rounded bg-neutral-50 dark:bg-neutral-800"
                          >
                            <div className="flex items-center space-x-2">
                              {getStatusIcon(file.status)}
                              <span className="text-sm text-neutral-700 dark:text-neutral-300">
                                {file.name}
                              </span>
                            </div>
                            <span className="text-xs text-neutral-500">
                              {file.chunks || 0} chunks
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {jobDetails.logs && jobDetails.logs.length > 0 && (
                    <div>
                      <p className="text-sm font-medium text-neutral-900 dark:text-neutral-100 mb-2">
                        Logs
                      </p>
                      <div className="space-y-1 max-h-48 overflow-y-auto font-mono text-xs bg-neutral-900 text-neutral-100 p-3 rounded-lg">
                        {jobDetails.logs.map((log: any, idx: number) => (
                          <div key={idx}>
                            <span className="text-neutral-500">
                              {new Date(log.timestamp).toLocaleTimeString()}
                            </span>
                            {' '}
                            <span className={
                              log.level === 'error' ? 'text-error-400' :
                              log.level === 'warning' ? 'text-warning-400' :
                              'text-neutral-300'
                            }>
                              [{log.level}]
                            </span>
                            {' '}
                            <span>{log.message}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default JobMonitor;