import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Search,
  Brain,
  ChevronDown,
  ChevronUp,
  Copy,
  Quote,
  Sparkles,
  FileText,
  AlertTriangle,
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { Dataset, QueryResult, UserRole } from '../types';
import api from '../services/api';
import Card from './ui/Card';
import Button from './ui/Button';
import Select from './ui/Select';
import TextArea from './ui/TextArea';
import Alert from './ui/Alert';
import Badge from './ui/Badge';

interface QuerySectionProps {
  datasets: Dataset[];
  onQueryComplete: (result: QueryResult) => void;
  previousResult: QueryResult | null;
}

const QuerySection: React.FC<QuerySectionProps> = ({
  datasets,
  onQueryComplete,
  previousResult,
}) => {
  const [query, setQuery] = useState('');
  const [selectedDatasets, setSelectedDatasets] = useState<string[]>([]);
  const [role, setRole] = useState<UserRole>('general');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<QueryResult | null>(previousResult);
  const [expandedCitations, setExpandedCitations] = useState<Set<number>>(new Set());

  const roleOptions = [
    { value: 'general', label: 'General', description: 'General purpose analysis' },
    { value: 'doctor', label: 'Doctor', description: 'Medical professional context with health disclaimers' },
    { value: 'lawyer', label: 'Lawyer', description: 'Legal professional context with legal disclaimers' },
    { value: 'hr', label: 'HR', description: 'HR professional context with compliance focus' },
  ];

  const exampleQueries: Record<UserRole, string[]> = {
    general: [
      'Summarize the key findings from the documents',
      'What are the main themes discussed?',
    ],
    doctor: [
      'What are the patient history and current medications?',
      'Identify potential drug interactions',
    ],
    lawyer: [
      'Find precedents related to breach of contract',
      'What are the key legal arguments presented?',
    ],
    hr: [
      'Compare policy changes between versions',
      'What are the compliance requirements?',
    ],
  };

  const handleQuery = async () => {
    if (!query.trim()) {
      setError('Please enter a query');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await api.queryDocuments(
        query,
        selectedDatasets.length > 0 ? selectedDatasets : undefined,
        role
      );

      setResult(response.data);
      onQueryComplete(response.data);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Query failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const toggleCitation = (index: number) => {
    const newExpanded = new Set(expandedCitations);
    if (newExpanded.has(index)) {
      newExpanded.delete(index);
    } else {
      newExpanded.add(index);
    }
    setExpandedCitations(newExpanded);
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
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
          Query Documents
        </h1>
        <p className="text-neutral-600 dark:text-neutral-400">
          Ask questions about your documents and get AI-generated answers with citations.
        </p>
      </div>

      {/* Query Form */}
      <Card variant="gradient" glow>
        <div className="p-8 space-y-6">
          {/* Dataset Selection */}
          <div>
            <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-1.5">
              Datasets (Optional)
            </label>
            <select
              multiple
              value={selectedDatasets}
              onChange={(e) => {
                const selected = Array.from(e.target.selectedOptions, option => option.value);
                setSelectedDatasets(selected);
              }}
              className="w-full px-4 py-3 rounded-xl border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent min-h-[100px]"
            >
              {datasets.map((dataset) => (
                <option key={dataset.name} value={dataset.name}>
                  {dataset.name} ({dataset.document_count} docs)
                </option>
              ))}
            </select>
            {selectedDatasets.length > 0 && (
              <div className="mt-2 flex flex-wrap gap-2">
                {selectedDatasets.map((name) => (
                  <Badge key={name} variant="primary" size="sm">
                    {name}
                  </Badge>
                ))}
              </div>
            )}
          </div>

          {/* Role Selection */}
          <Select
            label="Role Context"
            options={roleOptions}
            value={role}
            onChange={(value) => setRole(value as UserRole)}
          />

          {/* Query Input */}
          <TextArea
            label="Your Question"
            placeholder="Enter your question here..."
            rows={4}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />

          {/* Example Queries */}
          <div>
            <p className="text-sm text-neutral-600 dark:text-neutral-400 mb-2">
              Example queries for {role}:
            </p>
            <div className="flex flex-wrap gap-2">
              {exampleQueries[role].map((example, index) => (
                <button
                  key={index}
                  onClick={() => setQuery(example)}
                  className="px-3 py-1.5 text-sm rounded-lg border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-800 text-neutral-700 dark:text-neutral-300 hover:bg-primary-50 dark:hover:bg-primary-900/20 hover:border-primary-400 dark:hover:border-primary-600 transition-all"
                >
                  {example}
                </button>
              ))}
            </div>
          </div>

          {/* Error Alert */}
          <AnimatePresence>
            {error && (
              <motion.div
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
              >
                <Alert variant="error" onClose={() => setError(null)}>
                  {error}
                </Alert>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Search Button */}
          <Button
            variant="primary"
            size="lg"
            onClick={handleQuery}
            disabled={loading || !query.trim()}
            isLoading={loading}
            leftIcon={!loading && <Search className="w-5 h-5" />}
            className="w-full"
          >
            {loading ? 'Searching...' : 'Search Documents'}
          </Button>
        </div>
      </Card>

      {/* Results */}
      <AnimatePresence>
        {result && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 20 }}
          >
            <Card variant="default">
              <div className="p-6">
                {/* Answer Header */}
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center space-x-2">
                    <div className="w-10 h-10 bg-gradient-to-br from-primary-400 to-accent-400 rounded-xl flex items-center justify-center">
                      <Brain className="w-6 h-6 text-white" />
                    </div>
                    <h2 className="text-xl font-semibold text-neutral-900 dark:text-neutral-100">
                      Answer
                    </h2>
                  </div>
                  <Badge
                    variant={
                      result.confidence === 'high'
                        ? 'success'
                        : result.confidence === 'medium'
                        ? 'warning'
                        : 'error'
                    }
                  >
                    Confidence: {result.confidence}
                  </Badge>
                </div>

                {/* Disclaimer */}
                {result.disclaimer && (
                  <Alert variant="warning" className="mb-4">
                    {result.disclaimer}
                  </Alert>
                )}

                {/* Answer Content */}
                <div className="bg-neutral-50 dark:bg-neutral-800 rounded-xl p-4 mb-6">
                  <div className="prose prose-neutral dark:prose-invert max-w-none">
                    <ReactMarkdown>{result.answer}</ReactMarkdown>
                  </div>
                  <button
                    onClick={() => copyToClipboard(result.answer)}
                    className="mt-3 p-2 rounded-lg hover:bg-neutral-100 dark:hover:bg-neutral-700 text-neutral-500 hover:text-neutral-700 dark:hover:text-neutral-300 transition-colors"
                  >
                    <Copy className="w-4 h-4" />
                  </button>
                </div>

                {/* Highlights */}
                {result.highlights && result.highlights.length > 0 && (
                  <div className="mb-6">
                    <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 mb-3">
                      Key Highlights
                    </h3>
                    <div className="space-y-2">
                      {result.highlights.map((highlight, index) => (
                        <div
                          key={index}
                          className="flex items-start space-x-2 p-3 rounded-lg bg-primary-50 dark:bg-primary-900/20"
                        >
                          <Quote className="w-5 h-5 text-primary-600 dark:text-primary-400 flex-shrink-0 mt-0.5" />
                          <p className="text-sm text-neutral-700 dark:text-neutral-300">
                            {highlight}
                          </p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Citations */}
                <div>
                  <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 mb-3">
                    Sources & Citations ({result.citations.length})
                  </h3>
                  <div className="space-y-3">
                    {result.citations.map((citation, index) => (
                      <div
                        key={index}
                        className="border border-neutral-200 dark:border-neutral-700 rounded-xl overflow-hidden"
                      >
                        <button
                          onClick={() => toggleCitation(index)}
                          className="w-full px-4 py-3 flex items-center justify-between bg-white dark:bg-neutral-800 hover:bg-neutral-50 dark:hover:bg-neutral-700 transition-colors"
                        >
                          <div className="flex items-start space-x-3">
                            <FileText className="w-5 h-5 text-primary-500 mt-0.5" />
                            <div className="text-left">
                              <p className="font-medium text-neutral-900 dark:text-neutral-100">
                                [Source {citation.source_number}] {citation.filename}
                              </p>
                              <p className="text-xs text-neutral-500 dark:text-neutral-400 mt-1">
                                Dataset: {citation.dataset} | Chunk: {citation.chunk_index} | 
                                Relevance: {(citation.relevance_score * 100).toFixed(1)}%
                              </p>
                            </div>
                          </div>
                          {expandedCitations.has(index) ? (
                            <ChevronUp className="w-5 h-5 text-neutral-400" />
                          ) : (
                            <ChevronDown className="w-5 h-5 text-neutral-400" />
                          )}
                        </button>
                        <AnimatePresence>
                          {expandedCitations.has(index) && (
                            <motion.div
                              initial={{ height: 0 }}
                              animate={{ height: 'auto' }}
                              exit={{ height: 0 }}
                              className="overflow-hidden"
                            >
                              <div className="px-4 py-3 bg-neutral-50 dark:bg-neutral-900 border-t border-neutral-200 dark:border-neutral-700">
                                <p className="text-sm italic text-neutral-600 dark:text-neutral-400">
                                  "{citation.snippet}"
                                </p>
                              </div>
                            </motion.div>
                          )}
                        </AnimatePresence>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Suggested Follow-up */}
                {result.suggested_followup && (
                  <div className="mt-6 p-4 rounded-xl bg-primary-50 dark:bg-primary-900/20 border border-primary-200 dark:border-primary-800">
                    <div className="flex items-start space-x-2">
                      <Sparkles className="w-5 h-5 text-primary-600 dark:text-primary-400 mt-0.5" />
                      <div>
                        <p className="text-sm font-medium text-primary-900 dark:text-primary-100 mb-1">
                          Suggested follow-up:
                        </p>
                        <button
                          onClick={() => setQuery(result.suggested_followup!)}
                          className="text-sm text-primary-700 dark:text-primary-300 hover:text-primary-900 dark:hover:text-primary-100 underline"
                        >
                          {result.suggested_followup}
                        </button>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
};

export default QuerySection;