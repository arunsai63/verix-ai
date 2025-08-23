import React, { useState } from 'react';
import {
  Box,
  Button,
  TextField,
  Typography,
  Card,
  CardContent,
  Chip,
  Stack,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Alert,
  CircularProgress,
  Divider,
  Paper,
  IconButton,
  Collapse,
  SelectChangeEvent,
  OutlinedInput,
} from '@mui/material';
import {
  Search,
  Psychology,
  ExpandMore,
  ExpandLess,
  ContentCopy,
  FormatQuote,
} from '@mui/icons-material';
import ReactMarkdown from 'react-markdown';
import { Dataset, QueryResult, UserRole } from '../types';
import api from '../services/api';

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
  const [maxResults, setMaxResults] = useState(10);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<QueryResult | null>(previousResult);
  const [expandedCitations, setExpandedCitations] = useState<Set<number>>(new Set());

  const roleDescriptions: Record<UserRole, string> = {
    general: 'General purpose analysis',
    doctor: 'Medical professional context with health disclaimers',
    lawyer: 'Legal professional context with legal disclaimers',
    hr: 'HR professional context with compliance focus',
  };

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
        role,
        maxResults
      );

      setResult(response.data);
      onQueryComplete(response.data);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Query failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleDatasetChange = (event: SelectChangeEvent<string[]>) => {
    const value = event.target.value;
    setSelectedDatasets(typeof value === 'string' ? value.split(',') : value);
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

  const handleExampleQuery = (exampleQuery: string) => {
    setQuery(exampleQuery);
  };

  return (
    <Box>
      <Typography variant="h5" gutterBottom>
        Query Documents
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        Ask questions about your documents and get AI-generated answers with citations.
      </Typography>

      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Stack spacing={3}>
            <FormControl fullWidth>
              <InputLabel>Datasets (Optional)</InputLabel>
              <Select
                multiple
                value={selectedDatasets}
                onChange={handleDatasetChange}
                input={<OutlinedInput label="Datasets (Optional)" />}
                renderValue={(selected) => (
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                    {selected.map((value) => (
                      <Chip key={value} label={value} size="small" />
                    ))}
                  </Box>
                )}
              >
                {datasets.map((dataset) => (
                  <MenuItem key={dataset.name} value={dataset.name}>
                    {dataset.name} ({dataset.document_count} docs)
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            <FormControl fullWidth>
              <InputLabel>Role Context</InputLabel>
              <Select
                value={role}
                onChange={(e) => setRole(e.target.value as UserRole)}
              >
                {Object.entries(roleDescriptions).map(([key, description]) => (
                  <MenuItem key={key} value={key}>
                    {key.charAt(0).toUpperCase() + key.slice(1)} - {description}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            <TextField
              fullWidth
              multiline
              rows={4}
              label="Your Question"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Enter your question here..."
              variant="outlined"
            />

            <Box>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                Example queries for {role}:
              </Typography>
              <Stack direction="row" spacing={1} flexWrap="wrap">
                {exampleQueries[role].map((example, index) => (
                  <Chip
                    key={index}
                    label={example}
                    onClick={() => handleExampleQuery(example)}
                    clickable
                    size="small"
                    variant="outlined"
                  />
                ))}
              </Stack>
            </Box>

            {error && (
              <Alert severity="error" onClose={() => setError(null)}>
                {error}
              </Alert>
            )}

            <Button
              fullWidth
              variant="contained"
              size="large"
              startIcon={loading ? <CircularProgress size={20} /> : <Search />}
              onClick={handleQuery}
              disabled={loading || !query.trim()}
            >
              {loading ? 'Searching...' : 'Search Documents'}
            </Button>
          </Stack>
        </CardContent>
      </Card>

      {result && (
        <Card>
          <CardContent>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
              <Psychology sx={{ mr: 1, color: 'primary.main' }} />
              <Typography variant="h6">Answer</Typography>
              <Box sx={{ ml: 'auto' }}>
                <Chip
                  label={`Confidence: ${result.confidence}`}
                  color={
                    result.confidence === 'high'
                      ? 'success'
                      : result.confidence === 'medium'
                      ? 'warning'
                      : 'error'
                  }
                  size="small"
                />
              </Box>
            </Box>

            {result.disclaimer && (
              <Alert severity="warning" sx={{ mb: 2 }}>
                {result.disclaimer}
              </Alert>
            )}

            <Paper elevation={0} sx={{ p: 2, backgroundColor: 'grey.50', mb: 3 }}>
              <ReactMarkdown>{result.answer}</ReactMarkdown>
              <IconButton
                size="small"
                onClick={() => copyToClipboard(result.answer)}
                sx={{ mt: 1 }}
              >
                <ContentCopy fontSize="small" />
              </IconButton>
            </Paper>

            {result.highlights && result.highlights.length > 0 && (
              <>
                <Typography variant="h6" sx={{ mb: 2 }}>
                  Key Highlights
                </Typography>
                <Stack spacing={1} sx={{ mb: 3 }}>
                  {result.highlights.map((highlight, index) => (
                    <Box key={index} sx={{ display: 'flex', alignItems: 'flex-start' }}>
                      <FormatQuote sx={{ mr: 1, color: 'text.secondary' }} />
                      <Typography variant="body2">{highlight}</Typography>
                    </Box>
                  ))}
                </Stack>
              </>
            )}

            <Divider sx={{ my: 3 }} />

            <Typography variant="h6" sx={{ mb: 2 }}>
              Sources & Citations ({result.citations.length})
            </Typography>
            <Stack spacing={2}>
              {result.citations.map((citation, index) => (
                <Paper key={index} elevation={1} sx={{ p: 2 }}>
                  <Box
                    sx={{
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      cursor: 'pointer',
                    }}
                    onClick={() => toggleCitation(index)}
                  >
                    <Box>
                      <Typography variant="subtitle2">
                        [Source {citation.source_number}] {citation.filename}
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        Dataset: {citation.dataset} | Chunk: {citation.chunk_index} | 
                        Relevance: {(citation.relevance_score * 100).toFixed(1)}%
                      </Typography>
                    </Box>
                    <IconButton size="small">
                      {expandedCitations.has(index) ? <ExpandLess /> : <ExpandMore />}
                    </IconButton>
                  </Box>
                  <Collapse in={expandedCitations.has(index)}>
                    <Box sx={{ mt: 2, p: 2, backgroundColor: 'grey.50', borderRadius: 1 }}>
                      <Typography variant="body2" sx={{ fontStyle: 'italic' }}>
                        "{citation.snippet}"
                      </Typography>
                    </Box>
                  </Collapse>
                </Paper>
              ))}
            </Stack>

            {result.suggested_followup && (
              <Box sx={{ mt: 3 }}>
                <Typography variant="subtitle2" color="text.secondary">
                  Suggested follow-up:
                </Typography>
                <Chip
                  label={result.suggested_followup}
                  onClick={() => setQuery(result.suggested_followup!)}
                  clickable
                  variant="outlined"
                  sx={{ mt: 1 }}
                />
              </Box>
            )}
          </CardContent>
        </Card>
      )}
    </Box>
  );
};

export default QuerySection;