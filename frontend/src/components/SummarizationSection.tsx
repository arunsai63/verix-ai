import React, { useState, useEffect } from 'react';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Select } from '@/components/ui/Select';
import { Textarea } from '@/components/ui/TextArea';
import { Badge } from '@/components/ui/Badge';
import { Alert, AlertDescription } from '@/components/ui/Alert';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/Tabs';
import {
  FileText,
  BookOpen,
  Sparkles,
  Download,
  Copy,
  Loader2,
  AlertCircle,
} from 'lucide-react';
import api from '../services/api';

interface SummarizationSectionProps {
  datasets: Array<{ name: string; document_count: number }>;
}

const SummarizationSection: React.FC<SummarizationSectionProps> = ({ datasets }) => {
  const [selectedDataset, setSelectedDataset] = useState<string>('');
  const [summaryType, setSummaryType] = useState<string>('executive');
  const [summaryLength, setSummaryLength] = useState<string>('standard');
  const [customInstructions, setCustomInstructions] = useState<string>('');
  const [directContent, setDirectContent] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [summary, setSummary] = useState<any>(null);
  const [error, setError] = useState<string>('');
  const [summaryTypes, setSummaryTypes] = useState<any>(null);
  const [activeTab, setActiveTab] = useState<string>('dataset');

  useEffect(() => {
    fetchSummaryTypes();
  }, []);

  const fetchSummaryTypes = async () => {
    try {
      const response = await api.getSummaryTypes();
      setSummaryTypes(response.data);
    } catch (err) {
      console.error('Error fetching summary types:', err);
    }
  };

  const handleSummarize = async () => {
    setLoading(true);
    setError('');
    setSummary(null);

    try {
      const response = await api.summarizeDocument(
        activeTab === 'dataset' ? selectedDataset : undefined,
        undefined,
        activeTab === 'direct' ? directContent : undefined,
        summaryType,
        summaryLength,
        customInstructions || undefined,
        true
      );

      setSummary(response.data);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to generate summary');
    } finally {
      setLoading(false);
    }
  };

  const copySummary = () => {
    if (summary?.summary) {
      navigator.clipboard.writeText(summary.summary);
    }
  };

  const downloadSummary = () => {
    if (summary?.summary) {
      const blob = new Blob([summary.summary], { type: 'text/plain' });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `summary-${summaryType}-${Date.now()}.txt`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);
    }
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BookOpen className="h-5 w-5" />
            Document Summarization
          </CardTitle>
          <CardDescription>
            Generate intelligent summaries of your documents with various formats and lengths
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <Tabs value={activeTab} onValueChange={setActiveTab}>
            <TabsList className="grid w-full grid-cols-2">
              <TabsTrigger value="dataset">Summarize Dataset</TabsTrigger>
              <TabsTrigger value="direct">Direct Content</TabsTrigger>
            </TabsList>

            <TabsContent value="dataset" className="space-y-4">
              <div>
                <label className="text-sm font-medium">Select Dataset</label>
                <select
                  value={selectedDataset}
                  onChange={(e) => setSelectedDataset(e.target.value)}
                  className="w-full px-3 py-2 border rounded-lg bg-white dark:bg-gray-800 border-gray-300 dark:border-gray-600"
                >
                  <option value="">Choose a dataset to summarize</option>
                  {datasets.map((dataset) => (
                    <option key={dataset.name} value={dataset.name}>
                      {dataset.name} ({dataset.document_count} docs)
                    </option>
                  ))}
                </select>
              </div>
            </TabsContent>

            <TabsContent value="direct" className="space-y-4">
              <div>
                <label className="text-sm font-medium">Paste Content</label>
                <Textarea
                  placeholder="Paste your document content here..."
                  value={directContent}
                  onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) => setDirectContent(e.target.value)}
                  className="h-32"
                />
              </div>
            </TabsContent>
          </Tabs>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="text-sm font-medium">Summary Type</label>
              <select
                value={summaryType}
                onChange={(e) => setSummaryType(e.target.value)}
                className="w-full px-3 py-2 border rounded-lg bg-white dark:bg-gray-800 border-gray-300 dark:border-gray-600"
              >
                {summaryTypes?.summary_types?.map((type: string) => (
                  <option key={type} value={type}>
                    {type.replace(/_/g, ' ').replace(/\b\w/g, (l: string) => l.toUpperCase())}
                  </option>
                ))}
              </select>
              {summaryTypes?.descriptions?.types?.[summaryType] && (
                <p className="text-xs text-muted-foreground mt-1">
                  {summaryTypes.descriptions.types[summaryType]}
                </p>
              )}
            </div>

            <div>
              <label className="text-sm font-medium">Summary Length</label>
              <select
                value={summaryLength}
                onChange={(e) => setSummaryLength(e.target.value)}
                className="w-full px-3 py-2 border rounded-lg bg-white dark:bg-gray-800 border-gray-300 dark:border-gray-600"
              >
                {summaryTypes?.summary_lengths?.map((length: string) => (
                  <option key={length} value={length}>
                    {length.charAt(0).toUpperCase() + length.slice(1)}
                  </option>
                ))}
              </select>
              {summaryTypes?.descriptions?.lengths?.[summaryLength] && (
                <p className="text-xs text-muted-foreground mt-1">
                  {summaryTypes.descriptions.lengths[summaryLength]}
                </p>
              )}
            </div>
          </div>

          <div>
            <label className="text-sm font-medium">Custom Instructions (Optional)</label>
            <Textarea
              placeholder="Add any specific instructions for the summary..."
              value={customInstructions}
              onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) => setCustomInstructions(e.target.value)}
              className="h-20"
            />
          </div>

          <Button
            onClick={handleSummarize}
            disabled={
              loading ||
              (activeTab === 'dataset' && !selectedDataset) ||
              (activeTab === 'direct' && !directContent)
            }
            className="w-full"
          >
            {loading ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Generating Summary...
              </>
            ) : (
              <>
                <Sparkles className="mr-2 h-4 w-4" />
                Generate Summary
              </>
            )}
          </Button>

          {error && (
            <Alert variant="error">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}
        </CardContent>
      </Card>

      {summary && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              <span className="flex items-center gap-2">
                <FileText className="h-5 w-5" />
                Generated Summary
              </span>
              <div className="flex items-center gap-2">
                {summary.cached && <Badge variant="secondary">Cached</Badge>}
                <Badge variant="secondary">
                  {summary.word_count} words
                </Badge>
                <Badge variant="secondary">
                  {(summary.processing_time || 0).toFixed(2)}s
                </Badge>
              </div>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex justify-end gap-2">
              <Button variant="secondary" size="sm" onClick={copySummary}>
                <Copy className="mr-2 h-4 w-4" />
                Copy
              </Button>
              <Button variant="secondary" size="sm" onClick={downloadSummary}>
                <Download className="mr-2 h-4 w-4" />
                Download
              </Button>
            </div>

            <div className="prose max-w-none">
              <div className="whitespace-pre-wrap text-sm">{summary.summary}</div>
            </div>

            {summary.key_topics && summary.key_topics.length > 0 && (
              <div>
                <h4 className="text-sm font-medium mb-2">Key Topics</h4>
                <div className="flex flex-wrap gap-2">
                  {summary.key_topics.map((topic: string, index: number) => (
                    <Badge key={index} variant="secondary">
                      {topic}
                    </Badge>
                  ))}
                </div>
              </div>
            )}

            {summary.citations && summary.citations.length > 0 && (
              <div>
                <h4 className="text-sm font-medium mb-2">Sources</h4>
                <div className="space-y-2">
                  {summary.citations.map((citation: any, index: number) => (
                    <div
                      key={index}
                      className="text-xs text-muted-foreground border-l-2 pl-2"
                    >
                      <div className="font-medium">{citation.source}</div>
                      {citation.page && <div>Page {citation.page}</div>}
                      <div className="mt-1">{citation.excerpt}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            <div className="flex items-center justify-between pt-4 border-t">
              <div className="text-xs text-muted-foreground">
                Confidence Score: {(summary.confidence_score * 100).toFixed(0)}%
              </div>
              <div className="text-xs text-muted-foreground">
                Type: {summary.summary_type} | Length: {summary.length}
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default SummarizationSection;