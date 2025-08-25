import React, { useState, useEffect, useRef } from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/Tabs';
import { ScrollArea } from '@/components/ui/ScrollArea';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { Badge } from '@/components/ui/Badge';
import { Alert, AlertDescription } from '@/components/ui/Alert';
import {
  MessageCircle,
  Send,
  Loader2,
  AlertCircle,
  Download,
  Trash2,
  User,
  Bot,
  Plus,
  History,
} from 'lucide-react';
import api from '../services/api';
import Plot from 'react-plotly.js';

interface ChatSectionProps {
  datasets: Array<{ name: string; document_count: number }>;
}

interface Message {
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: string;
  citations?: any[];
  analytics?: any;
}

interface ChatSession {
  session_id: string;
  dataset_names: string[];
  message_count: number;
  created_at: string;
  updated_at: string;
}

const ChatSection: React.FC<ChatSectionProps> = ({ datasets }) => {
  const [selectedDatasets, setSelectedDatasets] = useState<string[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string>('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputMessage, setInputMessage] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>('');
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [activeTab, setActiveTab] = useState<string>('chat');
  const [showAnalytics, setShowAnalytics] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetchSessions();
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const fetchSessions = async () => {
    try {
      const response = await api.listChatSessions();
      setSessions(response.data.sessions || []);
    } catch (err) {
      console.error('Error fetching sessions:', err);
    }
  };

  const createNewSession = async () => {
    if (selectedDatasets.length === 0) {
      setError('Please select at least one dataset');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const response = await api.createChatSession(selectedDatasets);
      setCurrentSessionId(response.data.session_id);
      setMessages([]);
      fetchSessions();
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to create chat session');
    } finally {
      setLoading(false);
    }
  };

  const loadSession = async (sessionId: string) => {
    setLoading(true);
    setError('');

    try {
      const response = await api.getChatHistory(sessionId);
      setCurrentSessionId(sessionId);
      setSelectedDatasets(response.data.dataset_names);
      setMessages(
        response.data.messages
          .filter((msg: any) => msg.role !== 'system')
          .map((msg: any) => ({
            role: msg.role,
            content: msg.content,
            timestamp: msg.timestamp,
            citations: msg.citations,
            analytics: msg.analytics,
          }))
      );
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to load session');
    } finally {
      setLoading(false);
    }
  };

  const sendMessage = async () => {
    if (!inputMessage.trim() || !currentSessionId) return;

    const userMessage: Message = {
      role: 'user',
      content: inputMessage,
      timestamp: new Date().toISOString(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInputMessage('');
    setLoading(true);
    setError('');

    try {
      const response = await api.sendChatMessage(currentSessionId, userMessage.content);

      const assistantMessage: Message = {
        role: 'assistant',
        content: response.data.message,
        timestamp: response.data.timestamp,
        citations: response.data.citations,
        analytics: response.data.analytics,
      };

      setMessages((prev) => [...prev, assistantMessage]);

      if (response.data.analytics) {
        setShowAnalytics(true);
      }
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to send message');
      // Remove the user message if sending failed
      setMessages((prev) => prev.slice(0, -1));
      setInputMessage(userMessage.content);
    } finally {
      setLoading(false);
    }
  };

  const deleteSession = async (sessionId: string) => {
    try {
      await api.deleteChatSession(sessionId);
      if (sessionId === currentSessionId) {
        setCurrentSessionId('');
        setMessages([]);
      }
      fetchSessions();
    } catch (err) {
      console.error('Error deleting session:', err);
    }
  };

  const exportSession = async (format: 'json' | 'markdown') => {
    if (!currentSessionId) return;

    try {
      const response = await api.exportChatSession(currentSessionId, format);
      const blob = new Blob([response.data.data], {
        type: format === 'json' ? 'application/json' : 'text/markdown',
      });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `chat-session-${currentSessionId}.${format === 'json' ? 'json' : 'md'}`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);
    } catch (err) {
      console.error('Error exporting session:', err);
    }
  };

  const renderVisualization = (viz: any) => {
    if (viz.type === 'plotly' && viz.data) {
      return (
        <div className="my-4">
          <h4 className="text-sm font-medium mb-2">{viz.title}</h4>
          <Plot
            data={viz.data.data}
            layout={viz.data.layout}
            config={{ responsive: true }}
            style={{ width: '100%', height: '400px' }}
          />
        </div>
      );
    }
    return null;
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <MessageCircle className="h-5 w-5" />
            Chat with Documents
          </CardTitle>
          <CardDescription>
            Have an interactive conversation with your documents and data
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Tabs value={activeTab} onValueChange={setActiveTab}>
            <TabsList className="grid w-full grid-cols-2">
              <TabsTrigger value="chat">Chat</TabsTrigger>
              <TabsTrigger value="history">Session History</TabsTrigger>
            </TabsList>

            <TabsContent value="chat" className="space-y-4">
              {!currentSessionId ? (
                <div className="space-y-4">
                  <div>
                    <label className="text-sm font-medium">Select Datasets</label>
                    <div className="flex flex-wrap gap-2 mt-2">
                      {datasets.map((dataset) => (
                        <Badge
                          key={dataset.name}
                          variant={selectedDatasets.includes(dataset.name) ? 'default' : 'secondary'}
                          className="cursor-pointer"
                          onClick={() => {
                            setSelectedDatasets((prev) =>
                              prev.includes(dataset.name)
                                ? prev.filter((d) => d !== dataset.name)
                                : [...prev, dataset.name]
                            );
                          }}
                        >
                          {dataset.name} ({dataset.document_count})
                        </Badge>
                      ))}
                    </div>
                  </div>

                  <Button onClick={createNewSession} disabled={loading || selectedDatasets.length === 0}>
                    <Plus className="mr-2 h-4 w-4" />
                    Start New Chat Session
                  </Button>
                </div>
              ) : (
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Badge variant="secondary">Session: {currentSessionId.slice(0, 8)}...</Badge>
                      {selectedDatasets.map((ds) => (
                        <Badge key={ds} variant="secondary">
                          {ds}
                        </Badge>
                      ))}
                    </div>
                    <div className="flex gap-2">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => exportSession('markdown')}
                      >
                        <Download className="h-4 w-4" />
                      </Button>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => {
                          setCurrentSessionId('');
                          setMessages([]);
                        }}
                      >
                        New Chat
                      </Button>
                    </div>
                  </div>

                  <ScrollArea className="h-[400px] border rounded-lg p-4">
                    <div className="space-y-4">
                      {messages.map((message, index) => (
                        <div
                          key={index}
                          className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                        >
                          <div
                            className={`max-w-[80%] ${message.role === 'user'
                                ? 'bg-primary text-primary-foreground'
                                : 'bg-muted'
                              } rounded-lg p-3`}
                          >
                            <div className="flex items-center gap-2 mb-1">
                              {message.role === 'user' ? (
                                <User className="h-4 w-4" />
                              ) : (
                                <Bot className="h-4 w-4" />
                              )}
                              <span className="text-xs opacity-70">
                                {new Date(message.timestamp).toLocaleTimeString()}
                              </span>
                            </div>
                            <div className="whitespace-pre-wrap">{message.content}</div>

                            {message.citations && message.citations.length > 0 && (
                              <div className="mt-2 pt-2 border-t">
                                <div className="text-xs font-medium mb-1">Sources:</div>
                                {message.citations.map((citation: any, idx: number) => (
                                  <div key={idx} className="text-xs opacity-80">
                                    • {citation.source} (relevance: {(citation.relevance_score * 100).toFixed(0)}%)
                                  </div>
                                ))}
                              </div>
                            )}

                            {message.analytics?.visualizations && (
                              <div className="mt-2">
                                {message.analytics.visualizations.map((viz: any, idx: number) => (
                                  <div key={idx}>{renderVisualization(viz)}</div>
                                ))}
                              </div>
                            )}
                          </div>
                        </div>
                      ))}
                      <div ref={messagesEndRef} />
                    </div>
                  </ScrollArea>

                  <div className="flex gap-2">
                    <Input
                      placeholder="Ask a question about your documents..."
                      value={inputMessage}
                      onChange={(e: React.ChangeEvent<HTMLInputElement>) => setInputMessage(e.target.value)}
                      onKeyDown={(e: React.KeyboardEvent<HTMLInputElement>) => e.key === 'Enter' && !loading && sendMessage()}
                      disabled={loading}
                    />
                    <Button onClick={sendMessage} disabled={loading || !inputMessage.trim()}>
                      {loading ? (
                        <Loader2 className="h-4 w-4 animate-spin" />
                      ) : (
                        <Send className="h-4 w-4" />
                      )}
                    </Button>
                  </div>
                </div>
              )}

              {error && (
                <Alert variant="error">
                  <AlertCircle className="h-4 w-4" />
                  <AlertDescription>{error}</AlertDescription>
                </Alert>
              )}
            </TabsContent>

            <TabsContent value="history" className="space-y-4">
              <div className="space-y-2">
                {sessions.length === 0 ? (
                  <div className="text-center text-muted-foreground py-8">
                    No chat sessions yet. Start a new chat to begin.
                  </div>
                ) : (
                  sessions.map((session) => (
                    <Card key={session.session_id} className="p-4">
                      <div className="flex items-center justify-between">
                        <div>
                          <div className="font-medium">
                            Session: {session.session_id.slice(0, 8)}...
                          </div>
                          <div className="text-sm text-muted-foreground">
                            Datasets: {session.dataset_names.join(', ')}
                          </div>
                          <div className="text-xs text-muted-foreground">
                            {session.message_count} messages • Created{' '}
                            {new Date(session.created_at).toLocaleDateString()}
                          </div>
                        </div>
                        <div className="flex gap-2">
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => loadSession(session.session_id)}
                          >
                            <History className="h-4 w-4 mr-1" />
                            Load
                          </Button>
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => deleteSession(session.session_id)}
                          >
                            <Trash2 className="h-4 w-4" />
                          </Button>
                        </div>
                      </div>
                    </Card>
                  ))
                )}
              </div>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    </div>
  );
};

export default ChatSection;