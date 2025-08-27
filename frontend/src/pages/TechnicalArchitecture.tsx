import React, { useState, useEffect, useRef } from 'react';
import { motion, useScroll, useTransform, AnimatePresence } from 'framer-motion';
import { Link } from 'react-router-dom';
import {
  ArrowLeft,
  Brain,
  Database,
  Server,
  Code2,
  Layers,
  Cpu,
  GitBranch,
  Workflow,
  FileText,
  Upload,
  Search,
  Bot,
  Zap,
  Shield,
  Package,
  Terminal,
  Cloud,
  Activity,
  Clock,
  CheckCircle,
  ChevronRight,
  ExternalLink,
  Play,
  Pause,
  Copy,
  Check,
  ArrowRight,
  FileSearch,
  Hash,
  BookOpen,
  MessageSquare,
  BarChart3,
  Users,
  Settings,
  Gauge,
  Container,
  Network,
  HardDrive,
  MemoryStick,
  Braces,
  Binary,
  Rocket
} from 'lucide-react';

const TechnicalArchitecture: React.FC = () => {
  const [activeSection, setActiveSection] = useState(0);
  const [isFlowAnimating, setIsFlowAnimating] = useState(true);
  const [copiedCode, setCopiedCode] = useState<string | null>(null);
  const [selectedAgent, setSelectedAgent] = useState(0);
  const containerRef = useRef<HTMLDivElement>(null);

  const { scrollYProgress } = useScroll({
    target: containerRef,
    offset: ["start start", "end end"]
  });

  const progressBar = useTransform(scrollYProgress, [0, 1], ["0%", "100%"]);

  useEffect(() => {
    const handleScroll = () => {
      const sections = document.querySelectorAll('.tech-section');
      const scrollPosition = window.scrollY + window.innerHeight / 2;
      
      sections.forEach((section, index) => {
        const rect = section.getBoundingClientRect();
        const top = rect.top + window.scrollY;
        const bottom = top + rect.height;
        
        if (scrollPosition >= top && scrollPosition <= bottom) {
          setActiveSection(index);
        }
      });
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const copyCode = (code: string, id: string) => {
    navigator.clipboard.writeText(code);
    setCopiedCode(id);
    setTimeout(() => setCopiedCode(null), 2000);
  };

  const techStack = {
    backend: [
      { name: 'FastAPI', version: 'Python 3.10+', icon: '⚡', description: 'High-performance async framework' },
      { name: 'ChromaDB', version: 'Latest', icon: '🎯', description: 'Vector database for embeddings' },
      { name: 'LangChain', version: '0.1.x', icon: '🔗', description: 'LLM orchestration framework' },
      { name: 'Ollama', version: 'llama3.2:1b', icon: '🦙', description: 'Local LLM inference' },
      { name: 'Poetry', version: '1.7+', icon: '📦', description: 'Dependency management' },
      { name: 'AsyncIO', version: 'Native', icon: '🔄', description: 'Concurrent processing' }
    ],
    frontend: [
      { name: 'React', version: '19.0', icon: '⚛️', description: 'UI library with latest features' },
      { name: 'TypeScript', version: '5.x', icon: '📘', description: 'Type-safe development' },
      { name: 'Vite', version: '5.4', icon: '⚡', description: 'Lightning-fast build tool' },
      { name: 'Tailwind CSS', version: '3.x', icon: '🎨', description: 'Utility-first CSS' },
      { name: 'Framer Motion', version: '11.x', icon: '✨', description: 'Animation library' },
      { name: 'Radix UI', version: 'Latest', icon: '🎯', description: 'Accessible components' }
    ],
    infrastructure: [
      { name: 'Docker', version: 'Latest', icon: '🐳', description: 'Containerization' },
      { name: 'Docker Compose', version: '2.x', icon: '🎼', description: 'Multi-container orchestration' },
      { name: 'Redis', version: '7.x', icon: '🔴', description: 'Job tracking & caching' },
      { name: 'PostgreSQL', version: '15+', icon: '🐘', description: 'Relational database (optional)' }
    ]
  };

  const agents = [
    {
      name: 'DocumentIngestionAgent',
      role: 'Parallel document processing',
      color: 'from-blue-400 to-blue-600',
      capabilities: ['Batch processing', 'Format detection', 'Error recovery']
    },
    {
      name: 'ParallelRetrievalAgent',
      role: 'Multi-strategy search',
      color: 'from-purple-400 to-purple-600',
      capabilities: ['Semantic search', 'Keyword matching', 'Hybrid search']
    },
    {
      name: 'RankingAgent',
      role: 'Result scoring & ordering',
      color: 'from-green-400 to-green-600',
      capabilities: ['Composite scoring', 'Relevance weighting', 'Deduplication']
    },
    {
      name: 'SummarizationAgent',
      role: 'Answer generation',
      color: 'from-yellow-400 to-yellow-600',
      capabilities: ['Role-aware formatting', 'Context management', 'LLM orchestration']
    },
    {
      name: 'CitationAgent',
      role: 'Source attribution',
      color: 'from-red-400 to-red-600',
      capabilities: ['Citation extraction', 'Source validation', 'Metadata linking']
    },
    {
      name: 'ValidationAgent',
      role: 'Quality assurance',
      color: 'from-indigo-400 to-indigo-600',
      capabilities: ['Answer validation', 'Confidence scoring', 'Coverage checking']
    }
  ];

  const processingPipeline = [
    { step: 'Upload', duration: '~100ms', description: 'Multipart form data reception' },
    { step: 'AsyncDocumentProcessor', duration: '~2s/doc', description: '4 parallel workers' },
    { step: 'MarkItDown', duration: '~500ms', description: 'Convert to markdown' },
    { step: 'Chunking', duration: '~200ms', description: '1000 chars, 200 overlap' },
    { step: 'Embedding', duration: '~1s', description: 'Vector generation' },
    { step: 'ChromaDB', duration: '~100ms', description: 'Vector storage' }
  ];

  const performanceMetrics = {
    documentProcessing: [
      { metric: '10 PDFs (50MB)', sync: '45s', async: '9s', improvement: '5x' },
      { metric: '100 Documents', sync: '7m 30s', async: '1m 30s', improvement: '5x' },
      { metric: '1000 Chunks', sync: '2m', async: '24s', improvement: '5x' }
    ],
    queryProcessing: [
      { metric: 'Semantic Search', time: '~2s', accuracy: '92%' },
      { metric: 'Keyword Search', time: '~1s', accuracy: '85%' },
      { metric: 'Hybrid Search', time: '~2s', accuracy: '95%' },
      { metric: 'Full Pipeline', time: '~3s', accuracy: '97%' }
    ]
  };

  const apiEndpoints = [
    {
      method: 'POST',
      path: '/api/upload',
      description: 'Upload and process documents',
      body: '{ files: File[], dataset_name: string }'
    },
    {
      method: 'POST',
      path: '/api/query',
      description: 'Query documents with AI',
      body: '{ query: string, dataset_names: string[], role: string }'
    },
    {
      method: 'GET',
      path: '/api/datasets',
      description: 'List all datasets',
      body: null
    },
    {
      method: 'GET',
      path: '/api/system/status',
      description: 'System configuration and health',
      body: null
    },
    {
      method: 'POST',
      path: '/api/chat',
      description: 'Interactive chat with documents',
      body: '{ message: string, dataset: string, history: Message[] }'
    },
    {
      method: 'POST',
      path: '/api/summarize',
      description: 'Generate document summaries',
      body: '{ dataset: string, type: string, max_length: number }'
    }
  ];

  const codeExamples = {
    asyncProcessor: `# Asynchronous Document Processing
async def process_batch_async(
    file_paths: List[str],
    dataset_name: str,
    batch_size: int = 5
) -> List[Dict[str, Any]]:
    async with AsyncDocumentProcessor(max_workers=4) as processor:
        results = []
        for i in range(0, len(file_paths), batch_size):
            batch = file_paths[i:i + batch_size]
            batch_results = await asyncio.gather(
                *[processor.process_file_async(f) for f in batch]
            )
            results.extend(batch_results)
        return results`,
    
    multiAgent: `# Multi-Agent Query System
class MultiAgentOrchestrator:
    def __init__(self):
        self.agents = {
            'retrieval': ParallelRetrievalAgent(),
            'ranking': RankingAgent(),
            'summarization': SummarizationAgent(),
            'citation': CitationAgent(),
            'validation': ValidationAgent()
        }
    
    async def process_query(self, query: str) -> Dict:
        # Phase 1: Parallel retrieval
        results = await self.agents['retrieval'].search(query)
        
        # Phase 2: Ranking
        ranked = await self.agents['ranking'].rank(results)
        
        # Phase 3: Generate answer
        answer = await self.agents['summarization'].generate(
            query, ranked
        )
        
        # Phase 4: Parallel validation
        citations, validation = await asyncio.gather(
            self.agents['citation'].extract(answer),
            self.agents['validation'].validate(answer, query)
        )
        
        return {
            'answer': answer,
            'citations': citations,
            'confidence': validation['score']
        }`,
    
    dockerCompose: `# Docker Compose Configuration
version: '3.8'

services:
  backend:
    build: ./backend
    environment:
      - ASYNC_PROCESSING_ENABLED=true
      - MAX_PARALLEL_DOCUMENTS=5
      - MULTI_AGENT_ENABLED=true
      - LLM_PROVIDER=ollama
    volumes:
      - ./datasets:/app/datasets
    depends_on:
      - chromadb
      - ollama
      - redis
    
  ollama:
    image: ollama/ollama:latest
    volumes:
      - ollama_data:/root/.ollama
    command: |
      sh -c "ollama serve &
             ollama pull llama3.2:1b &&
             ollama pull nomic-embed-text &&
             wait"
  
  chromadb:
    image: chromadb/chroma:latest
    volumes:
      - chroma_data:/chroma/chroma
    environment:
      - IS_PERSISTENT=TRUE`,
    
    configuration: `# Environment Configuration
# Asynchronous Processing
ASYNC_PROCESSING_ENABLED=true
MAX_PARALLEL_DOCUMENTS=5
MAX_PARALLEL_CHUNKS=10

# Multi-Agent System
MULTI_AGENT_ENABLED=true
MAX_AGENTS_PER_QUERY=6

# LLM Configuration
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://ollama:11434
OLLAMA_CHAT_MODEL=llama3.2:1b
OLLAMA_EMBEDDING_MODEL=nomic-embed-text

# Document Processing
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_FILE_SIZE_MB=100

# Vector Store
CHROMA_PERSIST_DIRECTORY=./chroma_db
CHROMA_COLLECTION_NAME=verixai_docs`
  };

  return (
    <div ref={containerRef} className="min-h-screen bg-gradient-to-br from-neutral-900 via-neutral-800 to-neutral-900 text-white overflow-hidden">
      {/* Progress Bar */}
      <div className="fixed top-0 left-0 right-0 z-50 h-1 bg-neutral-800">
        <motion.div
          className="h-full bg-gradient-to-r from-primary-500 to-accent-500"
          style={{ width: progressBar }}
        />
      </div>

      {/* Navigation */}
      <motion.nav
        initial={{ y: -100 }}
        animate={{ y: 0 }}
        className="fixed top-0 left-0 right-0 z-40 bg-neutral-900/80 backdrop-blur-md border-b border-neutral-800"
      >
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <Link to="/" className="flex items-center space-x-2 text-neutral-400 hover:text-white transition-colors">
              <ArrowLeft className="w-5 h-5" />
              <span>Back to Home</span>
            </Link>
            <div className="w-px h-6 bg-neutral-700"></div>
            <div className="flex items-center space-x-2">
              <Binary className="w-6 h-6 text-primary-500" />
              <span className="text-xl font-bold">Technical Architecture</span>
            </div>
          </div>

          <div className="hidden md:flex items-center space-x-6 text-sm">
            {['Stack', 'Agents', 'Pipeline', 'API', 'Performance'].map((item, index) => (
              <button
                key={item}
                onClick={() => {
                  document.querySelectorAll('.tech-section')[index + 1]?.scrollIntoView({ behavior: 'smooth' });
                }}
                className={`hover:text-primary-400 transition-colors ${activeSection === index + 1 ? 'text-primary-400' : 'text-neutral-400'}`}
              >
                {item}
              </button>
            ))}
          </div>
        </div>
      </motion.nav>

      {/* Hero Section - System Architecture */}
      <section className="tech-section relative min-h-screen flex items-center justify-center px-6 pt-20">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-center mb-16"
          >
            <h1 className="text-5xl lg:text-7xl font-bold mb-6">
              <span className="bg-gradient-to-r from-primary-400 to-accent-400 bg-clip-text text-transparent">
                Under the Hood
              </span>
            </h1>
            <p className="text-xl text-neutral-400 max-w-3xl mx-auto">
              A deep dive into VerixAI's technical architecture, featuring multi-agent systems,
              parallel processing, and production-grade infrastructure
            </p>
          </motion.div>

          {/* Interactive System Diagram */}
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.2 }}
            className="relative bg-neutral-800/50 backdrop-blur rounded-2xl p-8 border border-neutral-700"
          >
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
              {/* Frontend Layer */}
              <motion.div
                whileHover={{ scale: 1.02 }}
                className="bg-gradient-to-br from-blue-900/20 to-blue-800/20 rounded-xl p-6 border border-blue-700/50"
              >
                <div className="flex items-center space-x-3 mb-4">
                  <div className="w-10 h-10 bg-blue-600 rounded-lg flex items-center justify-center">
                    <Code2 className="w-6 h-6" />
                  </div>
                  <h3 className="text-lg font-bold">Frontend Layer</h3>
                </div>
                <ul className="space-y-2 text-sm text-neutral-300">
                  <li className="flex items-center space-x-2">
                    <CheckCircle className="w-4 h-4 text-green-500" />
                    <span>React 19 + TypeScript</span>
                  </li>
                  <li className="flex items-center space-x-2">
                    <CheckCircle className="w-4 h-4 text-green-500" />
                    <span>Vite Build System</span>
                  </li>
                  <li className="flex items-center space-x-2">
                    <CheckCircle className="w-4 h-4 text-green-500" />
                    <span>Tailwind CSS + Radix UI</span>
                  </li>
                </ul>
              </motion.div>

              {/* Backend Layer */}
              <motion.div
                whileHover={{ scale: 1.02 }}
                className="bg-gradient-to-br from-green-900/20 to-green-800/20 rounded-xl p-6 border border-green-700/50"
              >
                <div className="flex items-center space-x-3 mb-4">
                  <div className="w-10 h-10 bg-green-600 rounded-lg flex items-center justify-center">
                    <Server className="w-6 h-6" />
                  </div>
                  <h3 className="text-lg font-bold">Backend Layer</h3>
                </div>
                <ul className="space-y-2 text-sm text-neutral-300">
                  <li className="flex items-center space-x-2">
                    <CheckCircle className="w-4 h-4 text-green-500" />
                    <span>FastAPI + AsyncIO</span>
                  </li>
                  <li className="flex items-center space-x-2">
                    <CheckCircle className="w-4 h-4 text-green-500" />
                    <span>6 Specialized AI Agents</span>
                  </li>
                  <li className="flex items-center space-x-2">
                    <CheckCircle className="w-4 h-4 text-green-500" />
                    <span>LangChain + Ollama/OpenAI</span>
                  </li>
                </ul>
              </motion.div>

              {/* Data Layer */}
              <motion.div
                whileHover={{ scale: 1.02 }}
                className="bg-gradient-to-br from-purple-900/20 to-purple-800/20 rounded-xl p-6 border border-purple-700/50"
              >
                <div className="flex items-center space-x-3 mb-4">
                  <div className="w-10 h-10 bg-purple-600 rounded-lg flex items-center justify-center">
                    <Database className="w-6 h-6" />
                  </div>
                  <h3 className="text-lg font-bold">Data Layer</h3>
                </div>
                <ul className="space-y-2 text-sm text-neutral-300">
                  <li className="flex items-center space-x-2">
                    <CheckCircle className="w-4 h-4 text-green-500" />
                    <span>ChromaDB Vector Store</span>
                  </li>
                  <li className="flex items-center space-x-2">
                    <CheckCircle className="w-4 h-4 text-green-500" />
                    <span>Redis Job Tracking</span>
                  </li>
                  <li className="flex items-center space-x-2">
                    <CheckCircle className="w-4 h-4 text-green-500" />
                    <span>PostgreSQL (Optional)</span>
                  </li>
                </ul>
              </motion.div>
            </div>

            {/* Animated Flow Lines */}
            <svg className="absolute inset-0 w-full h-full pointer-events-none">
              <motion.line
                x1="33%" y1="50%" x2="66%" y2="50%"
                stroke="url(#gradient1)"
                strokeWidth="2"
                strokeDasharray="5 5"
                initial={{ pathLength: 0 }}
                animate={{ pathLength: 1 }}
                transition={{ duration: 2, repeat: Infinity }}
              />
              <defs>
                <linearGradient id="gradient1">
                  <stop offset="0%" stopColor="#3B82F6" />
                  <stop offset="100%" stopColor="#10B981" />
                </linearGradient>
              </defs>
            </svg>
          </motion.div>
        </div>
      </section>

      {/* Technology Stack Section */}
      <section className="tech-section min-h-screen px-6 py-20">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl lg:text-5xl font-bold mb-4">
              Technology <span className="text-primary-400">Stack</span>
            </h2>
            <p className="text-xl text-neutral-400">
              Production-grade technologies powering VerixAI
            </p>
          </motion.div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            {/* Backend Technologies */}
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              className="space-y-4"
            >
              <h3 className="text-2xl font-bold text-green-400 mb-6">Backend</h3>
              {techStack.backend.map((tech, index) => (
                <motion.div
                  key={tech.name}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ delay: index * 0.1 }}
                  whileHover={{ x: 10 }}
                  className="bg-neutral-800/50 backdrop-blur rounded-lg p-4 border border-neutral-700 hover:border-green-600 transition-all"
                >
                  <div className="flex items-start space-x-3">
                    <span className="text-2xl">{tech.icon}</span>
                    <div className="flex-grow">
                      <div className="flex items-center justify-between mb-1">
                        <h4 className="font-semibold">{tech.name}</h4>
                        <span className="text-xs text-neutral-500">{tech.version}</span>
                      </div>
                      <p className="text-sm text-neutral-400">{tech.description}</p>
                    </div>
                  </div>
                </motion.div>
              ))}
            </motion.div>

            {/* Frontend Technologies */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              className="space-y-4"
            >
              <h3 className="text-2xl font-bold text-blue-400 mb-6">Frontend</h3>
              {techStack.frontend.map((tech, index) => (
                <motion.div
                  key={tech.name}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ delay: index * 0.1 }}
                  whileHover={{ x: 10 }}
                  className="bg-neutral-800/50 backdrop-blur rounded-lg p-4 border border-neutral-700 hover:border-blue-600 transition-all"
                >
                  <div className="flex items-start space-x-3">
                    <span className="text-2xl">{tech.icon}</span>
                    <div className="flex-grow">
                      <div className="flex items-center justify-between mb-1">
                        <h4 className="font-semibold">{tech.name}</h4>
                        <span className="text-xs text-neutral-500">{tech.version}</span>
                      </div>
                      <p className="text-sm text-neutral-400">{tech.description}</p>
                    </div>
                  </div>
                </motion.div>
              ))}
            </motion.div>

            {/* Infrastructure */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              className="space-y-4"
            >
              <h3 className="text-2xl font-bold text-purple-400 mb-6">Infrastructure</h3>
              {techStack.infrastructure.map((tech, index) => (
                <motion.div
                  key={tech.name}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ delay: index * 0.1 }}
                  whileHover={{ x: 10 }}
                  className="bg-neutral-800/50 backdrop-blur rounded-lg p-4 border border-neutral-700 hover:border-purple-600 transition-all"
                >
                  <div className="flex items-start space-x-3">
                    <span className="text-2xl">{tech.icon}</span>
                    <div className="flex-grow">
                      <div className="flex items-center justify-between mb-1">
                        <h4 className="font-semibold">{tech.name}</h4>
                        <span className="text-xs text-neutral-500">{tech.version}</span>
                      </div>
                      <p className="text-sm text-neutral-400">{tech.description}</p>
                    </div>
                  </div>
                </motion.div>
              ))}
            </motion.div>
          </div>
        </div>
      </section>

      {/* Multi-Agent Architecture */}
      <section className="tech-section min-h-screen px-6 py-20 bg-neutral-900/50">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl lg:text-5xl font-bold mb-4">
              Multi-Agent <span className="text-primary-400">Architecture</span>
            </h2>
            <p className="text-xl text-neutral-400">
              Six specialized agents working in parallel for optimal performance
            </p>
          </motion.div>

          {/* Agent Flow Visualization */}
          <div className="relative mb-12">
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              whileInView={{ opacity: 1, scale: 1 }}
              viewport={{ once: true }}
              className="bg-neutral-800/30 backdrop-blur rounded-2xl p-8 border border-neutral-700"
            >
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-xl font-bold">Agent Orchestration</h3>
                <button
                  onClick={() => setIsFlowAnimating(!isFlowAnimating)}
                  className="px-4 py-2 bg-primary-600 hover:bg-primary-700 rounded-lg transition-colors flex items-center space-x-2"
                >
                  {isFlowAnimating ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                  <span>{isFlowAnimating ? 'Pause' : 'Play'}</span>
                </button>
              </div>

              {/* Agent Grid */}
              <div className="grid grid-cols-2 md:grid-cols-3 gap-6">
                {agents.map((agent, index) => (
                  <motion.div
                    key={agent.name}
                    initial={{ opacity: 0, scale: 0.5 }}
                    whileInView={{ opacity: 1, scale: 1 }}
                    viewport={{ once: true }}
                    transition={{ delay: index * 0.1 }}
                    whileHover={{ scale: 1.05 }}
                    onClick={() => setSelectedAgent(index)}
                    className={`cursor-pointer p-4 rounded-xl border transition-all ${
                      selectedAgent === index 
                        ? 'border-primary-500 bg-primary-900/20' 
                        : 'border-neutral-600 bg-neutral-800/50'
                    }`}
                  >
                    <div className={`w-12 h-12 bg-gradient-to-br ${agent.color} rounded-lg flex items-center justify-center mb-3`}>
                      <Bot className="w-6 h-6 text-white" />
                    </div>
                    <h4 className="font-semibold text-sm mb-1">{agent.name.replace('Agent', '')}</h4>
                    <p className="text-xs text-neutral-400">{agent.role}</p>
                    {isFlowAnimating && (
                      <motion.div
                        className="mt-2 h-1 bg-neutral-700 rounded-full overflow-hidden"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                      >
                        <motion.div
                          className="h-full bg-gradient-to-r from-primary-500 to-accent-500"
                          animate={{ x: ["-100%", "100%"] }}
                          transition={{ duration: 2, repeat: Infinity, delay: index * 0.3 }}
                        />
                      </motion.div>
                    )}
                  </motion.div>
                ))}
              </div>

              {/* Selected Agent Details */}
              <AnimatePresence mode="wait">
                <motion.div
                  key={selectedAgent}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  className="mt-8 p-6 bg-neutral-900/50 rounded-lg border border-neutral-700"
                >
                  <h4 className="text-lg font-bold mb-3">{agents[selectedAgent].name}</h4>
                  <p className="text-neutral-400 mb-4">{agents[selectedAgent].role}</p>
                  <div className="space-y-2">
                    <p className="text-sm font-semibold text-neutral-300">Capabilities:</p>
                    <ul className="space-y-1">
                      {agents[selectedAgent].capabilities.map((cap, i) => (
                        <li key={i} className="flex items-center space-x-2 text-sm text-neutral-400">
                          <CheckCircle className="w-4 h-4 text-green-500" />
                          <span>{cap}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                </motion.div>
              </AnimatePresence>
            </motion.div>
          </div>

          {/* Agent Code Example */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="bg-neutral-900 rounded-xl p-6 border border-neutral-700"
          >
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center space-x-2">
                <Code2 className="w-5 h-5 text-primary-400" />
                <span className="font-mono text-sm">multi_agent_system.py</span>
              </div>
              <button
                onClick={() => copyCode(codeExamples.multiAgent, 'multiAgent')}
                className="p-2 hover:bg-neutral-800 rounded-lg transition-colors"
              >
                {copiedCode === 'multiAgent' ? <Check className="w-4 h-4 text-green-500" /> : <Copy className="w-4 h-4" />}
              </button>
            </div>
            <pre className="text-sm text-green-400 overflow-x-auto">
              <code>{codeExamples.multiAgent}</code>
            </pre>
          </motion.div>
        </div>
      </section>

      {/* Document Processing Pipeline */}
      <section className="tech-section min-h-screen px-6 py-20">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl lg:text-5xl font-bold mb-4">
              Document Processing <span className="text-primary-400">Pipeline</span>
            </h2>
            <p className="text-xl text-neutral-400">
              High-performance async processing with 5x speed improvement
            </p>
          </motion.div>

          {/* Pipeline Flow */}
          <div className="mb-12">
            <motion.div
              initial={{ opacity: 0 }}
              whileInView={{ opacity: 1 }}
              viewport={{ once: true }}
              className="bg-gradient-to-r from-neutral-900 to-neutral-800 rounded-2xl p-8 border border-neutral-700"
            >
              <div className="space-y-4">
                {processingPipeline.map((step, index) => (
                  <motion.div
                    key={step.step}
                    initial={{ opacity: 0, x: -20 }}
                    whileInView={{ opacity: 1, x: 0 }}
                    viewport={{ once: true }}
                    transition={{ delay: index * 0.1 }}
                    className="flex items-center space-x-4"
                  >
                    <div className="flex-shrink-0 w-12 h-12 bg-primary-600 rounded-full flex items-center justify-center text-white font-bold">
                      {index + 1}
                    </div>
                    <div className="flex-grow">
                      <div className="flex items-center justify-between">
                        <h4 className="font-semibold text-lg">{step.step}</h4>
                        <span className="text-sm text-green-400">{step.duration}</span>
                      </div>
                      <p className="text-sm text-neutral-400">{step.description}</p>
                    </div>
                    {index < processingPipeline.length - 1 && (
                      <ChevronRight className="w-5 h-5 text-neutral-600" />
                    )}
                  </motion.div>
                ))}
              </div>
            </motion.div>
          </div>

          {/* Performance Comparison */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              className="bg-neutral-800/50 backdrop-blur rounded-xl p-6 border border-neutral-700"
            >
              <h3 className="text-xl font-bold mb-4 text-green-400">Document Processing Speed</h3>
              <div className="space-y-3">
                {performanceMetrics.documentProcessing.map((metric, index) => (
                  <div key={index} className="space-y-2">
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-neutral-300">{metric.metric}</span>
                      <span className="text-green-400 font-bold">{metric.improvement}</span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <div className="flex-grow bg-neutral-700 rounded-full h-2 overflow-hidden">
                        <motion.div
                          className="h-full bg-red-500"
                          initial={{ width: 0 }}
                          whileInView={{ width: '100%' }}
                          viewport={{ once: true }}
                          transition={{ duration: 1 }}
                        />
                      </div>
                      <span className="text-xs text-neutral-500 w-12">{metric.sync}</span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <div className="flex-grow bg-neutral-700 rounded-full h-2 overflow-hidden">
                        <motion.div
                          className="h-full bg-green-500"
                          initial={{ width: 0 }}
                          whileInView={{ width: '20%' }}
                          viewport={{ once: true }}
                          transition={{ duration: 1, delay: 0.5 }}
                        />
                      </div>
                      <span className="text-xs text-neutral-500 w-12">{metric.async}</span>
                    </div>
                  </div>
                ))}
              </div>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, x: 20 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              className="bg-neutral-800/50 backdrop-blur rounded-xl p-6 border border-neutral-700"
            >
              <h3 className="text-xl font-bold mb-4 text-blue-400">Query Performance</h3>
              <div className="space-y-3">
                {performanceMetrics.queryProcessing.map((metric, index) => (
                  <div key={index} className="flex items-center justify-between p-3 bg-neutral-900/50 rounded-lg">
                    <span className="text-sm">{metric.metric}</span>
                    <div className="flex items-center space-x-3">
                      <span className="text-sm text-blue-400">{metric.time}</span>
                      <span className="text-sm text-green-400">{metric.accuracy}</span>
                    </div>
                  </div>
                ))}
              </div>
            </motion.div>
          </div>

          {/* Async Processing Code */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="mt-8 bg-neutral-900 rounded-xl p-6 border border-neutral-700"
          >
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center space-x-2">
                <Code2 className="w-5 h-5 text-primary-400" />
                <span className="font-mono text-sm">async_document_processor.py</span>
              </div>
              <button
                onClick={() => copyCode(codeExamples.asyncProcessor, 'asyncProcessor')}
                className="p-2 hover:bg-neutral-800 rounded-lg transition-colors"
              >
                {copiedCode === 'asyncProcessor' ? <Check className="w-4 h-4 text-green-500" /> : <Copy className="w-4 h-4" />}
              </button>
            </div>
            <pre className="text-sm text-green-400 overflow-x-auto">
              <code>{codeExamples.asyncProcessor}</code>
            </pre>
          </motion.div>
        </div>
      </section>

      {/* API Architecture */}
      <section className="tech-section min-h-screen px-6 py-20 bg-neutral-900/50">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl lg:text-5xl font-bold mb-4">
              RESTful <span className="text-primary-400">API</span>
            </h2>
            <p className="text-xl text-neutral-400">
              Comprehensive API endpoints for seamless integration
            </p>
          </motion.div>

          {/* API Endpoints */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-12">
            {apiEndpoints.map((endpoint, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.1 }}
                whileHover={{ scale: 1.02 }}
                className="bg-neutral-800/50 backdrop-blur rounded-lg p-6 border border-neutral-700 hover:border-primary-600 transition-all"
              >
                <div className="flex items-start space-x-3 mb-3">
                  <span className={`px-3 py-1 text-xs font-bold rounded ${
                    endpoint.method === 'POST' ? 'bg-green-600' : 
                    endpoint.method === 'GET' ? 'bg-blue-600' : 
                    'bg-red-600'
                  }`}>
                    {endpoint.method}
                  </span>
                  <code className="text-sm text-primary-400 font-mono">{endpoint.path}</code>
                </div>
                <p className="text-sm text-neutral-300 mb-2">{endpoint.description}</p>
                {endpoint.body && (
                  <pre className="text-xs text-neutral-500 bg-neutral-900 rounded p-2 overflow-x-auto">
                    <code>{endpoint.body}</code>
                  </pre>
                )}
              </motion.div>
            ))}
          </div>

          {/* Configuration Example */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="bg-neutral-900 rounded-xl p-6 border border-neutral-700"
          >
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center space-x-2">
                <Settings className="w-5 h-5 text-primary-400" />
                <span className="font-mono text-sm">.env.production</span>
              </div>
              <button
                onClick={() => copyCode(codeExamples.configuration, 'configuration')}
                className="p-2 hover:bg-neutral-800 rounded-lg transition-colors"
              >
                {copiedCode === 'configuration' ? <Check className="w-4 h-4 text-green-500" /> : <Copy className="w-4 h-4" />}
              </button>
            </div>
            <pre className="text-sm text-green-400 overflow-x-auto">
              <code>{codeExamples.configuration}</code>
            </pre>
          </motion.div>
        </div>
      </section>

      {/* Infrastructure & Deployment */}
      <section className="tech-section min-h-screen px-6 py-20">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl lg:text-5xl font-bold mb-4">
              Infrastructure & <span className="text-primary-400">Deployment</span>
            </h2>
            <p className="text-xl text-neutral-400">
              Production-ready containerized deployment
            </p>
          </motion.div>

          {/* Docker Architecture */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-12">
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              whileInView={{ opacity: 1, scale: 1 }}
              viewport={{ once: true }}
              className="bg-gradient-to-br from-blue-900/20 to-blue-800/20 rounded-xl p-6 border border-blue-700/50"
            >
              <Container className="w-10 h-10 text-blue-400 mb-4" />
              <h3 className="text-lg font-bold mb-3">Container Architecture</h3>
              <ul className="space-y-2 text-sm text-neutral-300">
                <li>• Frontend: Node.js Alpine</li>
                <li>• Backend: Python 3.10 Slim</li>
                <li>• Ollama: Official Image</li>
                <li>• ChromaDB: Persistent Storage</li>
                <li>• Redis: Alpine</li>
              </ul>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              whileInView={{ opacity: 1, scale: 1 }}
              viewport={{ once: true }}
              transition={{ delay: 0.1 }}
              className="bg-gradient-to-br from-green-900/20 to-green-800/20 rounded-xl p-6 border border-green-700/50"
            >
              <Network className="w-10 h-10 text-green-400 mb-4" />
              <h3 className="text-lg font-bold mb-3">Service Communication</h3>
              <ul className="space-y-2 text-sm text-neutral-300">
                <li>• Internal Docker Network</li>
                <li>• Service Discovery</li>
                <li>• Health Checks</li>
                <li>• Auto-restart Policies</li>
                <li>• Load Balancing Ready</li>
              </ul>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              whileInView={{ opacity: 1, scale: 1 }}
              viewport={{ once: true }}
              transition={{ delay: 0.2 }}
              className="bg-gradient-to-br from-purple-900/20 to-purple-800/20 rounded-xl p-6 border border-purple-700/50"
            >
              <HardDrive className="w-10 h-10 text-purple-400 mb-4" />
              <h3 className="text-lg font-bold mb-3">Resource Requirements</h3>
              <ul className="space-y-2 text-sm text-neutral-300">
                <li>• CPU: 4+ cores</li>
                <li>• RAM: 8GB minimum</li>
                <li>• Storage: 50GB+</li>
                <li>• Network: 100Mbps+</li>
                <li>• GPU: Optional (CUDA)</li>
              </ul>
            </motion.div>
          </div>

          {/* Docker Compose Configuration */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="bg-neutral-900 rounded-xl p-6 border border-neutral-700"
          >
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center space-x-2">
                <Cloud className="w-5 h-5 text-primary-400" />
                <span className="font-mono text-sm">docker-compose.yml</span>
              </div>
              <button
                onClick={() => copyCode(codeExamples.dockerCompose, 'dockerCompose')}
                className="p-2 hover:bg-neutral-800 rounded-lg transition-colors"
              >
                {copiedCode === 'dockerCompose' ? <Check className="w-4 h-4 text-green-500" /> : <Copy className="w-4 h-4" />}
              </button>
            </div>
            <pre className="text-sm text-green-400 overflow-x-auto">
              <code>{codeExamples.dockerCompose}</code>
            </pre>
          </motion.div>

          {/* Deployment Commands */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="mt-8 grid grid-cols-1 md:grid-cols-2 gap-6"
          >
            <div className="bg-neutral-800/50 backdrop-blur rounded-lg p-6 border border-neutral-700">
              <Terminal className="w-8 h-8 text-green-400 mb-3" />
              <h3 className="font-bold mb-2">Quick Start</h3>
              <pre className="text-sm text-green-400 bg-neutral-900 rounded p-3">
                <code>{`# Clone repository
git clone https://github.com/arunsai63/verix-ai
cd verix-ai

# Start all services
docker-compose up --build -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f backend`}</code>
              </pre>
            </div>

            <div className="bg-neutral-800/50 backdrop-blur rounded-lg p-6 border border-neutral-700">
              <Gauge className="w-8 h-8 text-blue-400 mb-3" />
              <h3 className="font-bold mb-2">Production Scaling</h3>
              <pre className="text-sm text-green-400 bg-neutral-900 rounded p-3">
                <code>{`# Scale backend workers
docker-compose up -d --scale backend=3

# Monitor resources
docker stats

# Update with zero downtime
docker-compose up -d --no-deps backend

# Backup data
docker-compose exec -T chromadb \
  tar -czf - /chroma > backup.tar.gz`}</code>
              </pre>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Footer CTA */}
      <section className="px-6 py-20 border-t border-neutral-800">
        <div className="max-w-4xl mx-auto text-center">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
          >
            <h2 className="text-3xl lg:text-4xl font-bold mb-6">
              Ready to Deploy?
            </h2>
            <p className="text-xl text-neutral-400 mb-8">
              Get started with VerixAI's production-ready architecture
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <a
                href="https://github.com/arunsai63/verix-ai"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center space-x-2 px-8 py-4 bg-primary-600 hover:bg-primary-700 text-white rounded-xl font-semibold transition-all transform hover:scale-105"
              >
                <Package className="w-5 h-5" />
                <span>View on GitHub</span>
                <ExternalLink className="w-4 h-4" />
              </a>
              <Link
                to="/dashboard"
                className="inline-flex items-center space-x-2 px-8 py-4 bg-neutral-800 hover:bg-neutral-700 text-white rounded-xl font-semibold transition-all transform hover:scale-105 border border-neutral-700"
              >
                <Rocket className="w-5 h-5" />
                <span>Try Dashboard</span>
                <ArrowRight className="w-4 h-4" />
              </Link>
            </div>
          </motion.div>
        </div>
      </section>
    </div>
  );
};

export default TechnicalArchitecture;