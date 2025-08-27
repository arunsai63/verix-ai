import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Link } from 'react-router-dom';
import {
  ArrowRight,
  Brain,
  Shield,
  Zap,
  FileSearch,
  Users,
  BarChart3,
  Upload,
  Search,
  Database,
  CheckCircle,
  Star,
  Github,
  Linkedin,
  Globe,
  BookOpen,
  MessageCircle,
  LineChart,
  Cpu,
  Layers,
  GitBranch,
  Settings,
  Lock,
  Server,
  Palette,
  Plug,
  TrendingUp,
  Clock,
  FileText,
  Award,
  Sparkles,
  Activity,
  Workflow,
  Target,
  Rocket,
  ChevronRight,
  ExternalLink,
  Play,
  Pause,
  RefreshCw,
  Binary,
  Bot,
  Command,
  Braces,
  Package,
  Cloud,
  Code2
} from 'lucide-react';
import InteractiveDemo from '../components/InteractiveDemo';

const LandingPage: React.FC = () => {
  const [activeFeature, setActiveFeature] = useState(0);
  const [isProcessing, setIsProcessing] = useState(false);
  const [selectedAgent, setSelectedAgent] = useState(0);
  const [animatedNumber, setAnimatedNumber] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setActiveFeature((prev) => (prev + 1) % 6);
    }, 3000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    const interval = setInterval(() => {
      setAnimatedNumber((prev) => {
        if (prev >= 1000) return 0;
        return prev + Math.floor(Math.random() * 5);
      });
    }, 500);
    return () => clearInterval(interval);
  }, []);

  const fadeIn = {
    initial: { opacity: 0, y: 20 },
    animate: { opacity: 1, y: 0 },
    transition: { duration: 0.6 }
  };

  const staggerChildren = {
    animate: {
      transition: {
        staggerChildren: 0.1
      }
    }
  };

  const agents = [
    { name: 'Document Ingestion', status: 'Active', progress: 85, color: 'primary' },
    { name: 'Parallel Retrieval', status: 'Processing', progress: 60, color: 'accent' },
    { name: 'Ranking Engine', status: 'Optimizing', progress: 90, color: 'success' },
    { name: 'Summarization', status: 'Ready', progress: 100, color: 'warning' },
    { name: 'Citation Validator', status: 'Verified', progress: 95, color: 'primary' },
    { name: 'Quality Control', status: 'Monitoring', progress: 75, color: 'accent' }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-neutral-50 via-primary-50/30 to-accent-50/20 overflow-hidden">
      {/* Animated background elements */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="blob-1 top-0 -left-40"></div>
        <div className="blob-2 top-1/2 -right-40"></div>
        <div className="blob-3 -bottom-40 left-1/2"></div>
      </div>

      {/* Navigation */}
      <motion.nav
        initial={{ y: -100 }}
        animate={{ y: 0 }}
        transition={{ duration: 0.5 }}
        className="relative z-50 px-6 py-4 lg:px-12"
      >
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <motion.div
              whileHover={{ rotate: 360 }}
              transition={{ duration: 0.5 }}
              className="w-10 h-10 bg-gradient-to-br from-primary-500 to-accent-500 rounded-xl flex items-center justify-center"
            >
              <Brain className="w-6 h-6 text-white" />
            </motion.div>
            <span className="text-2xl font-display font-bold gradient-text">VerixAI</span>
          </div>

          <div className="hidden md:flex items-center space-x-8">
            <a href="#features" className="text-neutral-600 hover:text-primary-600 transition-colors">Features</a>
            <a href="#architecture" className="text-neutral-600 hover:text-primary-600 transition-colors">Architecture</a>
            <a href="#performance" className="text-neutral-600 hover:text-primary-600 transition-colors">Performance</a>
            <a href="https://github.com/arunsai63/verix-ai" target="_blank" rel="noopener noreferrer" className="text-neutral-600 hover:text-primary-600 transition-colors flex items-center space-x-1">
              <span>GitHub</span>
              <ExternalLink className="w-3 h-3" />
            </a>
            <Link
              to="/dashboard"
              className="px-6 py-2.5 bg-gradient-to-r from-primary-500 to-primary-600 text-white rounded-xl font-medium hover:shadow-lg transform hover:scale-105 transition-all duration-200"
            >
              Launch Dashboard
            </Link>
          </div>
        </div>
      </motion.nav>

      {/* Hero Section */}
      <section className="relative px-6 py-20 lg:px-12 lg:py-32">
        <div className="max-w-7xl mx-auto">
          <motion.div
            variants={staggerChildren}
            initial="initial"
            animate="animate"
            className="text-center max-w-4xl mx-auto"
          >
            <motion.div
              variants={fadeIn}
              className="inline-flex items-center space-x-2 px-4 py-2 bg-primary-100 rounded-full mb-6"
            >
              <Sparkles className="w-4 h-4 text-primary-600" />
              <span className="text-primary-700 font-medium text-sm">Multi-Agent AI Platform</span>
            </motion.div>

            <motion.h1
              variants={fadeIn}
              className="text-5xl lg:text-7xl font-display font-bold mb-6"
            >
              Enterprise Document
              <span className="block mt-2 gradient-text break-words">Intelligence at Scale</span>
            </motion.h1>

            <motion.p
              variants={fadeIn}
              className="text-xl text-neutral-600 mb-8 leading-relaxed"
            >
              Process documents 5x faster with parallel AI agents. Extract insights, generate reports, and get cited answers with source citations.
              <span className="block mt-2 text-lg font-medium">Self-hosted • Customizable • Production-ready</span>
            </motion.p>

            <motion.div
              variants={fadeIn}
              className="flex flex-col sm:flex-row gap-4 justify-center items-center"
            >
              <Link
                to="/dashboard"
                className="group px-8 py-4 bg-gradient-to-r from-primary-500 to-primary-600 text-white rounded-xl font-semibold text-lg hover:shadow-2xl transform hover:scale-105 transition-all duration-200 flex items-center space-x-2"
              >
                <span>Launch Platform</span>
                <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
              </Link>

              <button
                onClick={() => {
                  const demoSection = document.getElementById('interactive-demo');
                  if (demoSection) {
                    demoSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
                  }
                }}
                className="px-8 py-4 bg-white/80 backdrop-blur-sm border-2 border-primary-200 text-primary-600 rounded-xl font-semibold text-lg hover:bg-white hover:shadow-lg transform hover:scale-105 transition-all duration-200 flex items-center space-x-2"
              >
                <Play className="w-5 h-5" />
                <span>Live Demo</span>
              </button>
            </motion.div>

            <motion.div
              variants={fadeIn}
              className="mt-12 grid grid-cols-3 gap-8 max-w-2xl mx-auto"
            >
              <div className="text-center">
                <div className="text-2xl font-bold gradient-text">100%</div>
                <div className="text-sm text-neutral-500">Open Source</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold gradient-text">6+</div>
                <div className="text-sm text-neutral-500">AI Agents</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold gradient-text">∞</div>
                <div className="text-sm text-neutral-500">Scalability</div>
              </div>
            </motion.div>
          </motion.div>

          {/* Animated Processing Visual */}
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5, duration: 0.8 }}
            className="mt-20 relative"
          >
            <div className="relative mx-auto max-w-6xl">
              <div className="absolute inset-0 bg-gradient-to-r from-primary-400/20 to-accent-400/20 blur-3xl"></div>
              
              {/* Live Processing Animation */}
              <div className="relative glass rounded-2xl p-8 shadow-2xl">
                <div className="flex items-center justify-between mb-6">
                  <h3 className="text-xl font-bold">Real-time Processing Pipeline</h3>
                  <div className="flex items-center space-x-2">
                    <Activity className="w-5 h-5 text-success-500 animate-pulse" />
                    <span className="text-sm text-success-600 font-medium">Live</span>
                  </div>
                </div>
                
                <div className="space-y-4">
                  {/* Processing Steps */}
                  <div className="flex items-center space-x-4">
                    <motion.div
                      animate={{ rotate: isProcessing ? 360 : 0 }}
                      transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                      className="w-12 h-12 bg-gradient-to-br from-primary-500 to-accent-500 rounded-xl flex items-center justify-center"
                    >
                      <Upload className="w-6 h-6 text-white" />
                    </motion.div>
                    <div className="flex-grow">
                      <div className="h-2 bg-neutral-200 rounded-full overflow-hidden">
                        <motion.div
                          animate={{ width: ["0%", "100%"] }}
                          transition={{ duration: 3, repeat: Infinity }}
                          className="h-full bg-gradient-to-r from-primary-500 to-accent-500"
                        />
                      </div>
                    </div>
                    <div className="text-sm font-mono text-neutral-600">
                      {animatedNumber} docs processed
                    </div>
                  </div>
                  
                  {/* Agent Status Grid */}
                  <div className="grid grid-cols-3 gap-4">
                    {agents.slice(0, 3).map((agent, index) => (
                      <motion.div
                        key={index}
                        whileHover={{ scale: 1.05 }}
                        className="bg-white/50 backdrop-blur rounded-lg p-3 cursor-pointer"
                        onClick={() => setSelectedAgent(index)}
                      >
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-xs font-medium">{agent.name}</span>
                          <Bot className="w-4 h-4 text-primary-500" />
                        </div>
                        <div className="h-1 bg-neutral-200 rounded-full overflow-hidden">
                          <motion.div
                            animate={{ width: `${agent.progress}%` }}
                            transition={{ duration: 1 }}
                            className={`h-full bg-gradient-to-r from-${agent.color}-400 to-${agent.color}-600`}
                          />
                        </div>
                      </motion.div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Interactive Demo Section */}
      <section id="interactive-demo" className="px-3 py-20 lg:px-12 lg:py-32 bg-gradient-to-br from-primary-50/50 to-accent-50/30 relative">
        <InteractiveDemo />
      </section>

      {/* Live Architecture Visualization */}
      <section id="architecture" className="px-6 py-20 lg:px-12 lg:py-32 relative bg-gradient-to-br from-neutral-50 to-primary-50/20">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl lg:text-5xl font-display font-bold mb-4">
              Living <span className="gradient-text">Multi-Agent System</span>
            </h2>
            <p className="text-xl text-neutral-600 max-w-3xl mx-auto">
              Watch AI agents collaborate in real-time to process your documents
            </p>
          </motion.div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-12">
            {/* Interactive Agent Network */}
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              className="relative"
            >
              <div className="bg-neutral-900 rounded-2xl p-8 shadow-2xl h-full flex flex-col">
                <div className="flex items-center justify-between mb-6">
                  <h3 className="text-white font-bold text-xl">Live Agent Orchestration</h3>
                  <button
                    onClick={() => setIsProcessing(!isProcessing)}
                    className="px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors flex items-center space-x-2"
                  >
                    {isProcessing ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                    <span className="text-sm">{isProcessing ? 'Pause' : 'Start'}</span>
                  </button>
                </div>

                {/* Agent Nodes Visualization */}
                <div className="relative flex-grow min-h-[400px] flex items-center justify-center overflow-hidden">
                  {/* Multiple Animated Background Layers */}
                  <div className="absolute inset-0">
                    {/* Rotating gradient background */}
                    <motion.div
                      className="absolute inset-0 opacity-30"
                      animate={{ rotate: 360 }}
                      transition={{ duration: 60, repeat: Infinity, ease: "linear" }}
                    >
                      <div className="w-full h-full bg-gradient-conic from-primary-600 via-accent-600 via-primary-600 to-primary-600"></div>
                    </motion.div>
                    
                    {/* Pulsing circles */}
                    {isProcessing && [0, 1, 2].map((i) => (
                      <motion.div
                        key={`circle-${i}`}
                        className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 rounded-full border border-primary-500/20"
                        initial={{ width: 100, height: 100, opacity: 0 }}
                        animate={{ 
                          width: [100, 300, 300],
                          height: [100, 300, 300],
                          opacity: [0.8, 0, 0]
                        }}
                        transition={{ 
                          duration: 3,
                          delay: i * 1,
                          repeat: Infinity,
                          ease: "easeOut"
                        }}
                      />
                    ))}
                  </div>

                  {/* Enhanced Connection Lines Canvas */}
                  <svg className="absolute inset-0 w-full h-full pointer-events-none">
                    <defs>
                      {/* Multiple gradient definitions */}
                      <radialGradient id="centerGlow">
                        <stop offset="0%" stopColor="#8B5CF6" stopOpacity="0.8" />
                        <stop offset="100%" stopColor="#8B5CF6" stopOpacity="0" />
                      </radialGradient>
                      <filter id="glow">
                        <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
                        <feMerge>
                          <feMergeNode in="coloredBlur"/>
                          <feMergeNode in="SourceGraphic"/>
                        </feMerge>
                      </filter>
                      {agents.map((agent, index) => (
                        <linearGradient key={`grad-${index}`} id={`gradient-${index}`} x1="0%" y1="0%" x2="100%" y2="0%">
                          <stop offset="0%" stopColor={index % 2 === 0 ? "#8B5CF6" : "#3B82F6"} stopOpacity="0" />
                          <stop offset="50%" stopColor={index % 2 === 0 ? "#EC4899" : "#10B981"} stopOpacity="1" />
                          <stop offset="100%" stopColor={index % 2 === 0 ? "#8B5CF6" : "#3B82F6"} stopOpacity="0" />
                        </linearGradient>
                      ))}
                    </defs>
                    
                    {/* Enhanced connection lines with multiple paths */}
                    {agents.map((agent, index) => {
                      const angle = (index * 360) / agents.length;
                      const radius = 140;
                      const x = 50 + (radius * Math.cos((angle * Math.PI) / 180)) / 3.5;
                      const y = 50 + (radius * Math.sin((angle * Math.PI) / 180)) / 3.5;
                      
                      return (
                        <g key={`enhanced-group-${index}`}>
                          {/* Base connection line */}
                          <line
                            x1="50%"
                            y1="50%"
                            x2={`${x}%`}
                            y2={`${y}%`}
                            stroke={`url(#gradient-${index})`}
                            strokeWidth="1"
                            opacity={isProcessing ? 0.3 : 0.1}
                            filter="url(#glow)"
                          />
                          
                          {/* Multiple animated particles per line */}
                          {isProcessing && [0, 1, 2].map((particleIndex) => (
                            <motion.circle
                              key={`particle-${index}-${particleIndex}`}
                              r={particleIndex === 0 ? "4" : particleIndex === 1 ? "3" : "2"}
                              fill={index % 2 === 0 ? "#EC4899" : "#10B981"}
                              filter="url(#glow)"
                              opacity={0.8 - particleIndex * 0.2}
                              initial={{ cx: "50%", cy: "50%" }}
                              animate={{
                                cx: ["50%", `${x}%`, `${x}%`, "50%"],
                                cy: ["50%", `${y}%`, `${y}%`, "50%"],
                                opacity: [0, 1, 1, 0]
                              }}
                              transition={{
                                duration: 2 + particleIndex * 0.5,
                                delay: index * 0.2 + particleIndex * 0.3,
                                repeat: Infinity,
                                ease: "easeInOut"
                              }}
                            />
                          ))}
                          
                          {/* Data packet visualization */}
                          {isProcessing && selectedAgent === index && (
                            <motion.rect
                              width="20"
                              height="10"
                              rx="2"
                              fill="#8B5CF6"
                              opacity={0.7}
                              initial={{ x: "49%", y: "49%" }}
                              animate={{
                                x: [`49%`, `${x - 1}%`, `49%`],
                                y: [`49%`, `${y - 1}%`, `49%`]
                              }}
                              transition={{
                                duration: 1.5,
                                repeat: Infinity,
                                ease: "easeInOut"
                              }}
                            />
                          )}
                        </g>
                      );
                    })}
                    
                    {/* Central glow effect */}
                    <circle
                      cx="50%"
                      cy="50%"
                      r="40"
                      fill="url(#centerGlow)"
                      opacity={isProcessing ? 0.5 : 0.2}
                    />
                    
                    {/* Rotating hexagon pattern */}
                    {isProcessing && (
                      <motion.g
                        animate={{ rotate: 360 }}
                        transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
                        opacity={0.3}
                      >
                        <polygon
                          points="50,35 60,42 60,58 50,65 40,58 40,42"
                          fill="none"
                          stroke="#8B5CF6"
                          strokeWidth="0.5"
                          transform="translate(50, 50) scale(2) translate(-50, -50)"
                        />
                      </motion.g>
                    )}
                  </svg>

                  {/* Central Orchestrator */}
                  <motion.div
                    animate={isProcessing ? { 
                      scale: [1, 1.05, 1],
                      rotate: [0, 180, 360]
                    } : {}}
                    transition={{ duration: 4, repeat: Infinity }}
                    className="absolute w-20 h-20 bg-gradient-to-br from-primary-500 to-accent-500 rounded-full flex items-center justify-center shadow-2xl z-20"
                  >
                    <Workflow className="w-10 h-10 text-white" />
                  </motion.div>

                  {/* Agent Nodes */}
                  {agents.map((agent, index) => {
                    const angle = (index * 360) / agents.length;
                    const radius = 120;
                    const x = 50 + (radius * Math.cos((angle * Math.PI) / 180)) / 3;
                    const y = 50 + (radius * Math.sin((angle * Math.PI) / 180)) / 3;

                    return (
                      <motion.div
                        key={index}
                        initial={{ scale: 0, opacity: 0 }}
                        animate={{ 
                          scale: isProcessing && selectedAgent === index ? 1.1 : 1,
                          opacity: 1
                        }}
                        transition={{ 
                          duration: 0.5,
                          delay: index * 0.1,
                          scale: { duration: 0.3 }
                        }}
                        className={`absolute w-14 h-14 rounded-xl flex items-center justify-center cursor-pointer shadow-lg z-10 ${
                          selectedAgent === index 
                            ? 'bg-gradient-to-br from-primary-500 to-primary-600 ring-2 ring-primary-400' 
                            : 'bg-gradient-to-br from-neutral-700 to-neutral-800 hover:from-neutral-600 hover:to-neutral-700'
                        }`}
                        style={{ 
                          left: `calc(${x}% - 28px)`, 
                          top: `calc(${y}% - 28px)`,
                          transform: `translate(-50%, -50%)`
                        }}
                        onClick={() => setSelectedAgent(index)}
                      >
                        <Bot className={`w-7 h-7 ${selectedAgent === index ? 'text-white' : 'text-neutral-300'}`} />
                        {isProcessing && selectedAgent === index && (
                          <motion.div
                            className="absolute inset-0 rounded-xl bg-white/20"
                            animate={{ opacity: [0, 0.5, 0] }}
                            transition={{ duration: 1, repeat: Infinity }}
                          />
                        )}
                      </motion.div>
                    );
                  })}

                  {/* Agent Labels */}
                  {agents.map((agent, index) => {
                    const angle = (index * 360) / agents.length;
                    const radius = 160;
                    const x = 50 + (radius * Math.cos((angle * Math.PI) / 180)) / 3;
                    const y = 50 + (radius * Math.sin((angle * Math.PI) / 180)) / 3;

                    return (
                      <motion.div
                        key={`label-${index}`}
                        initial={{ opacity: 0 }}
                        animate={{ opacity: selectedAgent === index ? 1 : 0 }}
                        className="absolute text-xs text-white font-medium pointer-events-none"
                        style={{ 
                          left: `${x}%`, 
                          top: `${y}%`,
                          transform: `translate(-50%, -50%)`
                        }}
                      >
                        {agent.name.split(' ')[0]}
                      </motion.div>
                    );
                  })}
                </div>

                {/* Agent Details */}
                <div className="mt-6 p-4 bg-neutral-800 rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-white font-medium">{agents[selectedAgent].name}</span>
                    <span className={`text-sm text-${agents[selectedAgent].color}-400`}>{agents[selectedAgent].status}</span>
                  </div>
                  <div className="h-2 bg-neutral-700 rounded-full overflow-hidden">
                    <motion.div
                      animate={{ width: `${agents[selectedAgent].progress}%` }}
                      className={`h-full bg-gradient-to-r from-${agents[selectedAgent].color}-400 to-${agents[selectedAgent].color}-600`}
                    />
                  </div>
                </div>
              </div>
            </motion.div>

            {/* Processing Stats Visual */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              className="relative"
            >
              <div className="bg-gradient-to-br from-neutral-800 to-neutral-900 rounded-2xl p-8 shadow-2xl border border-neutral-700 h-full flex flex-col">
                <h3 className="text-xl font-bold text-white mb-6">System Performance</h3>
                
                {/* Live Performance Metrics */}
                <div className="flex-grow flex flex-col justify-between">
                  {/* Stats Grid */}
                  <div className="grid grid-cols-2 gap-4 mb-6">
                    {[
                      { icon: Zap, value: '5x', label: 'Faster Processing', color: 'yellow' },
                      { icon: Cpu, value: '4', label: 'Worker Threads', color: 'blue' },
                      { icon: FileText, value: '92%', label: 'Accuracy Rate', color: 'green' },
                      { icon: Database, value: '10K+', label: 'Documents/Day', color: 'purple' }
                    ].map((stat, index) => (
                      <motion.div
                        key={index}
                        initial={{ scale: 0 }}
                        whileInView={{ scale: 1 }}
                        transition={{ delay: index * 0.1 }}
                        whileHover={{ scale: 1.05 }}
                        className="bg-neutral-800/50 rounded-lg p-4 border border-neutral-700 relative overflow-hidden group"
                      >
                        <motion.div
                          className={`absolute inset-0 bg-gradient-to-br from-${stat.color}-500/10 to-transparent`}
                          initial={{ x: '-100%' }}
                          whileHover={{ x: 0 }}
                          transition={{ duration: 0.3 }}
                        />
                        <div className="relative">
                          <stat.icon className={`w-5 h-5 text-${stat.color}-400 mb-2`} />
                          <div className="text-2xl font-bold text-white">{stat.value}</div>
                          <p className="text-xs text-neutral-400">{stat.label}</p>
                        </div>
                      </motion.div>
                    ))}
                  </div>

                  {/* Live Processing Animation */}
                  <div className="bg-neutral-900/50 rounded-lg p-4 mb-6">
                    <div className="flex items-center justify-between mb-3">
                      <span className="text-sm text-neutral-300">Real-time Processing</span>
                      <motion.div
                        animate={{ opacity: [0.5, 1, 0.5] }}
                        transition={{ duration: 1.5, repeat: Infinity }}
                        className="flex items-center space-x-1"
                      >
                        <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                        <span className="text-xs text-green-400">Live</span>
                      </motion.div>
                    </div>
                    
                    {/* Animated Progress Bars */}
                    <div className="space-y-3">
                      {[
                        { name: 'Document Analysis', progress: 75, color: 'primary' },
                        { name: 'Embedding Generation', progress: 60, color: 'accent' },
                        { name: 'Vector Storage', progress: 90, color: 'success' }
                      ].map((task, index) => (
                        <div key={index} className="space-y-1">
                          <div className="flex justify-between text-xs">
                            <span className="text-neutral-400">{task.name}</span>
                            <span className="text-neutral-500">{task.progress}%</span>
                          </div>
                          <div className="h-1.5 bg-neutral-800 rounded-full overflow-hidden">
                            <motion.div
                              className={`h-full bg-gradient-to-r from-${task.color}-500 to-${task.color}-600`}
                              initial={{ width: 0 }}
                              animate={{ width: `${task.progress}%` }}
                              transition={{ duration: 2, delay: index * 0.2 }}
                            />
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Format Support */}
                  <div>
                    <p className="text-sm font-semibold text-neutral-300 mb-3">Supported Formats</p>
                    <div className="grid grid-cols-4 gap-2">
                      {['PDF', 'DOCX', 'XLSX', 'CSV', 'HTML', 'TXT', 'MD', 'PPTX'].map((format, index) => (
                        <motion.div
                          key={format}
                          initial={{ opacity: 0, y: 10 }}
                          whileInView={{ opacity: 1, y: 0 }}
                          transition={{ delay: index * 0.05 }}
                          whileHover={{ scale: 1.1, rotate: 5 }}
                          className="bg-primary-900/20 text-primary-400 rounded-lg px-2 py-1.5 text-xs font-medium border border-primary-800/30 text-center cursor-default"
                        >
                          {format}
                        </motion.div>
                      ))}
                    </div>
                  </div>

                  {/* Flow Indicator */}
                  <div className="mt-6 p-3 bg-neutral-900/30 rounded-lg">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-xs text-neutral-400">Processing Pipeline</span>
                      <span className="text-xs text-primary-400">{animatedNumber} docs</span>
                    </div>
                    <div className="flex items-center space-x-1">
                      {[Upload, Binary, Database, Search].map((Icon, index) => (
                        <React.Fragment key={index}>
                          <motion.div
                            animate={isProcessing ? { scale: [1, 1.2, 1] } : {}}
                            transition={{ duration: 2, delay: index * 0.5, repeat: Infinity }}
                          >
                            <Icon className="w-4 h-4 text-neutral-500" />
                          </motion.div>
                          {index < 3 && (
                            <motion.div
                              className="flex-grow h-0.5 bg-neutral-700 relative overflow-hidden"
                            >
                              {isProcessing && (
                                <motion.div
                                  className="absolute inset-y-0 left-0 w-1/3 bg-gradient-to-r from-transparent via-primary-500 to-transparent"
                                  animate={{ x: ['-100%', '400%'] }}
                                  transition={{ duration: 2, delay: index * 0.5, repeat: Infinity }}
                                />
                              )}
                            </motion.div>
                          )}
                        </React.Fragment>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Technical Architecture CTA */}
      <section className="px-6 py-20 lg:px-12 relative bg-gradient-to-br from-neutral-900 via-primary-900/10 to-neutral-900">
        <div className="max-w-6xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="relative"
          >
            <div className="absolute inset-0 bg-gradient-to-r from-primary-500/20 to-accent-500/20 blur-3xl"></div>
            <div className="relative bg-gradient-to-br from-neutral-800/90 to-neutral-900/90 backdrop-blur-sm rounded-3xl p-8 lg:p-12 border border-primary-500/20 shadow-2xl">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-center">
                <div>
                  <div className="flex items-center space-x-2 mb-4">
                    <Binary className="w-6 h-6 text-primary-400" />
                    <span className="text-sm font-semibold text-primary-400 uppercase tracking-wider">For Engineers</span>
                  </div>
                  <h3 className="text-3xl lg:text-4xl font-bold mb-4">
                    Explore the <span className="text-primary-400">Technical Architecture</span>
                  </h3>
                  <p className="text-lg text-neutral-300 mb-6">
                    Deep dive into our multi-agent system, async processing pipeline, and production-grade infrastructure.
                    See the actual code, performance metrics, and implementation details.
                  </p>
                  <Link
                    to="/technical"
                    className="inline-flex items-center space-x-2 px-6 py-3 bg-primary-600 hover:bg-primary-700 text-white rounded-xl font-semibold transition-all transform hover:scale-105 group"
                  >
                    <Code2 className="w-5 h-5" />
                    <span>View Technical Docs</span>
                    <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
                  </Link>
                </div>
                <div className="relative">
                  <div className="grid grid-cols-2 gap-4">
                    <motion.div
                      whileHover={{ scale: 1.05 }}
                      className="bg-neutral-800/50 rounded-lg p-4 border border-neutral-700"
                    >
                      <div className="text-2xl font-bold text-primary-400 mb-1">6</div>
                      <div className="text-sm text-neutral-400">AI Agents</div>
                    </motion.div>
                    <motion.div
                      whileHover={{ scale: 1.05 }}
                      className="bg-neutral-800/50 rounded-lg p-4 border border-neutral-700"
                    >
                      <div className="text-2xl font-bold text-green-400 mb-1">5x</div>
                      <div className="text-sm text-neutral-400">Faster Processing</div>
                    </motion.div>
                    <motion.div
                      whileHover={{ scale: 1.05 }}
                      className="bg-neutral-800/50 rounded-lg p-4 border border-neutral-700"
                    >
                      <div className="text-2xl font-bold text-blue-400 mb-1">97%</div>
                      <div className="text-sm text-neutral-400">Accuracy</div>
                    </motion.div>
                    <motion.div
                      whileHover={{ scale: 1.05 }}
                      className="bg-neutral-800/50 rounded-lg p-4 border border-neutral-700"
                    >
                      <div className="text-2xl font-bold text-purple-400 mb-1">AsyncIO</div>
                      <div className="text-sm text-neutral-400">Parallel Execution</div>
                    </motion.div>
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Performance Dashboard */}
      <section id="performance" className="px-6 py-20 lg:px-12 lg:py-32 relative">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl lg:text-5xl font-display font-bold mb-4">
              Real-time <span className="gradient-text">Performance Metrics</span>
            </h2>
            <p className="text-xl text-neutral-600 max-w-3xl mx-auto">
              Monitor system performance and scale dynamically
            </p>
          </motion.div>

          {/* Live Metrics Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-12">
            {[
              { 
                label: "Docs/Minute", 
                value: "12-15", 
                change: "+5x",
                icon: FileText,
                chart: [20, 35, 40, 25, 45, 60, 55, 70, 65, 80]
              },
              { 
                label: "Query Response", 
                value: "2-3s", 
                change: "Fast",
                icon: Clock,
                chart: [80, 70, 65, 75, 60, 55, 45, 40, 35, 30]
              },
              { 
                label: "Active Agents", 
                value: "6", 
                change: "Ready",
                icon: Bot,
                chart: [30, 32, 35, 38, 40, 42, 44, 45, 47, 48]
              },
              { 
                label: "Relevance", 
                value: "92%", 
                change: "+3%",
                icon: Target,
                chart: [85, 86, 87, 88, 89, 90, 90, 91, 91, 92]
              }
            ].map((metric, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.1 }}
                whileHover={{ y: -5 }}
                className="bg-white/90 backdrop-blur-sm rounded-2xl p-6 shadow-lg hover:shadow-xl transition-all"
              >
                <div className="flex items-center justify-between mb-4">
                  <metric.icon className="w-8 h-8 text-primary-500" />
                  <span className={`text-sm font-medium ${metric.change.startsWith('+') ? 'text-success-600' : 'text-warning-600'}`}>
                    {metric.change}
                  </span>
                </div>
                <div className="text-3xl font-bold mb-2">{metric.value}</div>
                <div className="text-sm text-neutral-600 mb-3">{metric.label}</div>
                
                {/* Mini Chart */}
                <div className="h-12 flex items-end space-x-1">
                  {metric.chart.map((value, i) => (
                    <motion.div
                      key={i}
                      initial={{ height: 0 }}
                      animate={{ height: `${value * 0.5}px` }}
                      transition={{ delay: i * 0.05 }}
                      className="flex-1 bg-gradient-to-t from-primary-500 to-primary-300 rounded-t"
                    />
                  ))}
                </div>
              </motion.div>
            ))}
          </div>

          {/* System Status */}
          <motion.div
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            className="bg-gradient-to-br from-primary-600 to-accent-600 rounded-3xl p-8 text-white"
          >
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
              <div>
                <h3 className="text-xl font-bold mb-4">Infrastructure Status</h3>
                <div className="space-y-3">
                  {[
                    { name: "API Gateway", status: "Healthy", load: 45 },
                    { name: "Vector Database", status: "Optimal", load: 62 },
                    { name: "LLM Cluster", status: "Scaling", load: 78 },
                    { name: "Cache Layer", status: "Active", load: 35 }
                  ].map((service, index) => (
                    <div key={index} className="flex items-center justify-between">
                      <span className="text-sm">{service.name}</span>
                      <div className="flex items-center space-x-2">
                        <div className="w-24 h-2 bg-white/20 rounded-full overflow-hidden">
                          <motion.div
                            animate={{ width: `${service.load}%` }}
                            className="h-full bg-white/80"
                          />
                        </div>
                        <span className="text-xs text-white/60">{service.load}%</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div>
                <h3 className="text-xl font-bold mb-4">Recent Activity</h3>
                <div className="space-y-2">
                  {[
                    "PDF document processed (3.2s)",
                    "Query answered with 5 citations (2.8s)",
                    "Dataset created: 'reports' (1.5s)",
                    "Embeddings generated (0.8s)",
                    "Cache refreshed (0.3s)"
                  ].map((activity, index) => (
                    <motion.div
                      key={index}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: index * 0.1 }}
                      className="flex items-center space-x-2 text-sm"
                    >
                      <CheckCircle className="w-4 h-4 text-success-300" />
                      <span className="text-white/80">{activity}</span>
                    </motion.div>
                  ))}
                </div>
              </div>

              <div>
                <h3 className="text-xl font-bold mb-4">Auto-scaling</h3>
                <div className="space-y-4">
                  <div className="flex items-center justify-between p-3 bg-white/10 rounded-lg">
                    <span className="text-sm">CPU Usage</span>
                    <span className="font-bold">45%</span>
                  </div>
                  <div className="flex items-center justify-between p-3 bg-white/10 rounded-lg">
                    <span className="text-sm">Memory</span>
                    <span className="font-bold">2.1GB</span>
                  </div>
                  <button className="w-full px-4 py-2 bg-white/20 hover:bg-white/30 rounded-lg transition-colors flex items-center justify-center space-x-2">
                    <RefreshCw className="w-4 h-4" />
                    <span>Force Scale</span>
                  </button>
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Interactive Features Grid */}
      <section id="features" className="px-3 py-20 lg:px-12 lg:py-32 relative bg-gradient-to-br from-accent-50/30 to-primary-50/30">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl lg:text-5xl font-display font-bold mb-4">
              Powerful Features,
              <span className="gradient-text"> Infinite Possibilities</span>
            </h2>
            <p className="text-xl text-neutral-600 max-w-3xl mx-auto">
              Everything you need to build intelligent document systems
            </p>
          </motion.div>

          {/* 3D Feature Cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {[
              {
                icon: Rocket,
                title: "Instant Deployment",
                description: "One-command setup with Docker. Production-ready in minutes.",
                demo: "docker run -p 8000:8000 verixai/latest",
                color: "primary"
              },
              {
                icon: Package,
                title: "Plugin System",
                description: "Extend functionality with custom plugins and processors.",
                demo: "verix plugin install custom-nlp",
                color: "accent"
              },
              {
                icon: Cloud,
                title: "Cloud Native",
                description: "Kubernetes-ready with auto-scaling and load balancing.",
                demo: "kubectl apply -f verix-deploy.yaml",
                color: "success"
              },
              {
                icon: Command,
                title: "CLI Tools",
                description: "Powerful command-line interface for automation.",
                demo: "verix analyze --dataset legal --format json",
                color: "warning"
              },
              {
                icon: Braces,
                title: "REST API",
                description: "Comprehensive API for seamless integration.",
                demo: "POST /api/v1/analyze",
                color: "primary"
              },
              {
                icon: Shield,
                title: "Zero Trust Security",
                description: "End-to-end encryption with role-based access control.",
                demo: "verix security audit --verbose",
                color: "accent"
              }
            ].map((feature, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.1 }}
                whileHover={{ y: -10, rotateY: 5 }}
                className="relative group"
                style={{ transformStyle: 'preserve-3d' }}
              >
                <div className="absolute inset-0 bg-gradient-to-br from-primary-400/20 to-accent-400/20 rounded-2xl blur-xl group-hover:blur-2xl transition-all"></div>
                <div className="relative bg-white/90 backdrop-blur-sm rounded-2xl p-6 shadow-xl hover:shadow-2xl transition-all">
                  <div className={`w-14 h-14 bg-gradient-to-br from-${feature.color}-400 to-${feature.color}-600 rounded-xl flex items-center justify-center mb-4`}>
                    <feature.icon className="w-8 h-8 text-white" />
                  </div>
                  <h3 className="text-xl font-bold mb-3">{feature.title}</h3>
                  <p className="text-neutral-600 mb-4">{feature.description}</p>
                  
                  <div className="p-3 bg-neutral-900 rounded-lg">
                    <code className="text-xs text-green-400 font-mono">{feature.demo}</code>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Integration Ecosystem */}
      <section className="px-6 py-20 lg:px-12 lg:py-32">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl lg:text-5xl font-display font-bold mb-4">
              Seamless <span className="gradient-text">Integration</span>
            </h2>
            <p className="text-xl text-neutral-600 max-w-3xl mx-auto">
              Works with your existing tech stack out of the box
            </p>
          </motion.div>

          <div className="relative">
            <div className="absolute inset-0 bg-gradient-to-r from-primary-400/10 to-accent-400/10 blur-3xl"></div>
            
            {/* Circular Integration Display */}
            <div className="relative bg-white/80 backdrop-blur-sm rounded-3xl p-12 shadow-2xl">
              <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-8">
                {[
                  { name: "OpenAI", icon: "🤖" },
                  { name: "Claude", icon: "🧠" },
                  { name: "Ollama", icon: "🦙" },
                  { name: "PostgreSQL", icon: "🐘" },
                  { name: "Redis", icon: "🔴" },
                  { name: "Elasticsearch", icon: "🔍" },
                  { name: "Kafka", icon: "📊" },
                  { name: "S3", icon: "☁️" },
                  { name: "Slack", icon: "💬" },
                  { name: "Teams", icon: "👥" },
                  { name: "Jira", icon: "📋" },
                  { name: "GitHub", icon: "🐙" }
                ].map((integration, index) => (
                  <motion.div
                    key={index}
                    initial={{ opacity: 0, scale: 0.5 }}
                    whileInView={{ opacity: 1, scale: 1 }}
                    viewport={{ once: true }}
                    transition={{ delay: index * 0.05 }}
                    whileHover={{ scale: 1.1, rotate: 5 }}
                    className="flex flex-col items-center justify-center p-4 bg-gradient-to-br from-white to-neutral-100 rounded-xl shadow-lg hover:shadow-xl transition-all cursor-pointer"
                  >
                    <span className="text-3xl mb-2">{integration.icon}</span>
                    <span className="text-sm font-medium text-neutral-700">{integration.name}</span>
                  </motion.div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="px-6 py-20 lg:px-12 lg:py-32">
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          whileInView={{ opacity: 1, scale: 1 }}
          viewport={{ once: true }}
          className="max-w-4xl mx-auto text-center"
        >
          <div className="relative">
            <div className="absolute inset-0 bg-gradient-to-r from-primary-400/30 to-accent-400/30 blur-3xl"></div>
            <div className="relative bg-gradient-to-r from-primary-600 to-accent-600 rounded-3xl p-12 lg:p-16 shadow-2xl">
              <h2 className="text-3xl lg:text-5xl font-display font-bold text-white mb-6">
                Ready to Transform Your Document Workflow?
              </h2>
              <p className="text-xl text-white/90 mb-8 max-w-2xl mx-auto">
                Deploy your own intelligent document analysis system today
              </p>
              <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
                <Link
                  to="/dashboard"
                  className="inline-flex items-center space-x-2 px-8 py-4 bg-white text-primary-600 rounded-xl font-semibold text-lg hover:shadow-2xl transform hover:scale-105 transition-all duration-200"
                >
                  <span>Launch Platform</span>
                  <Rocket className="w-5 h-5" />
                </Link>
                <a
                  href="https://github.com/arunsai63/verix-ai"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center space-x-2 px-8 py-4 bg-white/20 backdrop-blur text-white border-2 border-white/30 rounded-xl font-semibold text-lg hover:bg-white/30 transform hover:scale-105 transition-all duration-200"
                >
                  <Github className="w-5 h-5" />
                  <span>View on GitHub</span>
                </a>
              </div>

              <div className="mt-12 grid grid-cols-3 gap-8 max-w-2xl mx-auto">
                <div className="text-center">
                  <div className="text-3xl font-bold text-white mb-1">100%</div>
                  <div className="text-sm text-white/70">Open Source</div>
                </div>
                <div className="text-center">
                  <div className="text-3xl font-bold text-white mb-1">24/7</div>
                  <div className="text-sm text-white/70">Support</div>
                </div>
                <div className="text-center">
                  <div className="text-3xl font-bold text-white mb-1">∞</div>
                  <div className="text-sm text-white/70">Scale</div>
                </div>
              </div>
            </div>
          </div>
        </motion.div>
      </section>

      {/* Footer */}
      <footer className="px-6 py-12 lg:px-12 border-t border-neutral-200">
        <div className="max-w-7xl mx-auto">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-8 mb-8">
            <div>
              <div className="flex items-center space-x-2 mb-4">
                <div className="w-8 h-8 bg-gradient-to-br from-primary-500 to-accent-500 rounded-lg flex items-center justify-center">
                  <Brain className="w-5 h-5 text-white" />
                </div>
                <span className="text-xl font-display font-bold">VerixAI</span>
              </div>
              <p className="text-neutral-600 text-sm">
                Enterprise-grade document intelligence platform.
                <br />
                <a href="https://arunsai63.github.io/verix-ai/" target="_blank" rel="noopener noreferrer" className="text-primary-600 hover:text-primary-700 transition-colors text-xs mt-2 inline-block">View Live Demo →</a>
              </p>
            </div>

            <div>
              <h4 className="font-semibold mb-4">Product</h4>
              <ul className="space-y-2 text-neutral-600 text-sm">
                <li><a href="#features" className="hover:text-primary-600 transition-colors">Features</a></li>
                <li><a href="https://github.com/arunsai63/verix-ai" target="_blank" rel="noopener noreferrer" className="hover:text-primary-600 transition-colors">GitHub</a></li>
                <li><a href="#" className="hover:text-primary-600 transition-colors">API Docs</a></li>
                <li><a href="#" className="hover:text-primary-600 transition-colors">Changelog</a></li>
              </ul>
            </div>

            <div>
              <h4 className="font-semibold mb-4">Developer</h4>
              <ul className="space-y-2 text-neutral-600 text-sm">
                <li><a href="https://arunsai63.github.io/portfolio" target="_blank" rel="noopener noreferrer" className="hover:text-primary-600 transition-colors">Portfolio</a></li>
                <li><a href="https://arunsai63.github.io/blogs" target="_blank" rel="noopener noreferrer" className="hover:text-primary-600 transition-colors">Blogs</a></li>
                <li><a href="https://arunsai63.github.io/resume.pdf" target="_blank" rel="noopener noreferrer" className="hover:text-primary-600 transition-colors">Resume</a></li>
                <li><a href="https://www.linkedin.com/in/arunmunaganti" target="_blank" rel="noopener noreferrer" className="hover:text-primary-600 transition-colors">Contact</a></li>
              </ul>
            </div>

            <div>
              <h4 className="font-semibold mb-4">Legal</h4>
              <ul className="space-y-2 text-neutral-600 text-sm">
                <li><a href="#" className="hover:text-primary-600 transition-colors">Privacy</a></li>
                <li><a href="#" className="hover:text-primary-600 transition-colors">Terms</a></li>
                <li><a href="#" className="hover:text-primary-600 transition-colors">Security</a></li>
                <li><a href="#" className="hover:text-primary-600 transition-colors">Compliance</a></li>
              </ul>
            </div>
          </div>

          <div className="border-t border-neutral-200 pt-8 flex flex-col md:flex-row items-center justify-between">
            <p className="text-neutral-600 text-sm mb-4 md:mb-0">
              © 2025 VerixAI. Built by <a href="https://arunsai63.github.io/" target="_blank" rel="noopener noreferrer" className="text-primary-600 hover:text-primary-700 transition-colors">Arun Munaganti</a>
            </p>
            <div className="flex items-center space-x-4">
              <a href="https://github.com/arunsai63" target="_blank" rel="noopener noreferrer" className="text-neutral-400 hover:text-primary-600 transition-colors">
                <Github className="w-5 h-5" />
              </a>
              <a href="https://www.linkedin.com/in/arunmunaganti" target="_blank" rel="noopener noreferrer" className="text-neutral-400 hover:text-primary-600 transition-colors">
                <Linkedin className="w-5 h-5" />
              </a>
              <a href="https://arunsai63.github.io/" target="_blank" rel="noopener noreferrer" className="text-neutral-400 hover:text-primary-600 transition-colors">
                <Globe className="w-5 h-5" />
              </a>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default LandingPage;