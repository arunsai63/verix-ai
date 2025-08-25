import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Upload,
  FileText,
  ArrowRight,
  Brain,
  Sparkles,
  CheckCircle,
  FileSearch,
  ChevronRight,
  Zap
} from 'lucide-react';

const InteractiveDemo: React.FC = () => {
  const [activeStep, setActiveStep] = useState(0);
  const [isProcessing, setIsProcessing] = useState(false);
  const [typedQuery, setTypedQuery] = useState('');
  const [showResult, setShowResult] = useState(false);

  const query = "What are the key findings in the quarterly report?";
  const demoSteps = [
    { id: 'upload', label: 'Upload Document', icon: Upload },
    { id: 'process', label: 'AI Processing', icon: Brain },
    { id: 'query', label: 'Ask Question', icon: FileSearch },
    { id: 'result', label: 'Get Answer', icon: Sparkles }
  ];

  const documentSamples = [
    { name: 'Q4_Report_2024.pdf', size: '2.4 MB', type: 'PDF' },
    { name: 'Financial_Analysis.docx', size: '1.8 MB', type: 'DOCX' },
    { name: 'Market_Research.pptx', size: '5.2 MB', type: 'PPTX' }
  ];

  const resultAnswer = {
    text: "Based on the quarterly report, here are the key findings:",
    highlights: [
      "Revenue increased by 34% YoY to $12.5M",
      "Customer acquisition cost decreased by 15%",
      "User retention improved to 95%",
      "New market expansion successful in 3 regions"
    ],
    citations: [
      { page: 3, text: "Executive Summary" },
      { page: 12, text: "Financial Performance" },
      { page: 24, text: "Market Analysis" }
    ],
    confidence: 'High'
  };

  useEffect(() => {
    if (activeStep === 2) {
      // Simulate typing animation
      let index = 0;
      const typingInterval = setInterval(() => {
        if (index < query.length) {
          setTypedQuery(query.substring(0, index + 1));
          index++;
        } else {
          clearInterval(typingInterval);
          setTimeout(() => {
            setActiveStep(3);
            setIsProcessing(true);
            setTimeout(() => {
              setIsProcessing(false);
              setShowResult(true);
            }, 2000);
          }, 500);
        }
      }, 50);

      return () => clearInterval(typingInterval);
    }
  }, [activeStep]);

  useEffect(() => {
    // Auto-advance through steps
    const timer = setTimeout(() => {
      if (activeStep < 3 && activeStep !== 2) {
        if (activeStep === 1) {
          setIsProcessing(true);
          setTimeout(() => {
            setIsProcessing(false);
            setActiveStep(activeStep + 1);
          }, 2000);
        } else {
          setActiveStep(activeStep + 1);
        }
      } else if (activeStep === 3 && showResult) {
        // Reset after showing result
        setTimeout(() => {
          setActiveStep(0);
          setTypedQuery('');
          setShowResult(false);
          setIsProcessing(false);
        }, 5000);
      }
    }, activeStep === 0 ? 3000 : activeStep === 1 ? 100 : 1000);

    return () => clearTimeout(timer);
  }, [activeStep, showResult]);

  return (
    <div className="relative w-full max-w-6xl mx-auto py-16">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center mb-12"
      >
        <h2 className="text-3xl lg:text-4xl font-display font-bold mb-4">
          See It In <span className="gradient-text">Action</span>
        </h2>
        <p className="text-lg text-neutral-600 max-w-2xl mx-auto">
          Watch how VerixAI transforms your documents into intelligent, searchable knowledge
        </p>
      </motion.div>

      {/* Progress Steps */}
      <div className="flex items-center justify-between mb-12 max-w-3xl mx-auto">
        {demoSteps.map((step, index) => (
          <React.Fragment key={step.id}>
            <motion.div
              className="flex flex-col items-center"
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ 
                opacity: activeStep >= index ? 1 : 0.5,
                scale: activeStep === index ? 1.1 : 1
              }}
              transition={{ duration: 0.3 }}
            >
              <div className={`
                w-12 h-12 rounded-full flex items-center justify-center transition-all duration-300
                ${activeStep >= index 
                  ? 'bg-gradient-to-br from-primary-500 to-accent-500 text-white shadow-lg' 
                  : 'bg-neutral-200 text-neutral-500'}
              `}>
                <step.icon className="w-6 h-6" />
              </div>
              <span className={`
                text-xs mt-2 font-medium transition-colors
                ${activeStep >= index ? 'text-primary-600' : 'text-neutral-500'}
              `}>
                {step.label}
              </span>
            </motion.div>
            {index < demoSteps.length - 1 && (
              <div className={`
                flex-1 h-1 mx-2 rounded-full transition-all duration-500
                ${activeStep > index 
                  ? 'bg-gradient-to-r from-primary-500 to-accent-500' 
                  : 'bg-neutral-200'}
              `} />
            )}
          </React.Fragment>
        ))}
      </div>

      {/* Demo Container */}
      <div className="relative">
        <div className="absolute inset-0 bg-gradient-to-r from-primary-400/10 to-accent-400/10 blur-3xl"></div>
        <motion.div 
          className="relative bg-white/90 backdrop-blur-sm rounded-2xl shadow-2xl overflow-hidden border border-white/50"
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5 }}
        >
          <div className="bg-gradient-to-r from-primary-500/10 to-accent-500/10 p-4 border-b border-neutral-200">
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 rounded-full bg-error-500"></div>
              <div className="w-3 h-3 rounded-full bg-warning-500"></div>
              <div className="w-3 h-3 rounded-full bg-success-500"></div>
              <span className="ml-4 text-sm font-medium text-neutral-600">VerixAI Document Analysis</span>
            </div>
          </div>

          <div className="p-8 min-h-[400px]">
            <AnimatePresence mode="wait">
              {/* Step 1: Upload */}
              {activeStep === 0 && (
                <motion.div
                  key="upload"
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: 20 }}
                  className="space-y-4"
                >
                  <div className="border-2 border-dashed border-primary-300 rounded-xl p-8 text-center bg-primary-50/50">
                    <Upload className="w-12 h-12 text-primary-500 mx-auto mb-3" />
                    <p className="text-neutral-700 font-medium mb-4">Drop your documents here</p>
                    <div className="space-y-2">
                      {documentSamples.map((doc, index) => (
                        <motion.div
                          key={doc.name}
                          initial={{ opacity: 0, y: 10 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ delay: index * 0.2 }}
                          className="flex items-center justify-between p-3 bg-white rounded-lg shadow-sm"
                        >
                          <div className="flex items-center space-x-3">
                            <FileText className="w-5 h-5 text-primary-500" />
                            <div className="text-left">
                              <p className="text-sm font-medium text-neutral-900">{doc.name}</p>
                              <p className="text-xs text-neutral-500">{doc.size}</p>
                            </div>
                          </div>
                          <CheckCircle className="w-5 h-5 text-success-500" />
                        </motion.div>
                      ))}
                    </div>
                  </div>
                </motion.div>
              )}

              {/* Step 2: Processing */}
              {activeStep === 1 && (
                <motion.div
                  key="process"
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.9 }}
                  className="flex flex-col items-center justify-center py-12"
                >
                  <div className="relative">
                    <motion.div
                      animate={{ rotate: 360 }}
                      transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                      className="w-20 h-20 rounded-full border-4 border-primary-200 border-t-primary-500"
                    />
                    <Brain className="w-10 h-10 text-primary-500 absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2" />
                  </div>
                  <p className="mt-6 text-lg font-medium text-neutral-700">Processing documents with AI...</p>
                  <div className="mt-4 flex space-x-2">
                    <motion.div
                      animate={{ opacity: [0.3, 1, 0.3] }}
                      transition={{ duration: 1.5, repeat: Infinity }}
                      className="w-2 h-2 rounded-full bg-primary-500"
                    />
                    <motion.div
                      animate={{ opacity: [0.3, 1, 0.3] }}
                      transition={{ duration: 1.5, repeat: Infinity, delay: 0.2 }}
                      className="w-2 h-2 rounded-full bg-primary-500"
                    />
                    <motion.div
                      animate={{ opacity: [0.3, 1, 0.3] }}
                      transition={{ duration: 1.5, repeat: Infinity, delay: 0.4 }}
                      className="w-2 h-2 rounded-full bg-primary-500"
                    />
                  </div>
                  <div className="mt-6 grid grid-cols-3 gap-4 text-center">
                    <div>
                      <p className="text-2xl font-bold text-primary-600">100%</p>
                      <p className="text-xs text-neutral-500">Extracted</p>
                    </div>
                    <div>
                      <p className="text-2xl font-bold text-accent-600">842</p>
                      <p className="text-xs text-neutral-500">Chunks</p>
                    </div>
                    <div>
                      <p className="text-2xl font-bold text-success-600">Ready</p>
                      <p className="text-xs text-neutral-500">Status</p>
                    </div>
                  </div>
                </motion.div>
              )}

              {/* Step 3: Query */}
              {activeStep === 2 && (
                <motion.div
                  key="query"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  className="space-y-4"
                >
                  <div className="bg-neutral-50 rounded-xl p-6">
                    <div className="flex items-start space-x-3">
                      <FileSearch className="w-6 h-6 text-primary-500 mt-1" />
                      <div className="flex-1">
                        <p className="text-sm text-neutral-500 mb-2">Ask your question:</p>
                        <div className="relative">
                          <p className="text-lg font-medium text-neutral-900 min-h-[28px]">
                            {typedQuery}
                            <motion.span
                              animate={{ opacity: [1, 0] }}
                              transition={{ duration: 0.5, repeat: Infinity }}
                              className="inline-block w-0.5 h-5 bg-primary-500 ml-1"
                            />
                          </p>
                        </div>
                      </div>
                    </div>
                  </div>
                  <div className="flex justify-end">
                    <button className="px-6 py-3 bg-gradient-to-r from-primary-500 to-accent-500 text-white rounded-xl font-medium flex items-center space-x-2">
                      <span>Analyze</span>
                      <ArrowRight className="w-4 h-4" />
                    </button>
                  </div>
                </motion.div>
              )}

              {/* Step 4: Result */}
              {activeStep === 3 && (
                <motion.div
                  key="result"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="space-y-4"
                >
                  {isProcessing ? (
                    <div className="flex items-center justify-center py-12">
                      <div className="flex items-center space-x-3">
                        <Zap className="w-6 h-6 text-primary-500 animate-pulse" />
                        <p className="text-lg font-medium text-neutral-700">Analyzing documents...</p>
                      </div>
                    </div>
                  ) : showResult && (
                    <motion.div
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="space-y-4"
                    >
                      {/* Answer */}
                      <div className="bg-gradient-to-r from-primary-50 to-accent-50 rounded-xl p-6">
                        <div className="flex items-start space-x-3 mb-4">
                          <Sparkles className="w-6 h-6 text-primary-500 mt-1" />
                          <div className="flex-1">
                            <div className="flex items-center justify-between mb-2">
                              <p className="font-medium text-neutral-900">AI Answer</p>
                              <span className="px-2 py-1 bg-success-100 text-success-700 text-xs rounded-full font-medium">
                                {resultAnswer.confidence} Confidence
                              </span>
                            </div>
                            <p className="text-neutral-700 mb-3">{resultAnswer.text}</p>
                            <ul className="space-y-2">
                              {resultAnswer.highlights.map((highlight, index) => (
                                <motion.li
                                  key={index}
                                  initial={{ opacity: 0, x: -10 }}
                                  animate={{ opacity: 1, x: 0 }}
                                  transition={{ delay: index * 0.1 }}
                                  className="flex items-start space-x-2"
                                >
                                  <ChevronRight className="w-4 h-4 text-primary-500 mt-0.5 flex-shrink-0" />
                                  <span className="text-sm text-neutral-700">{highlight}</span>
                                </motion.li>
                              ))}
                            </ul>
                          </div>
                        </div>
                      </div>

                      {/* Citations */}
                      <div className="bg-white rounded-xl p-4 border border-neutral-200">
                        <p className="text-sm font-medium text-neutral-700 mb-3">Citations:</p>
                        <div className="flex flex-wrap gap-2">
                          {resultAnswer.citations.map((citation, index) => (
                            <motion.div
                              key={index}
                              initial={{ opacity: 0, scale: 0.8 }}
                              animate={{ opacity: 1, scale: 1 }}
                              transition={{ delay: 0.3 + index * 0.1 }}
                              className="px-3 py-1.5 bg-neutral-100 rounded-lg text-xs text-neutral-700 flex items-center space-x-2"
                            >
                              <FileText className="w-3 h-3" />
                              <span>Page {citation.page}: {citation.text}</span>
                            </motion.div>
                          ))}
                        </div>
                      </div>
                    </motion.div>
                  )}
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </motion.div>
      </div>

      {/* Features highlight */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5 }}
        className="mt-12 grid grid-cols-1 md:grid-cols-3 gap-6 max-w-4xl mx-auto"
      >
        <div className="text-center">
          <div className="w-12 h-12 bg-primary-100 rounded-xl flex items-center justify-center mx-auto mb-3">
            <Zap className="w-6 h-6 text-primary-600" />
          </div>
          <h3 className="font-semibold text-neutral-900 mb-1">Lightning Fast</h3>
          <p className="text-sm text-neutral-600">Get answers in seconds, not hours</p>
        </div>
        <div className="text-center">
          <div className="w-12 h-12 bg-accent-100 rounded-xl flex items-center justify-center mx-auto mb-3">
            <Brain className="w-6 h-6 text-accent-600" />
          </div>
          <h3 className="font-semibold text-neutral-900 mb-1">AI-Powered</h3>
          <p className="text-sm text-neutral-600">Advanced language models for accuracy</p>
        </div>
        <div className="text-center">
          <div className="w-12 h-12 bg-success-100 rounded-xl flex items-center justify-center mx-auto mb-3">
            <CheckCircle className="w-6 h-6 text-success-600" />
          </div>
          <h3 className="font-semibold text-neutral-900 mb-1">Verified Citations</h3>
          <p className="text-sm text-neutral-600">Every answer backed by source documents</p>
        </div>
      </motion.div>
    </div>
  );
};

export default InteractiveDemo;