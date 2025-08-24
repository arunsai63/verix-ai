import React from 'react';
import { motion } from 'framer-motion';
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
  Twitter,
  Linkedin,
  Globe
} from 'lucide-react';

const LandingPage: React.FC = () => {
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
            <a href="#how-it-works" className="text-neutral-600 hover:text-primary-600 transition-colors">How it Works</a>
            <a href="#use-cases" className="text-neutral-600 hover:text-primary-600 transition-colors">Use Cases</a>
            <a href="#pricing" className="text-neutral-600 hover:text-primary-600 transition-colors">Pricing</a>
            <Link
              to="/dashboard"
              className="px-6 py-2.5 bg-gradient-to-r from-primary-500 to-primary-600 text-white rounded-xl font-medium hover:shadow-lg transform hover:scale-105 transition-all duration-200"
            >
              Dashboard
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
              <Zap className="w-4 h-4 text-primary-600" />
              <span className="text-primary-700 font-medium text-sm">AI-Powered Document Intelligence</span>
            </motion.div>

            <motion.h1
              variants={fadeIn}
              className="text-5xl lg:text-7xl font-display font-bold mb-6"
            >
              Transform Your Documents Into
              <span className="block mt-2 gradient-text">Actionable Intelligence</span>
            </motion.h1>

            <motion.p
              variants={fadeIn}
              className="text-xl text-neutral-600 mb-8 leading-relaxed"
            >
              Upload any document, ask any question, and get instant, accurate answers
              with precise citations. Built for professionals who need reliable document analysis.
            </motion.p>

            <motion.div
              variants={fadeIn}
              className="flex flex-col sm:flex-row gap-4 justify-center items-center"
            >
              <Link
                to="/dashboard"
                className="group px-8 py-4 bg-gradient-to-r from-primary-500 to-primary-600 text-white rounded-xl font-semibold text-lg hover:shadow-2xl transform hover:scale-105 transition-all duration-200 flex items-center space-x-2"
              >
                <span>Start Analyzing</span>
                <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
              </Link>

              <button className="px-8 py-4 bg-white/80 backdrop-blur-sm border-2 border-primary-200 text-primary-600 rounded-xl font-semibold text-lg hover:bg-white hover:shadow-lg transform hover:scale-105 transition-all duration-200">
                Watch Demo
              </button>
            </motion.div>

            <motion.div
              variants={fadeIn}
              className="mt-12 flex items-center justify-center space-x-8 text-neutral-500"
            >
              <div className="flex items-center space-x-2">
                <CheckCircle className="w-5 h-5 text-success-500" />
                <span>No credit card required</span>
              </div>
              <div className="flex items-center space-x-2">
                <Shield className="w-5 h-5 text-primary-500" />
                <span>Enterprise-grade security</span>
              </div>
            </motion.div>
          </motion.div>

          {/* Hero Image/Demo */}
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5, duration: 0.8 }}
            className="mt-20 relative"
          >
            <div className="relative mx-auto max-w-6xl">
              <div className="absolute inset-0 bg-gradient-to-r from-primary-400/20 to-accent-400/20 blur-3xl"></div>
              <div className="relative glass rounded-2xl p-8 shadow-2xl">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <motion.div
                    whileHover={{ scale: 1.05 }}
                    className="bg-white/90 backdrop-blur rounded-xl p-6 shadow-lg hover:shadow-xl transition-all"
                  >
                    <Upload className="w-10 h-10 text-primary-500 mb-4" />
                    <h3 className="font-semibold text-lg mb-2">Upload Documents</h3>
                    <p className="text-neutral-600 text-sm">Support for PDF, DOCX, PPTX, and more</p>
                  </motion.div>

                  <motion.div
                    whileHover={{ scale: 1.05 }}
                    className="bg-white/90 backdrop-blur rounded-xl p-6 shadow-lg hover:shadow-xl transition-all"
                  >
                    <Search className="w-10 h-10 text-accent-500 mb-4" />
                    <h3 className="font-semibold text-lg mb-2">Ask Questions</h3>
                    <p className="text-neutral-600 text-sm">Natural language queries with context</p>
                  </motion.div>

                  <motion.div
                    whileHover={{ scale: 1.05 }}
                    className="bg-white/90 backdrop-blur rounded-xl p-6 shadow-lg hover:shadow-xl transition-all"
                  >
                    <FileSearch className="w-10 h-10 text-success-500 mb-4" />
                    <h3 className="font-semibold text-lg mb-2">Get Insights</h3>
                    <p className="text-neutral-600 text-sm">AI-powered answers with citations</p>
                  </motion.div>
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="px-6 py-20 lg:px-12 lg:py-32 relative">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl lg:text-5xl font-display font-bold mb-4">
              Powerful Features for
              <span className="gradient-text"> Every Need</span>
            </h2>
            <p className="text-xl text-neutral-600 max-w-3xl mx-auto">
              Built with cutting-edge AI technology to deliver accurate, contextual answers from your documents
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {[
              {
                icon: Brain,
                title: "Advanced AI Processing",
                description: "Powered by GPT-4 and state-of-the-art embeddings for superior understanding",
                color: "primary"
              },
              {
                icon: Shield,
                title: "Enterprise Security",
                description: "Your documents are encrypted and never used for training AI models",
                color: "accent"
              },
              {
                icon: Zap,
                title: "Lightning Fast",
                description: "Get answers in seconds, not minutes, with our optimized pipeline",
                color: "warning"
              },
              {
                icon: Users,
                title: "Role-Based Responses",
                description: "Tailored answers for doctors, lawyers, HR professionals, and more",
                color: "success"
              },
              {
                icon: Database,
                title: "Dataset Management",
                description: "Organize documents into searchable collections for better results",
                color: "primary"
              },
              {
                icon: BarChart3,
                title: "Confidence Scoring",
                description: "Know how reliable each answer is with our confidence indicators",
                color: "accent"
              }
            ].map((feature, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.1 }}
                whileHover={{ scale: 1.05 }}
                className="bg-white/80 backdrop-blur-sm rounded-2xl p-8 shadow-lg hover:shadow-2xl transition-all duration-300 border border-white/50"
              >
                <div className={`w-14 h-14 bg-gradient-to-br from-${feature.color}-400 to-${feature.color}-600 rounded-xl flex items-center justify-center mb-6`}>
                  <feature.icon className="w-8 h-8 text-white" />
                </div>
                <h3 className="text-xl font-semibold mb-3">{feature.title}</h3>
                <p className="text-neutral-600 leading-relaxed">{feature.description}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* How It Works Section */}
      <section id="how-it-works" className="px-6 py-20 lg:px-12 lg:py-32 bg-gradient-to-br from-primary-50/50 to-accent-50/30 relative">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl lg:text-5xl font-display font-bold mb-4">
              How It Works
            </h2>
            <p className="text-xl text-neutral-600 max-w-3xl mx-auto">
              Three simple steps to unlock insights from your documents
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 relative">
            {/* Connection lines */}
            <div className="hidden md:block absolute top-1/2 left-1/4 right-1/4 h-0.5 bg-gradient-to-r from-primary-300 to-accent-300"></div>

            {[
              {
                step: "01",
                title: "Upload Your Documents",
                description: "Drag and drop your files or browse to upload. We support PDF, Word, PowerPoint, and more.",
                icon: Upload
              },
              {
                step: "02",
                title: "Ask Your Questions",
                description: "Type your questions in natural language. Our AI understands context and nuance.",
                icon: Search
              },
              {
                step: "03",
                title: "Get Cited Answers",
                description: "Receive accurate answers with specific citations to source documents.",
                icon: FileSearch
              }
            ].map((step, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.2 }}
                className="relative"
              >
                <div className="bg-white rounded-2xl p-8 shadow-xl hover:shadow-2xl transition-all duration-300 relative z-10">
                  <div className="text-5xl font-bold gradient-text mb-4">{step.step}</div>
                  <step.icon className="w-12 h-12 text-primary-500 mb-4" />
                  <h3 className="text-xl font-semibold mb-3">{step.title}</h3>
                  <p className="text-neutral-600">{step.description}</p>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Use Cases Section */}
      <section id="use-cases" className="px-6 py-20 lg:px-12 lg:py-32">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl lg:text-5xl font-display font-bold mb-4">
              Built for <span className="gradient-text">Professionals</span>
            </h2>
            <p className="text-xl text-neutral-600 max-w-3xl mx-auto">
              Tailored solutions for different industries and use cases
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {[
              {
                title: "Healthcare",
                description: "Analyze medical records, research papers, and clinical guidelines",
                features: ["Patient history analysis", "Drug interaction checks", "Clinical decision support"],
                gradient: "from-success-400 to-success-600"
              },
              {
                title: "Legal",
                description: "Review contracts, case law, and legal documents efficiently",
                features: ["Contract analysis", "Precedent research", "Compliance checking"],
                gradient: "from-primary-400 to-primary-600"
              },
              {
                title: "Human Resources",
                description: "Process resumes, policies, and employee documents",
                features: ["Resume screening", "Policy compliance", "Benefits analysis"],
                gradient: "from-accent-400 to-accent-600"
              },
              {
                title: "Research & Education",
                description: "Extract insights from academic papers and research documents",
                features: ["Literature review", "Citation tracking", "Knowledge synthesis"],
                gradient: "from-warning-400 to-warning-600"
              }
            ].map((useCase, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, x: index % 2 === 0 ? -20 : 20 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.1 }}
                className="relative group"
              >
                <div className={`absolute inset-0 bg-gradient-to-r ${useCase.gradient} opacity-10 rounded-2xl blur-xl group-hover:opacity-20 transition-opacity`}></div>
                <div className="relative bg-white/90 backdrop-blur-sm rounded-2xl p-8 shadow-lg hover:shadow-2xl transition-all duration-300 border border-white/50">
                  <h3 className="text-2xl font-semibold mb-3">{useCase.title}</h3>
                  <p className="text-neutral-600 mb-6">{useCase.description}</p>
                  <ul className="space-y-2">
                    {useCase.features.map((feature, fIndex) => (
                      <li key={fIndex} className="flex items-center space-x-2">
                        <CheckCircle className="w-5 h-5 text-success-500 flex-shrink-0" />
                        <span className="text-neutral-700">{feature}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              </motion.div>
            ))}
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
                Join thousands of professionals who are already saving hours every week with VerixAI
              </p>
              <Link
                to="/dashboard"
                className="inline-flex items-center space-x-2 px-8 py-4 bg-white text-primary-600 rounded-xl font-semibold text-lg hover:shadow-2xl transform hover:scale-105 transition-all duration-200"
              >
                <span>Get Started Free</span>
                <ArrowRight className="w-5 h-5" />
              </Link>

              <div className="mt-8 flex items-center justify-center space-x-8">
                {/* <div className="d-none flex -space-x-2">
                  {[1, 2, 3, 4, 5].map((i) => (
                    <div key={i} className="w-10 h-10 rounded-full bg-white/20 backdrop-blur border-2 border-white/30"></div>
                  ))}
                </div> */}
                <div className="text-white/90">
                  {/* <div className="flex items-center space-x-1 mb-1">
                    {[1, 2, 3, 4, 5].map((i) => (
                      <Star key={i} className="w-4 h-4 fill-warning-400 text-warning-400" />
                    ))}
                  </div>
                  <p className="text-sm">Trusted by 10,000+ users</p> */}
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
                AI-powered document intelligence for modern professionals.
              </p>
            </div>

            <div>
              <h4 className="font-semibold mb-4">Product</h4>
              <ul className="space-y-2 text-neutral-600 text-sm">
                <li><a href="#" className="hover:text-primary-600 transition-colors">Features</a></li>
                <li><a href="#" className="hover:text-primary-600 transition-colors">Pricing</a></li>
                <li><a href="#" className="hover:text-primary-600 transition-colors">API Docs</a></li>
                <li><a href="#" className="hover:text-primary-600 transition-colors">Changelog</a></li>
              </ul>
            </div>

            <div>
              <h4 className="font-semibold mb-4">Company</h4>
              <ul className="space-y-2 text-neutral-600 text-sm">
                <li><a href="#" className="hover:text-primary-600 transition-colors">About</a></li>
                <li><a href="#" className="hover:text-primary-600 transition-colors">Blog</a></li>
                <li><a href="#" className="hover:text-primary-600 transition-colors">Careers</a></li>
                <li><a href="#" className="hover:text-primary-600 transition-colors">Contact</a></li>
              </ul>
            </div>

            <div>
              <h4 className="font-semibold mb-4">Legal</h4>
              <ul className="space-y-2 text-neutral-600 text-sm">
                <li><a href="#" className="hover:text-primary-600 transition-colors">Privacy</a></li>
                <li><a href="#" className="hover:text-primary-600 transition-colors">Terms</a></li>
                <li><a href="#" className="hover:text-primary-600 transition-colors">Security</a></li>
                <li><a href="#" className="hover:text-primary-600 transition-colors">GDPR</a></li>
              </ul>
            </div>
          </div>

          <div className="border-t border-neutral-200 pt-8 flex flex-col md:flex-row items-center justify-between">
            <p className="text-neutral-600 text-sm mb-4 md:mb-0">
              © 2025 VerixAI. All rights reserved.
            </p>
            <div className="flex items-center space-x-4">
              <a href="#" className="text-neutral-400 hover:text-primary-600 transition-colors">
                <Twitter className="w-5 h-5" />
              </a>
              <a href="#" className="text-neutral-400 hover:text-primary-600 transition-colors">
                <Github className="w-5 h-5" />
              </a>
              <a href="#" className="text-neutral-400 hover:text-primary-600 transition-colors">
                <Linkedin className="w-5 h-5" />
              </a>
              <a href="#" className="text-neutral-400 hover:text-primary-600 transition-colors">
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