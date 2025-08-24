import React, { useState } from 'react';
import { motion } from 'framer-motion';
import {
  HelpCircle,
  Book,
  MessageCircle,
  Video,
  FileText,
  ExternalLink,
  ChevronDown,
  ChevronUp,
  Mail,
  Github,
  Globe,
  Zap,
  Search,
  Upload,
  Database,
} from 'lucide-react';
import Card from '../components/ui/Card';
import Button from '../components/ui/Button';
import Badge from '../components/ui/Badge';
import Input from '../components/ui/Input';

interface FAQItem {
  question: string;
  answer: string;
  category: string;
}

const Help: React.FC = () => {
  const [expandedFAQ, setExpandedFAQ] = useState<number | null>(null);
  const [searchQuery, setSearchQuery] = useState('');

  const faqs: FAQItem[] = [
    {
      question: 'How do I upload documents?',
      answer: 'Navigate to the Upload section from the sidebar. You can drag and drop files or click to browse. Supported formats include PDF, DOCX, PPTX, HTML, TXT, MD, CSV, and XLSX. Files can be up to 100MB each.',
      category: 'Getting Started',
    },
    {
      question: 'What is a dataset?',
      answer: 'A dataset is a collection of related documents that are processed and indexed together. You can organize your documents into different datasets for better organization and more targeted queries.',
      category: 'Datasets',
    },
    {
      question: 'How accurate are the AI-generated answers?',
      answer: 'Our AI provides answers with confidence scores (high, medium, low) and always includes citations to the source documents. The accuracy depends on the quality and relevance of your uploaded documents.',
      category: 'Queries',
    },
    {
      question: 'Can I query across multiple datasets?',
      answer: 'Yes! When making a query, you can select multiple datasets or leave the selection empty to search across all your documents.',
      category: 'Queries',
    },
    {
      question: 'What are the different role contexts?',
      answer: 'Role contexts help tailor the AI responses: General (default analysis), Doctor (medical context with health disclaimers), Lawyer (legal context with legal disclaimers), and HR (compliance-focused responses).',
      category: 'Queries',
    },
    {
      question: 'How do I delete a dataset?',
      answer: 'Go to the Datasets section, find the dataset you want to delete, and click the Delete button. Note that this action is permanent and will remove all documents in that dataset.',
      category: 'Datasets',
    },
    {
      question: 'What file formats are supported?',
      answer: 'We support PDF, DOCX, PPTX, HTML, TXT, MD (Markdown), CSV, and XLSX files. Each file can be up to 100MB in size.',
      category: 'Getting Started',
    },
    {
      question: 'How are my documents processed?',
      answer: 'Documents are processed using advanced NLP techniques. They are chunked into smaller segments, embedded into vectors, and stored in a vector database for efficient semantic search.',
      category: 'Technical',
    },
  ];

  const filteredFAQs = faqs.filter(
    faq =>
      faq.question.toLowerCase().includes(searchQuery.toLowerCase()) ||
      faq.answer.toLowerCase().includes(searchQuery.toLowerCase()) ||
      faq.category.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const categories = Array.from(new Set(faqs.map(faq => faq.category)));

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="max-w-6xl mx-auto space-y-6 p-6"
    >
      {/* Header */}
      <div>
        <h1 className="text-3xl font-display font-bold text-neutral-900 dark:text-neutral-100 mb-2">
          Help Center
        </h1>
        <p className="text-neutral-600 dark:text-neutral-400">
          Find answers to common questions and learn how to use VerixAI effectively
        </p>
      </div>

      {/* Quick Links */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card variant="default" className="hover:shadow-lg transition-shadow cursor-pointer">
          <div className="p-6">
            <div className="w-12 h-12 bg-primary-100 dark:bg-primary-900/30 rounded-xl flex items-center justify-center mb-4">
              <Book className="w-6 h-6 text-primary-600 dark:text-primary-400" />
            </div>
            <h3 className="font-semibold text-neutral-900 dark:text-neutral-100 mb-2">
              Documentation
            </h3>
            <p className="text-sm text-neutral-600 dark:text-neutral-400">
              Comprehensive guides and API reference
            </p>
          </div>
        </Card>

        <Card variant="default" className="hover:shadow-lg transition-shadow cursor-pointer">
          <div className="p-6">
            <div className="w-12 h-12 bg-accent-100 dark:bg-accent-900/30 rounded-xl flex items-center justify-center mb-4">
              <Video className="w-6 h-6 text-accent-600 dark:text-accent-400" />
            </div>
            <h3 className="font-semibold text-neutral-900 dark:text-neutral-100 mb-2">
              Video Tutorials
            </h3>
            <p className="text-sm text-neutral-600 dark:text-neutral-400">
              Step-by-step video walkthroughs
            </p>
          </div>
        </Card>

        <Card variant="default" className="hover:shadow-lg transition-shadow cursor-pointer">
          <div className="p-6">
            <div className="w-12 h-12 bg-success-100 dark:bg-success-900/30 rounded-xl flex items-center justify-center mb-4">
              <MessageCircle className="w-6 h-6 text-success-600 dark:text-success-400" />
            </div>
            <h3 className="font-semibold text-neutral-900 dark:text-neutral-100 mb-2">
              Community
            </h3>
            <p className="text-sm text-neutral-600 dark:text-neutral-400">
              Join our Discord community
            </p>
          </div>
        </Card>

        <Card variant="default" className="hover:shadow-lg transition-shadow cursor-pointer">
          <div className="p-6">
            <div className="w-12 h-12 bg-warning-100 dark:bg-warning-900/30 rounded-xl flex items-center justify-center mb-4">
              <Zap className="w-6 h-6 text-warning-600 dark:text-warning-400" />
            </div>
            <h3 className="font-semibold text-neutral-900 dark:text-neutral-100 mb-2">
              API Status
            </h3>
            <p className="text-sm text-neutral-600 dark:text-neutral-400">
              Check service status and uptime
            </p>
          </div>
        </Card>
      </div>

      {/* Getting Started Guide */}
      <Card variant="gradient">
        <div className="p-8">
          <h2 className="text-2xl font-display font-semibold text-neutral-900 dark:text-neutral-100 mb-6">
            Getting Started with VerixAI
          </h2>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="flex items-start space-x-4">
              <div className="w-10 h-10 bg-primary-500 rounded-full flex items-center justify-center flex-shrink-0">
                <span className="text-white font-bold">1</span>
              </div>
              <div>
                <h3 className="font-semibold text-neutral-900 dark:text-neutral-100 mb-1">
                  <Upload className="w-4 h-4 inline mr-2" />
                  Upload Documents
                </h3>
                <p className="text-sm text-neutral-600 dark:text-neutral-400">
                  Start by uploading your documents to create or add to a dataset
                </p>
              </div>
            </div>

            <div className="flex items-start space-x-4">
              <div className="w-10 h-10 bg-primary-500 rounded-full flex items-center justify-center flex-shrink-0">
                <span className="text-white font-bold">2</span>
              </div>
              <div>
                <h3 className="font-semibold text-neutral-900 dark:text-neutral-100 mb-1">
                  <Database className="w-4 h-4 inline mr-2" />
                  Organize Datasets
                </h3>
                <p className="text-sm text-neutral-600 dark:text-neutral-400">
                  Group related documents into datasets for better organization
                </p>
              </div>
            </div>

            <div className="flex items-start space-x-4">
              <div className="w-10 h-10 bg-primary-500 rounded-full flex items-center justify-center flex-shrink-0">
                <span className="text-white font-bold">3</span>
              </div>
              <div>
                <h3 className="font-semibold text-neutral-900 dark:text-neutral-100 mb-1">
                  <Search className="w-4 h-4 inline mr-2" />
                  Query & Analyze
                </h3>
                <p className="text-sm text-neutral-600 dark:text-neutral-400">
                  Ask questions and get AI-powered answers with citations
                </p>
              </div>
            </div>
          </div>
        </div>
      </Card>

      {/* FAQ Section */}
      <div>
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-2xl font-display font-semibold text-neutral-900 dark:text-neutral-100">
            Frequently Asked Questions
          </h2>
          <div className="flex space-x-2">
            {categories.map(category => (
              <Badge key={category} variant="outline" size="sm">
                {category}
              </Badge>
            ))}
          </div>
        </div>

        {/* Search FAQ */}
        <div className="mb-6">
          <Input
            placeholder="Search FAQs..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            leftIcon={<Search className="w-4 h-4" />}
          />
        </div>

        {/* FAQ Items */}
        <div className="space-y-3">
          {filteredFAQs.map((faq, index) => (
            <Card key={index} variant="default">
              <button
                onClick={() => setExpandedFAQ(expandedFAQ === index ? null : index)}
                className="w-full p-4 flex items-center justify-between text-left hover:bg-neutral-50 dark:hover:bg-neutral-800 transition-colors rounded-xl"
              >
                <div className="flex items-start space-x-3">
                  <HelpCircle className="w-5 h-5 text-primary-500 mt-0.5" />
                  <div>
                    <h3 className="font-medium text-neutral-900 dark:text-neutral-100">
                      {faq.question}
                    </h3>
                    <Badge variant="outline" size="sm" className="mt-1">
                      {faq.category}
                    </Badge>
                  </div>
                </div>
                {expandedFAQ === index ? (
                  <ChevronUp className="w-5 h-5 text-neutral-400" />
                ) : (
                  <ChevronDown className="w-5 h-5 text-neutral-400" />
                )}
              </button>
              {expandedFAQ === index && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  className="px-4 pb-4"
                >
                  <p className="text-neutral-600 dark:text-neutral-400 pl-8">
                    {faq.answer}
                  </p>
                </motion.div>
              )}
            </Card>
          ))}
        </div>
      </div>

      {/* Contact Section */}
      <Card variant="gradient">
        <div className="p-8">
          <h2 className="text-2xl font-display font-semibold text-neutral-900 dark:text-neutral-100 mb-6">
            Still Need Help?
          </h2>
          <p className="text-neutral-600 dark:text-neutral-400 mb-6">
            Can't find what you're looking for? Our support team is here to help.
          </p>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Button
              variant="outline"
              leftIcon={<Mail className="w-4 h-4" />}
              className="justify-center"
            >
              Email Support
            </Button>
            <Button
              variant="outline"
              leftIcon={<Github className="w-4 h-4" />}
              className="justify-center"
            >
              GitHub Issues
            </Button>
            <Button
              variant="outline"
              leftIcon={<Globe className="w-4 h-4" />}
              className="justify-center"
            >
              Visit Website
            </Button>
          </div>
        </div>
      </Card>

      {/* Resources */}
      <div>
        <h2 className="text-2xl font-display font-semibold text-neutral-900 dark:text-neutral-100 mb-4">
          Additional Resources
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Card variant="default">
            <div className="p-4 flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <FileText className="w-5 h-5 text-neutral-500" />
                <div>
                  <h3 className="font-medium text-neutral-900 dark:text-neutral-100">
                    API Documentation
                  </h3>
                  <p className="text-sm text-neutral-600 dark:text-neutral-400">
                    Learn how to integrate VerixAI API
                  </p>
                </div>
              </div>
              <ExternalLink className="w-4 h-4 text-neutral-400" />
            </div>
          </Card>

          <Card variant="default">
            <div className="p-4 flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <FileText className="w-5 h-5 text-neutral-500" />
                <div>
                  <h3 className="font-medium text-neutral-900 dark:text-neutral-100">
                    Best Practices Guide
                  </h3>
                  <p className="text-sm text-neutral-600 dark:text-neutral-400">
                    Tips for optimal document analysis
                  </p>
                </div>
              </div>
              <ExternalLink className="w-4 h-4 text-neutral-400" />
            </div>
          </Card>
        </div>
      </div>
    </motion.div>
  );
};

export default Help;