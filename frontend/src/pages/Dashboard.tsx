import React, { useState, useEffect } from 'react';
import { Routes, Route, NavLink, useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Upload,
  Search,
  Database,
  Brain,
  Menu,
  X,
  Home,
  Settings,
  HelpCircle,
  LogOut,
  Sun,
  Moon,
  Bell,
  User,
  BookOpen,
  MessageCircle
} from 'lucide-react';
import UploadSection from '../components/UploadSection';
import QuerySection from '../components/QuerySection';
import DatasetsSection from '../components/DatasetsSection';
import SummarizationSection from '../components/SummarizationSection';
import ChatSection from '../components/ChatSection';
import SettingsPage from './Settings';
import HelpPage from './Help';
import { Dataset, QueryResult } from '../types';
import api from '../services/api';
import Button from '../components/ui/Button';
import Badge from '../components/ui/Badge';
import '../styles/dashboard.css';

const Dashboard: React.FC = () => {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [darkMode, setDarkMode] = useState(false);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [queryResult, setQueryResult] = useState<QueryResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [notification, setNotification] = useState<{
    open: boolean;
    message: string;
    type: 'success' | 'error' | 'info' | 'warning';
  }>({
    open: false,
    message: '',
    type: 'info',
  });

  const navigate = useNavigate();

  useEffect(() => {
    loadDatasets();
  }, []);

  const loadDatasets = async () => {
    try {
      setLoading(true);
      const response = await api.getDatasets();
      setDatasets(response.data);
    } catch (error) {
      console.error('Failed to load datasets:', error);
      showNotification('Failed to load datasets', 'error');
    } finally {
      setLoading(false);
    }
  };

  const showNotification = (message: string, type: 'success' | 'error' | 'info' | 'warning' = 'info') => {
    setNotification({
      open: true,
      message,
      type,
    });
    setTimeout(() => {
      setNotification(prev => ({ ...prev, open: false }));
    }, 5000);
  };

  const handleUploadComplete = () => {
    loadDatasets();
    showNotification('Documents uploaded successfully!', 'success');
  };

  const handleQuery = async (result: QueryResult) => {
    setQueryResult(result);
    navigate('/dashboard/query');
  };

  const handleDatasetDeleted = () => {
    loadDatasets();
    showNotification('Dataset deleted successfully', 'success');
  };

  const navItems = [
    { path: '/dashboard', icon: Home, label: 'Overview', exact: true },
    { path: '/dashboard/upload', icon: Upload, label: 'Upload' },
    { path: '/dashboard/query', icon: Search, label: 'Query' },
    { path: '/dashboard/chat', icon: MessageCircle, label: 'Chat' },
    { path: '/dashboard/summarize', icon: BookOpen, label: 'Summarize' },
    { path: '/dashboard/datasets', icon: Database, label: 'Datasets' },
  ];

  const bottomNavItems = [
    { path: '/dashboard/settings', icon: Settings, label: 'Settings' },
    { path: '/dashboard/help', icon: HelpCircle, label: 'Help' },
  ];

  return (
    <div className={`dashboard-container min-h-screen ${darkMode ? 'dark' : ''}`}>
      <div className="flex h-screen dashboard-container">
        {/* Sidebar */}
        <AnimatePresence mode="wait">
          {sidebarOpen && (
            <motion.aside
              initial={{ x: -300 }}
              animate={{ x: 0 }}
              exit={{ x: -300 }}
              transition={{ type: 'spring', stiffness: 300, damping: 30 }}
              className="w-64 dashboard-sidebar flex flex-col"
            >
              {/* Logo */}
              <div className="p-6 border-b border-gray-700" onClick={() => navigate('/dashboard')}>
                <div className="flex items-center space-x-3">
                  <motion.div
                    whileHover={{ rotate: 360 }}
                    transition={{ duration: 0.5 }}
                    className="w-10 h-10 bg-gradient-to-br from-primary-500 to-accent-500 rounded-xl flex items-center justify-center"
                  >
                    <Brain className="w-6 h-6 text-white" />
                  </motion.div>
                  <div>
                    <h1 className="text-xl font-display font-bold text-white">VerixAI</h1>
                    <p className="text-xs text-gray-400">Document Intelligence</p>
                  </div>
                </div>
              </div>

              {/* Navigation */}
              <nav className="flex-1 p-4 space-y-1">
                {navItems.map((item) => (
                  <NavLink
                    key={item.path}
                    to={item.path}
                    end={item.exact}
                    className={({ isActive }) =>
                      `flex items-center space-x-3 px-4 py-3 rounded-xl transition-all duration-200 ${isActive
                        ? 'dashboard-sidebar-item active'
                        : 'dashboard-sidebar-item'
                      }`
                    }
                  >
                    {({ isActive }) => (
                      <>
                        <item.icon className={`w-5 h-5 ${isActive ? 'text-primary-500' : ''}`} />
                        <span>{item.label}</span>
                        {item.label === 'Datasets' && datasets.length > 0 && (
                          <Badge variant="primary" size="sm">
                            {datasets.length}
                          </Badge>
                        )}
                      </>
                    )}
                  </NavLink>
                ))}
              </nav>

              {/* Bottom Navigation */}
              <div className="p-4 space-y-1 border-t border-gray-700">
                {bottomNavItems.map((item) => (
                  <NavLink
                    key={item.path}
                    to={item.path}
                    className={({ isActive }) =>
                      `flex items-center space-x-3 px-4 py-3 rounded-xl transition-all duration-200 ${isActive
                        ? 'dashboard-sidebar-item active'
                        : 'dashboard-sidebar-item'
                      }`
                    }
                  >
                    <item.icon className="w-5 h-5" />
                    <span>{item.label}</span>
                  </NavLink>
                ))}
              </div>

              {/* User Profile */}
              <div className="p-4 border-t border-gray-700">
                <div className="flex items-center space-x-3 p-3 rounded-xl hover:bg-gray-700 cursor-pointer transition-colors">
                  <div className="w-10 h-10 bg-gradient-to-br from-primary-400 to-accent-400 rounded-full flex items-center justify-center">
                    <User className="w-5 h-5 text-white" />
                  </div>
                  <div className="flex-1">
                    <p className="text-sm font-medium text-white">User</p>
                    <p className="text-xs text-gray-400">Local Instance</p>
                  </div>
                  <LogOut className="w-4 h-4 text-gray-400" />
                </div>
              </div>
            </motion.aside>
          )}
        </AnimatePresence>

        {/* Main Content */}
        <div className="flex-1 flex flex-col overflow-hidden">
          {/* Header */}
          <header className="dashboard-nav px-6 py-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-4">
                <button
                  onClick={() => setSidebarOpen(!sidebarOpen)}
                  className="p-2 rounded-lg hover:bg-gray-700 transition-colors text-white"
                >
                  {sidebarOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
                </button>
                <h2 className="text-xl font-semibold text-neutral-900 dark:text-neutral-100">
                  Document Analysis Dashboard
                </h2>
              </div>

              <div className="flex items-center space-x-3">
                <button
                  onClick={() => setDarkMode(!darkMode)}
                  className="p-2 rounded-lg hover:bg-gray-700 transition-colors text-white"
                >
                  {darkMode ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
                </button>

                <button className="relative p-2 rounded-lg hover:bg-gray-700 transition-colors">
                  <Bell className="w-5 h-5 text-gray-300 hover:text-white" />
                  <span className="absolute top-1 right-1 w-2 h-2 bg-error-500 rounded-full"></span>
                </button>
              </div>
            </div>
          </header>

          {/* Main Content Area */}
          <main className="flex-1 overflow-auto bg-gray-900 p-6">
            <AnimatePresence mode="wait">
              {notification.open && (
                <motion.div
                  initial={{ opacity: 0, y: -20, x: 20 }}
                  animate={{ opacity: 1, y: 0, x: 0 }}
                  exit={{ opacity: 0, y: -20, x: 20 }}
                  className="fixed top-20 right-6 z-50"
                >
                  <div className={`
                    px-6 py-4 rounded-xl shadow-2xl flex items-center space-x-3
                    ${notification.type === 'success' ? 'bg-success-500 text-white' : ''}
                    ${notification.type === 'error' ? 'bg-error-500 text-white' : ''}
                    ${notification.type === 'warning' ? 'bg-warning-500 text-white' : ''}
                    ${notification.type === 'info' ? 'bg-primary-500 text-white' : ''}
                  `}>
                    <span className="font-medium">{notification.message}</span>
                    <button
                      onClick={() => setNotification(prev => ({ ...prev, open: false }))}
                      className="ml-4 hover:opacity-80"
                    >
                      <X className="w-4 h-4" />
                    </button>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            <Routes>
              <Route path="/" element={<DashboardOverview datasets={datasets} />} />
              <Route
                path="/upload"
                element={
                  <UploadSection
                    onUploadComplete={handleUploadComplete}
                    datasets={datasets}
                  />
                }
              />
              <Route
                path="/query"
                element={
                  <QuerySection
                    datasets={datasets}
                    onQueryComplete={handleQuery}
                    previousResult={queryResult}
                  />
                }
              />
              <Route
                path="/chat"
                element={<ChatSection datasets={datasets} />}
              />
              <Route
                path="/summarize"
                element={<SummarizationSection datasets={datasets} />}
              />
              <Route
                path="/datasets"
                element={
                  <DatasetsSection
                    datasets={datasets}
                    onDatasetDeleted={handleDatasetDeleted}
                    onRefresh={loadDatasets}
                  />
                }
              />
              <Route path="/settings" element={<SettingsPage />} />
              <Route path="/help" element={<HelpPage />} />
            </Routes>
          </main>
        </div>
      </div>
    </div>
  );
};

// Dashboard Overview Component
const DashboardOverview: React.FC<{ datasets: Dataset[] }> = ({ datasets }) => {
  const navigate = useNavigate();
  const totalDocuments = datasets.reduce((acc, dataset) => acc + dataset.document_count, 0);
  const totalSize = datasets.reduce((acc, dataset) => acc + dataset.size_bytes, 0);

  const formatBytes = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
  };

  const stats = [
    { label: 'Total Datasets', value: datasets.length, icon: Database, color: 'primary' },
    { label: 'Total Documents', value: totalDocuments, icon: Upload, color: 'accent' },
    { label: 'Storage Used', value: formatBytes(totalSize), icon: Database, color: 'success' },
    { label: 'Queries Today', value: '0', icon: Search, color: 'warning' },
  ];

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="space-y-6"
    >
      <div>
        <h1 className="text-3xl font-display font-bold text-white mb-2">
          Welcome back!
        </h1>
        <p className="text-gray-300">
          Here's an overview of your document analysis workspace
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {stats.map((stat, index) => (
          <motion.div
            key={stat.label}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
            className="dashboard-card rounded-2xl p-6 hover:shadow-xl transition-all duration-300"
          >
            <div className="flex items-center justify-between mb-4">
              <div className={`w-12 h-12 bg-${stat.color}-100 dark:bg-${stat.color}-900/30 rounded-xl flex items-center justify-center`}>
                <stat.icon className={`w-6 h-6 text-${stat.color}-600 dark:text-${stat.color}-400`} />
              </div>
              <Badge variant={stat.color as any} size="sm">
                {index === 0 ? '+2' : index === 1 ? '+5' : index === 2 ? '+1.2MB' : 'New'}
              </Badge>
            </div>
            <h3 className="text-2xl font-bold text-white mb-1">
              {stat.value}
            </h3>
            <p className="text-sm text-gray-400">{stat.label}</p>
          </motion.div>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          className="dashboard-card rounded-2xl p-6"
        >
          <h2 className="text-xl font-semibold text-white mb-4">Quick Actions</h2>
          <div className="space-y-3">
            <Button
              variant="primary"
              fullWidth
              leftIcon={<Upload className="w-4 h-4" />}
              onClick={() => navigate('/dashboard/upload')}
            >
              Upload New Documents
            </Button>
            <Button
              variant="outline"
              fullWidth
              leftIcon={<Search className="w-4 h-4" />}
              onClick={() => navigate('/dashboard/query')}
            >
              Query Documents
            </Button>
            <Button
              variant="ghost"
              fullWidth
              leftIcon={<Database className="w-4 h-4" />}
              onClick={() => navigate('/dashboard/datasets')}
            >
              Manage Datasets
            </Button>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          className="dashboard-card rounded-2xl p-6"
        >
          <h2 className="text-xl font-semibold text-white mb-4">Recent Datasets</h2>
          {datasets.length > 0 ? (
            <div className="space-y-3">
              {datasets.slice(0, 5).map((dataset) => (
                <div
                  key={dataset.name}
                  className="flex items-center justify-between p-3 rounded-xl hover:bg-gray-700 transition-colors"
                >
                  <div className="flex items-center space-x-3">
                    <div className="w-10 h-10 bg-primary-100 dark:bg-primary-900/30 rounded-lg flex items-center justify-center">
                      <Database className="w-5 h-5 text-primary-600 dark:text-primary-400" />
                    </div>
                    <div>
                      <p className="font-medium text-white">
                        {dataset.name}
                      </p>
                      <p className="text-sm text-gray-400">
                        {dataset.document_count} documents
                      </p>
                    </div>
                  </div>
                  <Badge variant="default" size="sm">
                    {formatBytes(dataset.size_bytes)}
                  </Badge>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-8 text-neutral-500">
              <Database className="w-12 h-12 mx-auto mb-3 text-neutral-300" />
              <p>No datasets yet</p>
              <p className="text-sm mt-1">Upload documents to get started</p>
            </div>
          )}
        </motion.div>
      </div>
    </motion.div>
  );
};


export default Dashboard;