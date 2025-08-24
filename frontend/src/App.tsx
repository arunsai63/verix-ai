import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { AnimatePresence } from 'framer-motion';
import LandingPage from './pages/LandingPage';
import Dashboard from './pages/Dashboard';
import './index.css';

function App() {
  return (
    <Router basename="/verix-ai">
      <AnimatePresence mode="wait">
        <Routes>
          <Route path="/" element={<LandingPage />} />
          <Route path="/dashboard/*" element={<Dashboard />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </AnimatePresence>
    </Router>
  );
}

export default App;