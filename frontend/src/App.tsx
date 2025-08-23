import React, { useState, useEffect } from 'react';
import {
  ThemeProvider,
  createTheme,
  CssBaseline,
  Container,
  AppBar,
  Toolbar,
  Typography,
  Box,
  Tab,
  Tabs,
  Paper,
  Alert,
  Snackbar
} from '@mui/material';
import { DocumentScanner, Search, Storage } from '@mui/icons-material';
import UploadSection from './components/UploadSection';
import QuerySection from './components/QuerySection';
import DatasetsSection from './components/DatasetsSection';
import { Dataset, QueryResult } from './types';
import api from './services/api';

const theme = createTheme({
  palette: {
    primary: {
      main: '#2196f3',
    },
    secondary: {
      main: '#f50057',
    },
    background: {
      default: '#f5f5f5',
    },
  },
  typography: {
    h4: {
      fontWeight: 600,
    },
  },
});

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`tabpanel-${index}`}
      aria-labelledby={`tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

function App() {
  const [tabValue, setTabValue] = useState(0);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [queryResult, setQueryResult] = useState<QueryResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [notification, setNotification] = useState<{
    open: boolean;
    message: string;
    severity: 'success' | 'error' | 'info' | 'warning';
  }>({
    open: false,
    message: '',
    severity: 'info',
  });

  useEffect(() => {
    loadDatasets();
  }, []);

  const loadDatasets = async () => {
    try {
      const response = await api.getDatasets();
      setDatasets(response.data);
    } catch (error) {
      console.error('Failed to load datasets:', error);
      showNotification('Failed to load datasets', 'error');
    }
  };

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  const showNotification = (message: string, severity: 'success' | 'error' | 'info' | 'warning' = 'info') => {
    setNotification({
      open: true,
      message,
      severity,
    });
  };

  const handleCloseNotification = () => {
    setNotification({ ...notification, open: false });
  };

  const handleUploadComplete = () => {
    loadDatasets();
    showNotification('Documents uploaded successfully!', 'success');
  };

  const handleQuery = async (result: QueryResult) => {
    setQueryResult(result);
    setTabValue(1);
  };

  const handleDatasetDeleted = () => {
    loadDatasets();
    showNotification('Dataset deleted successfully', 'success');
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ flexGrow: 1 }}>
        <AppBar position="static" elevation={1}>
          <Toolbar>
            <DocumentScanner sx={{ mr: 2 }} />
            <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
              VerixAI - Document Analysis Assistant
            </Typography>
          </Toolbar>
        </AppBar>

        <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
          <Paper elevation={2}>
            <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
              <Tabs value={tabValue} onChange={handleTabChange} aria-label="main tabs">
                <Tab icon={<DocumentScanner />} label="Upload Documents" />
                <Tab icon={<Search />} label="Query Documents" />
                <Tab icon={<Storage />} label="Manage Datasets" />
              </Tabs>
            </Box>

            <TabPanel value={tabValue} index={0}>
              <UploadSection
                onUploadComplete={handleUploadComplete}
                datasets={datasets}
              />
            </TabPanel>

            <TabPanel value={tabValue} index={1}>
              <QuerySection
                datasets={datasets}
                onQueryComplete={handleQuery}
                previousResult={queryResult}
              />
            </TabPanel>

            <TabPanel value={tabValue} index={2}>
              <DatasetsSection
                datasets={datasets}
                onDatasetDeleted={handleDatasetDeleted}
                onRefresh={loadDatasets}
              />
            </TabPanel>
          </Paper>
        </Container>

        <Snackbar
          open={notification.open}
          autoHideDuration={6000}
          onClose={handleCloseNotification}
          anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
        >
          <Alert onClose={handleCloseNotification} severity={notification.severity} sx={{ width: '100%' }}>
            {notification.message}
          </Alert>
        </Snackbar>
      </Box>
    </ThemeProvider>
  );
}

export default App;
