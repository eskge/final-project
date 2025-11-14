
import React, { useState, useCallback, useEffect } from 'react';
import { Page, PipelineResult, Prediction } from './types';
import { runFullPipeline, predictNewData } from './services/apiService';
import Layout from './components/Layout';
import HomePage from './components/pages/HomePage';
import PipelinePage from './components/pages/PipelinePage';
import PredictionPage from './components/pages/PredictionPage';
import ReportsPage from './components/pages/ReportsPage';

const App: React.FC = () => {
  const [page, setPage] = useState<Page>('Home');
  const [pipelineState, setPipelineState] = useState<'idle' | 'running' | 'success' | 'error'>('idle');
  const [pipelineLog, setPipelineLog] = useState<string[]>([]);
  const [pipelineResult, setPipelineResult] = useState<PipelineResult | null>(null);
  
  const [predictionState, setPredictionState] = useState<'idle' | 'running' | 'success' | 'error'>('idle');
  const [predictionResult, setPredictionResult] = useState<Prediction[] | null>(null);
  const [uploadedFileName, setUploadedFileName] = useState<string | null>(null);

  const handleRunPipeline = useCallback(() => {
    setPipelineState('running');
    setPipelineResult(null);
    setPipelineLog([]);

    const onProgress = (message: string) => {
      setPipelineLog(prev => [...prev, message]);
    };

    const onResult = (result: PipelineResult) => {
      setPipelineResult(result);
      setPipelineState('success');
    };

    const onError = (error: string) => {
      console.error(error);
      setPipelineState('error');
      onProgress(`Pipeline execution failed: ${error}`);
    };

    runFullPipeline(onProgress, onResult, onError);
  }, []);

  const handlePredict = useCallback(async (file: File) => {
    setPredictionState('running');
    setPredictionResult(null);
    setUploadedFileName(file.name);
    try {
      const result = await predictNewData(file);
      setPredictionResult(result);
      setPredictionState('success');
    } catch (error) {
      console.error(error);
      setPredictionState('error');
    }
  }, []);

  const renderPage = () => {
    switch (page) {
      case 'Pipeline':
        return (
          <PipelinePage
            pipelineState={pipelineState}
            pipelineLog={pipelineLog}
            pipelineResult={pipelineResult}
            onRunPipeline={handleRunPipeline}
          />
        );
      case 'Prediction':
        return (
          <PredictionPage
            predictionState={predictionState}
            predictionResult={predictionResult}
            uploadedFileName={uploadedFileName}
            onPredict={handlePredict}
          />
        );
      case 'Reports':
        return <ReportsPage />;
      case 'Home':
      default:
        return <HomePage />;
    }
  };

  return (
    <Layout page={page} setPage={setPage}>
      {renderPage()}
    </Layout>
  );
};

export default App;
