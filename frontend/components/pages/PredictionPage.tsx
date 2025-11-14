
import React from 'react';
import { Prediction } from '../../types';
import Card from '../ui/Card';
import FileUpload from '../ui/FileUpload';
import ResultTable from '../ui/ResultTable';
import Spinner from '../ui/Spinner';

interface PredictionPageProps {
  predictionState: 'idle' | 'running' | 'success' | 'error';
  predictionResult: Prediction[] | null;
  uploadedFileName: string | null;
  onPredict: (file: File) => void;
}

const PredictionPage: React.FC<PredictionPageProps> = ({
  predictionState,
  predictionResult,
  uploadedFileName,
  onPredict,
}) => {
  return (
    <div className="animate-fade-in space-y-6">
      <h1 className="text-4xl font-bold text-white">Predict on New Data</h1>
      <p className="text-lg text-slate-400">
        Upload a CSV file containing new customer data to get churn predictions from the trained model.
      </p>

      <Card>
        <FileUpload onFileSelect={onPredict} disabled={predictionState === 'running'} />
      </Card>

      {predictionState === 'running' && (
        <Card>
          <div className="flex items-center justify-center p-8">
            <Spinner />
            <span className="ml-4 text-lg text-slate-300">Generating predictions for {uploadedFileName}...</span>
          </div>
        </Card>
      )}
      
      {predictionState === 'success' && predictionResult && (
         <Card title={`Prediction Results for ${uploadedFileName}`}>
            <ResultTable predictions={predictionResult} />
         </Card>
      )}

      {predictionState === 'error' && (
         <Card>
            <div className="text-red-400">An error occurred while generating predictions. Please try again.</div>
         </Card>
      )}

    </div>
  );
};

export default PredictionPage;
