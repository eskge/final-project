
import React from 'react';
import Card from '../ui/Card';

const HomePage: React.FC = () => {
  return (
    <div className="animate-fade-in">
      <h1 className="text-4xl font-bold text-white mb-4">Welcome to the Telco Customer Churn Dashboard</h1>
      <p className="text-lg text-slate-400 mb-8">
        An interactive frontend for an end-to-end machine learning solution to predict customer churn.
      </p>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card title="Project Overview">
          <p className="text-slate-300">
            This application demonstrates a full ML pipeline, from data ingestion to prediction. It's designed to mirror the functionality of a Python-based data science project, providing a user-friendly interface to interact with the model.
          </p>
        </Card>
        <Card title="How to Use">
          <p className="text-slate-300">
            Use the sidebar to navigate:
          </p>
          <ul className="list-disc list-inside mt-2 text-slate-300 space-y-1">
            <li><strong>Pipeline:</strong> Simulate the full model training process.</li>
            <li><strong>Prediction:</strong> Upload new customer data to get churn predictions.</li>
            <li><strong>Reports:</strong> View mock-ups of generated reports.</li>
          </ul>
        </Card>
      </div>

      <div className="mt-8">
        <Card title="Project Phases">
            <ol className="list-decimal list-inside space-y-2 text-slate-300">
                <li><strong>Data Acquisition & Ingestion:</strong> Loading raw customer data.</li>
                <li><strong>Data Profiling & Quality Assessment:</strong> Understanding data characteristics and identifying issues.</li>
                <li><strong>Data Cleaning & Preprocessing:</strong> Handling missing values, data type conversions, encoding.</li>
                <li><strong>Feature Engineering:</strong> Creating new, impactful features.</li>
                <li><strong>Model Training & Evaluation:</strong> Building and assessing a Machine Learning model.</li>
                <li><strong>Prediction & Reporting:</strong> Generating churn predictions and insights.</li>
                <li><strong>Deployment:</strong> A modern React web app to demonstrate the pipeline.</li>
            </ol>
        </Card>
      </div>
    </div>
  );
};

export default HomePage;
