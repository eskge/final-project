
import React from 'react';
import { PipelineResult } from '../../types';
import Button from '../ui/Button';
import Card from '../ui/Card';
import MetricCard from '../ui/MetricCard';
import ConfusionMatrix from '../ui/ConfusionMatrix';
import Spinner from '../ui/Spinner';

interface PipelinePageProps {
  pipelineState: 'idle' | 'running' | 'success' | 'error';
  pipelineLog: string[];
  pipelineResult: PipelineResult | null;
  onRunPipeline: () => void;
}

const PipelinePage: React.FC<PipelinePageProps> = ({
  pipelineState,
  pipelineLog,
  pipelineResult,
  onRunPipeline,
}) => {
  return (
    <div className="animate-fade-in space-y-6">
      <h1 className="text-4xl font-bold text-white">Full Pipeline Execution</h1>
      <p className="text-lg text-slate-400">
        This section simulates running the entire ML pipeline, from loading the raw dataset to model training and evaluation.
      </p>

      <Card>
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between">
          <div className="mb-4 md:mb-0">
            <h2 className="text-xl font-semibold text-white">Start the Process</h2>
            <p className="text-slate-400 mt-1">Click the button to begin the simulation.</p>
          </div>
          <Button onClick={onRunPipeline} disabled={pipelineState === 'running'}>
            {pipelineState === 'running' ? (
              <>
                <Spinner />
                Running...
              </>
            ) : (
              'Run Full Pipeline'
            )}
          </Button>
        </div>
      </Card>

      {pipelineState !== 'idle' && (
        <Card title="Execution Log">
          <div className="bg-slate-900/50 rounded-lg p-4 h-64 overflow-y-auto font-mono text-sm text-slate-300">
            {pipelineLog.map((log, index) => (
              <p key={index} className="animate-fade-in-up" style={{ animationDelay: `${index * 50}ms` }}>
                <span className="text-sky-400 mr-2">&gt;</span>{log}
              </p>
            ))}
             {pipelineState === 'running' && <div className="w-2 h-4 bg-sky-400 animate-pulse ml-1 inline-block" />}
          </div>
        </Card>
      )}

      {pipelineState === 'success' && pipelineResult && (
        <div className="animate-fade-in space-y-6">
          <Card title="Model Performance Metrics (on Test Set)">
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
              <MetricCard label="Accuracy" value={pipelineResult.metrics.accuracy.toFixed(4)} />
              <MetricCard label="Precision" value={pipelineResult.metrics.precision.toFixed(4)} />
              <MetricCard label="Recall" value={pipelineResult.metrics.recall.toFixed(4)} />
              <MetricCard label="F1-Score" value={pipelineResult.metrics.f1.toFixed(4)} />
              <MetricCard label="ROC-AUC" value={pipelineResult.metrics.roc_auc.toFixed(4)} />
            </div>
          </Card>
          <Card title="Confusion Matrix">
            <ConfusionMatrix data={pipelineResult.confusionMatrix} />
          </Card>
        </div>
      )}

      {pipelineState === 'error' && (
         <Card>
            <div className="text-red-400">An error occurred during the pipeline simulation. Please check the console for details.</div>
         </Card>
      )}
    </div>
  );
};

export default PipelinePage;
