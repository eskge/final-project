
export type Page = 'Home' | 'Pipeline' | 'Prediction' | 'Reports';

export interface PipelineMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  roc_auc: number;
}

export type ConfusionMatrixData = [[number, number], [number, number]];

export interface PipelineResult {
  metrics: PipelineMetrics;
  confusionMatrix: ConfusionMatrixData;
}

export interface Prediction {
  customerID: string;
  Predicted_Churn: 'Yes' | 'No';
  Churn_Probability: string;
}
