import { PipelineResult, Prediction } from '../types';

const API_BASE_URL = 'http://localhost:8000';

export const runFullPipeline = (
  onProgress: (message: string) => void,
  onResult: (result: PipelineResult) => void,
  onError: (error: string) => void
) => {
  const eventSource = new EventSource(`${API_BASE_URL}/pipeline`);

  eventSource.addEventListener('progress', (event) => {
    const data = JSON.parse(event.data);
    onProgress(data.message);
  });

  eventSource.addEventListener('result', (event) => {
    const data = JSON.parse(event.data);
    onResult(data);
  });

  eventSource.addEventListener('error', (event) => {
    const data = JSON.parse(event.data);
    onError(data.message);
    eventSource.close();
  });
  
  eventSource.addEventListener('close', (event) => {
    console.log('Closing connection:', event.data);
    eventSource.close();
  });

  return eventSource;
};

export const predictNewData = async (file: File): Promise<Prediction[]> => {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_BASE_URL}/predict`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || 'Prediction failed');
  }

  return response.json();
};
