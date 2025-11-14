import asyncio
import json
import pandas as pd
import os

# Import all necessary functions from your pipeline scripts
from .data_ingestion import load_raw_data
from .data_profiling import generate_profiling_report, run_custom_data_quality_checks
from .data_cleaning import clean_data, preprocess_for_ml
from .feature_engineering import create_features
from .model_training import train_model

async def run_pipeline_streaming():
    """
    Runs the full ML pipeline and yields progress updates for SSE.
    """
    # Global paths
    RAW_DATA_DIR = 'data/raw'
    PROCESSED_DATA_DIR = 'data/processed'
    REPORTS_DIR = 'reports'
    MODELS_DIR = 'models'
    
    async def send_event(event_name, data):
        return f"event: {event_name}\ndata: {json.dumps(data)}\n\n"

    try:
        yield await send_event('progress', {'message': 'Starting pipeline execution...'})
        await asyncio.sleep(1)

        # 1. Data Ingestion
        yield await send_event('progress', {'message': '1. Data Ingestion: Loading raw data...'})
        raw_data_path = os.path.join(RAW_DATA_DIR, 'Telco-Customer-Churn.csv')
        df_raw = load_raw_data(raw_data_path)
        yield await send_event('progress', {'message': 'Raw data loaded successfully.'})
        await asyncio.sleep(1)

        # 2. Data Profiling
        yield await send_event('progress', {'message': '2. Data Profiling & Quality Assessment...'})
        profiling_report_path = os.path.join(REPORTS_DIR, 'profiling_report.html')
        generate_profiling_report(df_raw.copy(), profiling_report_path)
        yield await send_event('progress', {'message': 'Profiling report generated.'})
        await asyncio.sleep(1)

        # 3. Data Cleaning
        yield await send_event('progress', {'message': '3. Data Cleaning & Preprocessing...'})
        df_cleaned = clean_data(df_raw.copy())
        df_preprocessed = preprocess_for_ml(df_cleaned.copy())
        cleaned_data_output_path = os.path.join(PROCESSED_DATA_DIR, 'cleaned_data.csv')
        df_preprocessed.to_csv(cleaned_data_output_path, index=False)
        yield await send_event('progress', {'message': 'Data cleaned and preprocessed.'})
        await asyncio.sleep(1)

        # 4. Feature Engineering
        yield await send_event('progress', {'message': '4. Feature Engineering...'})
        df_featured = create_features(df_preprocessed.copy())
        featured_data_output_path = os.path.join(PROCESSED_DATA_DIR, 'featured_data.csv')
        df_featured.to_csv(featured_data_output_path, index=False)
        yield await send_event('progress', {'message': 'New features created.'})
        await asyncio.sleep(1)

        # 5. Model Training
        yield await send_event('progress', {'message': '5. Model Training & Evaluation...'})
        yield await send_event('progress', {'message': 'This will take a while as it tests multiple models...'})
        
        # This is a synchronous function, so we run it in a thread pool
        # to avoid blocking the asyncio event loop.
        loop = asyncio.get_event_loop()
        model, scaler, _, accuracy, precision, recall, f1, roc_auc, conf_matrix = await loop.run_in_executor(
            None, train_model, df_featured.copy()
        )
        
        yield await send_event('progress', {'message': 'Model training and evaluation complete.'})
        await asyncio.sleep(1)

        # 6. Send final results
        result_data = {
            'metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'roc_auc': roc_auc,
            },
            'confusionMatrix': conf_matrix.tolist(), # convert numpy array to list
        }
        yield await send_event('result', result_data)

    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        yield await send_event('error', {'message': error_message})

    finally:
        # Signal the end of the stream
        yield await send_event('close', {'message': 'Pipeline execution finished.'})