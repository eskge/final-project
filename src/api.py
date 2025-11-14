import asyncio
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
import pandas as pd
import os

# Import your existing pipeline functions
from src.pipeline import run_pipeline_streaming
from src.prediction_service import load_model_and_scaler, make_predictions

# Create FastAPI app
app = FastAPI()

# CORS Middleware
origins = [
    "http://localhost:3000", # Default React port
    "http://localhost:5173", # Default Vite port
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Churn Prediction API"}

@app.get("/pipeline")
async def pipeline_endpoint():
    """
    Runs the full ML pipeline and streams progress updates.
    """
    return EventSourceResponse(run_pipeline_streaming())

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    """
    Accepts a CSV file, processes it, and returns churn predictions.
    """
    # This is a simplified version. In a real app, you'd save the file
    # and then process it. For now, we'll read it into a pandas DataFrame.
    df = pd.read_csv(file.file)

    # --- This part needs to be adapted from your prediction_service ---
    # 1. Load model and scaler
    model, scaler = load_model_and_scaler()
    
    # 2. Get the training features to align columns
    # This is a bit of a shortcut. Ideally, the feature engineering pipeline
    # should be more robust.
    train_full_featured_df = pd.read_csv('data/processed/featured_data.csv')
    training_features_df = train_full_featured_df.drop('Churn', axis=1)

    # 3. Make predictions
    # Note: The `make_predictions` function from your service might need adjustments
    # to work directly with a DataFrame instead of file paths.
    # For now, let's assume it works.
    predictions, probabilities = make_predictions(
        data_to_predict=df,
        model=model,
        scaler=scaler,
        training_features_df=training_features_df
    )

    # 4. Format results
    results = []
    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        results.append({
            "customerID": df['customerID'][i] if 'customerID' in df.columns else f"NewCust_{i}",
            "Predicted_Churn": "Yes" if pred == 1 else "No",
            "Churn_Probability": f"{prob:.2f}"
        })
    
    return results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
