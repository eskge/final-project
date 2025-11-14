import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split # Added for the if __name__ block
from data_cleaning import clean_data, preprocess_for_ml # Added for the if __name__ block
from feature_engineering import create_features # Added for the if __name__ block


def load_model_and_scaler(model_path='models/churn_predictor_model.pkl', scaler_path='models/scaler.pkl'):
    """
    Loads the trained ML model and scaler.
    """
    print("Loading trained model and scaler...")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please train the model first.")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found at {scaler_path}. Please train the model first.")
        
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print("Model and scaler loaded successfully.")
    return model, scaler

# CHANGE: Add `training_features_df` parameter
def make_predictions(data_to_predict, model, scaler, training_features_df, target_col='Churn'):
    """
    Makes churn predictions on new data.
    `training_features_df`: A DataFrame representing the features (without target) 
                            that the model was trained on, used for column alignment and dtypes.
    """
    print("Making predictions on new data...")

    # Drop the target column if present in the data to predict
    if target_col in data_to_predict.columns:
        data_to_predict = data_to_predict.drop(target_col, axis=1)

    # --- Crucial for alignment: Reindex `data_to_predict` to match training columns ---
    # Identify missing columns in data_to_predict compared to training data
    missing_cols_in_new_data = set(training_features_df.columns) - set(data_to_predict.columns)
    for col in missing_cols_in_new_data:
        data_to_predict[col] = 0  # Fill missing one-hot encoded cols with 0

    # Identify extra columns in data_to_predict that weren't in training data
    extra_cols_in_new_data = set(data_to_predict.columns) - set(training_features_df.columns)
    data_to_predict = data_to_predict.drop(columns=list(extra_cols_in_new_data))
    
    # Ensure column order matches the training data precisely
    data_to_predict = data_to_predict[training_features_df.columns]
    # ----------------------------------------------------------------------

    # Identify numerical columns from the training features for scaling
    numerical_cols = training_features_df.select_dtypes(include=['float64', 'int64']).columns
    
    # Scale numerical features using the *fitted* scaler
    data_to_predict[numerical_cols] = scaler.transform(data_to_predict[numerical_cols])
    
    predictions = model.predict(data_to_predict)
    probabilities = model.predict_proba(data_to_predict)[:, 1] # Probability of churn

    print("Predictions made.")
    return predictions, probabilities

# CHANGE: Renamed original_df to original_df_with_ids for clarity and ensured it's used for customerID
def save_predictions_report(original_df_with_ids, predictions, probabilities, output_path='reports/churn_predictions.csv'):
    """
    Saves the predictions along with customer IDs.
    `original_df_with_ids`: The raw DataFrame slice corresponding to the data that was predicted upon,
                            expected to contain 'customerID'.
    """
    print(f"Saving predictions to {output_path}...")
    
    if 'customerID' not in original_df_with_ids.columns:
        # If customerID is not present, create a dummy one or raise an error
        print("Warning: 'customerID' not found in the original DataFrame for prediction report. Generating dummy IDs.")
        customer_ids = [f"sim_cust_{i}" for i in range(len(original_df_with_ids))]
    else:
        customer_ids = original_df_with_ids['customerID'].reset_index(drop=True) # Ensure index alignment

    report_df = pd.DataFrame({
        'customerID': customer_ids,
        'Predicted_Churn': predictions,
        'Churn_Probability': probabilities
    })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    report_df.to_csv(output_path, index=False)
    print(f"Predictions report saved to {output_path}")

if __name__ == "__main__":
    # This block demonstrates how the prediction service would be used in a full workflow.
    # It simulates loading raw test data, preprocessing it, and then predicting.

    RAW_DATA_PATH = 'data/raw/Telco-Customer-Churn.csv'
    FE_DATA_PATH = 'data/processed/featured_data.csv' # Path to the full featured data (including target)

    # 1. Load the full featured data that was used for model training
    # This is needed to get the *exact* column structure (features and their dtypes)
    # that the model expects for prediction.
    full_training_data = pd.read_csv(FE_DATA_PATH)
    training_features_df_for_prediction = full_training_data.drop('Churn', axis=1) 

    # 2. Simulate new raw data for prediction (e.g., a test set from the original raw data)
    raw_df = pd.read_csv(RAW_DATA_PATH)
    # Split raw data to simulate a 'new' batch, keeping original customer IDs for reporting
    _, test_raw_df, _, _ = train_test_split(raw_df, raw_df['Churn'], test_size=0.2, random_state=42, stratify=raw_df['Churn'])

    # 3. Preprocess this simulated new raw data through the exact same pipeline steps
    print("\n--- Preprocessing simulated new data for prediction ---")
    cleaned_test_df = clean_data(test_raw_df.copy()) # Pass a copy
    preprocessed_test_df = preprocess_for_ml(cleaned_test_df.copy()) # Pass a copy
    X_predict_preprocessed_featured = create_features(preprocessed_test_df.copy()) # Pass a copy
    print("--- Preprocessing complete ---")

    # 4. Load the trained model and scaler
    model, scaler = load_model_and_scaler()
    
    # 5. Make predictions
    predictions, probabilities = make_predictions(
        data_to_predict=X_predict_preprocessed_featured.copy(), # The fully preprocessed & featured data for prediction
        model=model, 
        scaler=scaler, 
        training_features_df=training_features_df_for_prediction # Pass the full DataFrame of features used for training
    )

    # 6. Save the prediction report
    save_predictions_report(test_raw_df.reset_index(drop=True), predictions, probabilities) # Pass the raw test_df with customerIDs
    print("Simulated prediction run complete. Check reports/churn_predictions.csv")
