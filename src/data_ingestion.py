import pandas as pd
import os

def load_raw_data(file_path):
    """
    Loads raw data from a CSV file.
    """
    if not os.path.exists(file_path):
        print(f"Error: Raw data file not found at {file_path}")
        print("Please download 'Telco-Customer-Churn.csv' from Kaggle and place it in 'data/raw/'")
        exit() # Or handle more gracefully
    
    print(f"Loading raw data from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Raw data loaded successfully. Shape: {df.shape}")
    return df

def save_intermediate_data(df, file_path, index=False):
    """
    Saves a DataFrame to a CSV file.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path, index=index)
    print(f"Intermediate data saved to {file_path}")

if __name__ == "__main__":
    RAW_DATA_PATH = 'data/raw/Telco-Customer-Churn.csv'
    
    # Simulate data ingestion - just loading for now
    customer_df = load_raw_data(RAW_DATA_PATH)
    
    # Display basic info
    print("\n--- Raw Data Info ---")
    customer_df.info()
    print("\n--- First 5 rows ---")
    print(customer_df.head())