import pandas as pd
import numpy as np
import os

def clean_data(df):
    """
    Performs data cleaning and initial preprocessing steps.
    """
    print("Starting data cleaning and preprocessing...")

    # 1. Drop customerID column as it's not a feature for ML
    df = df.drop('customerID', axis=1)

    # 2. Handle 'TotalCharges' - Convert to numeric, coerce errors to NaN
    # It often comes as 'object' type with some empty strings which are not numbers
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # 3. Handle missing values
    # For 'TotalCharges', fill missing values with the median.
    # This happens when tenure is 0 (new customers)
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    print(f"Handled missing values in 'TotalCharges'. Now {df['TotalCharges'].isnull().sum()} missing.")

    # 4. Handle 'No internet service' and 'No phone service'
    # Replace 'No internet service' with 'No' for columns like 'OnlineSecurity', 'DeviceProtection', etc.
    # Replace 'No phone service' with 'No' for 'MultipleLines'
    internet_service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                             'TechSupport', 'StreamingTV', 'StreamingMovies']
    for col in internet_service_cols:
        df[col] = df[col].replace('No internet service', 'No')
    
    df['MultipleLines'] = df['MultipleLines'].replace('No phone service', 'No')

    # 5. Convert 'Yes'/'No' in 'Churn' to 1/0
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    print("Converted 'Churn' column to numerical (1/0).")

    # 6. Convert 'No' in 'Partner' and 'Dependents' to No (standardizing)
    # The dataset already uses 'Yes'/'No', so this might be redundant but good for robustness
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0})

    # 7. Convert Gender to numerical (0/1)
    df['gender'] = df['gender'].map({'Female': 0, 'Male': 1})

    print("Data cleaning and preprocessing complete.")
    return df

def preprocess_for_ml(df):
    """
    Further preprocesses data for ML, including one-hot encoding.
    """
    print("Applying one-hot encoding for categorical features...")
    
    # Identify categorical columns that are not binary and not 'Churn'
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    
    # Exclude 'Churn' if it's still object type (should be handled by clean_data)
    if 'Churn' in categorical_cols:
        categorical_cols.remove('Churn')

    # Apply One-Hot Encoding
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True) # drop_first to avoid multicollinearity
    print("One-hot encoding complete.")

    return df_encoded

if __name__ == "__main__":
    RAW_DATA_PATH = 'data/raw/Telco-Customer-Churn.csv'
    PROCESSED_DATA_PATH = 'data/processed/cleaned_data.csv'

    # Load raw data
    raw_df = pd.read_csv(RAW_DATA_PATH)

    # Clean the data
    cleaned_df = clean_data(raw_df.copy()) # Pass a copy to avoid modifying raw_df directly

    # Preprocess for ML (one-hot encoding)
    final_df = preprocess_for_ml(cleaned_df.copy())

    # Save processed data
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    final_df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Cleaned and preprocessed data saved to {PROCESSED_DATA_PATH}. Shape: {final_df.shape}")

    print("\n--- Cleaned Data Info (First 5 rows) ---")
    print(final_df.head())
    print("\n--- Cleaned Data Types ---")
    print(final_df.info())