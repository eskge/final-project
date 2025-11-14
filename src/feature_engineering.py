import pandas as pd
import numpy as np

def create_features(df):
    """
    Creates new features from existing ones.
    """
    print("Starting feature engineering...")

    # Feature 1: Monthly average spent (if TotalCharges and tenure are not 0)
    # Avoid division by zero: if tenure is 0, TotalCharges is usually 0 or imputed, so set ratio to 0
    df['AvgMonthlyCharges'] = np.where(df['tenure'] == 0, 
                                       0, 
                                       df['TotalCharges'] / df['tenure'])
    
    # Feature 2: Is the customer a Senior Citizen and has Partner?
    df['SeniorPartner'] = df['SeniorCitizen'] * df['Partner']

    # Feature 3: Has multiple services (count of specific services)
    # Assuming 'No'/'Yes' or 0/1 for service columns
    service_cols = [
        'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    # Ensure these are numeric before summing, as they might be boolean (True/False) after cleaning
    # For one-hot encoded columns this will be different, so let's use the pre-encoded form
    
    # For this project, let's simplify and assume these are already 0/1 after data_cleaning
    # If using one-hot encoded, sum the _Yes versions
    # For now, let's use the raw form and assume they become 0/1 after clean_data
    # This part might need adjustment based on final cleaned_data.csv columns
    
    # Let's count existing services where 'Yes' or 1 is present in relevant columns.
    # The clean_data.py has converted Partner, Dependents, PhoneService, PaperlessBilling to 0/1.
    # The others are one-hot encoded later, so we need to handle this carefully.
    
    # Let's count internet services and phone services
    df['NumInternetServices'] = (df['OnlineSecurity_Yes'] + df['OnlineBackup_Yes'] + 
                                  df['DeviceProtection_Yes'] + df['TechSupport_Yes'] + 
                                  df['StreamingTV_Yes'] + df['StreamingMovies_Yes'])
    
    df['NumPhoneServices'] = df['PhoneService'] + df['MultipleLines_Yes'] # Assuming PhoneService is 0/1 and MultipleLines_Yes is 0/1
    
    print("Feature engineering complete. Added AvgMonthlyCharges, SeniorPartner, NumInternetServices, NumPhoneServices.")
    return df

if __name__ == "__main__":
    PROCESSED_DATA_PATH = 'data/processed/cleaned_data.csv'
    FE_DATA_PATH = 'data/processed/featured_data.csv' # New path for data with engineered features

    # Load cleaned data
    df_cleaned = pd.read_csv(PROCESSED_DATA_PATH)

    # Create new features
    df_featured = create_features(df_cleaned.copy())

    # Save data with new features
    df_featured.to_csv(FE_DATA_PATH, index=False)
    print(f"Data with engineered features saved to {FE_DATA_PATH}. Shape: {df_featured.shape}")
    print("\n--- Data with Features (First 5 rows) ---")
    print(df_featured.head())