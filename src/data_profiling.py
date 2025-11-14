import pandas as pd
from ydata_profiling import ProfileReport # Using ydata-profiling as pandas-profiling's new name
import os

def generate_profiling_report(df, output_path='reports/profiling_report.html'):
    """
    Generates a detailed data profiling report using ydata-profiling.
    """
    print("Generating comprehensive profiling report...")
    profile = ProfileReport(df, title="Telco Customer Churn Data Profile", explorative=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    profile.to_file(output_path)
    print(f"Profiling report saved to {output_path}")

def run_custom_data_quality_checks(df):
    """
    Runs custom data quality checks and logs issues.
    """
    print("Running custom data quality checks...")
    issues = []

    # Check for 'customerID' uniqueness
    if not df['customerID'].is_unique:
        issues.append("ERROR: 'customerID' column contains duplicate values.")
    
    # Check for 'TotalCharges' data type consistency and potential errors
    # 'TotalCharges' is often loaded as object due to empty strings or spaces
    initial_total_charges_type = df['TotalCharges'].dtype
    try:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        if initial_total_charges_type == 'object':
             issues.append("WARNING: 'TotalCharges' column was initially object type, converted to numeric (coercing errors).")
    except ValueError:
        issues.append("ERROR: 'TotalCharges' column cannot be fully converted to numeric.")

    # Check for negative 'tenure' or 'MonthlyCharges' (shouldn't happen in this dataset, but good practice)
    if (df['tenure'] < 0).any():
        issues.append("ERROR: 'tenure' column contains negative values.")
    if (df['MonthlyCharges'] < 0).any():
        issues.append("ERROR: 'MonthlyCharges' column contains negative values.")

    # Check for 'Churn' values
    if not df['Churn'].isin(['Yes', 'No']).all():
        issues.append("ERROR: 'Churn' column contains values other than 'Yes' or 'No'.")
    
    # Check for missing values in critical columns
    critical_cols = ['customerID', 'MonthlyCharges', 'TotalCharges', 'tenure', 'Churn']
    for col in critical_cols:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            issues.append(f"WARNING: Column '{col}' has {missing_count} missing values.")

    if not issues:
        print("No critical data quality issues found by custom checks.")
    else:
        print("\n--- Custom Data Quality Issues Found ---")
        for issue in issues:
            print(issue)
        print("------------------------------------------")
            
    return issues

if __name__ == "__main__":
    RAW_DATA_PATH = 'data/raw/Telco-Customer-Churn.csv'
    
    # Load data
    df = pd.read_csv(RAW_DATA_PATH)

    # Generate full profiling report
    generate_profiling_report(df)

    # Run custom checks
    run_custom_data_quality_checks(df.copy()) # Pass a copy to avoid modifying original df