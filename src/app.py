import streamlit as st
import pandas as pd
import os
import sys
import webbrowser
from sklearn.model_selection import train_test_split # Added for splitting data for report

# Import all necessary functions from your pipeline scripts
from data_ingestion import load_raw_data
from data_profiling import generate_profiling_report, run_custom_data_quality_checks
from data_cleaning import clean_data, preprocess_for_ml
from feature_engineering import create_features
from model_training import train_model # <--- This will now return more
from prediction_service import load_model_and_scaler, make_predictions, save_predictions_report

# Global paths
RAW_DATA_DIR = 'data/raw'
PROCESSED_DATA_DIR = 'data/processed'
REPORTS_DIR = 'reports'
MODELS_DIR = 'models'

# Ensure directories exist
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

st.set_page_config(layout="wide")
st.title("Telco Customer Churn Prediction & Data Quality Monitor")

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Run Pipeline (Train & Predict)", "Upload & Predict New Data", "View Reports"])

if page == "Home":
    st.header("Welcome!")
    st.write("""
    This application provides an end-to-end solution for predicting customer churn
    for a telecommunications company, with a strong emphasis on data quality monitoring.

    You can:
    - Run the full ML pipeline from data ingestion to model training and prediction.
    - Upload new customer data and get instant churn predictions.
    - View comprehensive data profiling and prediction reports.
    """)
    # Placeholder image - replace with a relevant image if you have one
    st.image("https://via.placeholder.com/800x400.png?text=Telco+Churn+Dashboard+Concept", caption="Conceptual Dashboard View")
    st.markdown("---")
    st.subheader("Project Phases:")
    st.markdown("""
    1. **Data Acquisition & Ingestion**: Loading raw customer data.
    2. **Data Profiling & Quality Assessment**: Understanding data characteristics and identifying issues.
    3. **Data Cleaning & Preprocessing**: Handling missing values, data type conversions, encoding.
    4. **Feature Engineering**: Creating new, impactful features.
    5. **Model Training & Evaluation**: Building and assessing a Machine Learning model.
    6. **Prediction & Reporting**: Generating churn predictions and insights.
    7. **Deployment**: A simplified Streamlit app to demonstrate the pipeline.
    """)

elif page == "Run Pipeline (Train & Predict)":
    st.header("Full Pipeline Execution")
    st.write("This section runs the entire ML pipeline, from loading the raw dataset, through profiling, cleaning, feature engineering, to model training and saving.")

    st.warning("Ensure 'Telco-Customer-Churn.csv' is in the `data/raw/` directory before running!")

    st.warning("""
    **Note:** This process now runs a comprehensive model selection and hyperparameter tuning process
    for multiple models (RandomForest, XGBoost, LightGBM). 
    **This will take a significant amount of time to complete.**
    """)

    if st.button("Run Full Pipeline"):
        st.info("Starting pipeline execution... This may take a few moments.")

        # 1. Data Ingestion
        st.subheader("1. Data Ingestion")
        raw_data_path = os.path.join(RAW_DATA_DIR, 'Telco-Customer-Churn.csv')
        df_raw = load_raw_data(raw_data_path)
        st.success("Raw data loaded.")
        st.write(df_raw.head())

        # 2. Data Profiling & Quality Assessment
        st.subheader("2. Data Profiling & Quality Assessment")
        profiling_report_path = os.path.join(REPORTS_DIR, 'profiling_report.html')
        generate_profiling_report(df_raw.copy(), profiling_report_path)
        custom_issues = run_custom_data_quality_checks(df_raw.copy())
        if custom_issues:
            for issue in custom_issues:
                st.error(issue)
        else:
            st.success("Custom data quality checks passed with no critical issues.")
        st.markdown(f"[View Full Profiling Report]({profiling_report_path}) (opens locally)")

        # 3. Data Cleaning
        st.subheader("3. Data Cleaning & Preprocessing")
        df_cleaned = clean_data(df_raw.copy())
        df_preprocessed = preprocess_for_ml(df_cleaned.copy())
        cleaned_data_output_path = os.path.join(PROCESSED_DATA_DIR, 'cleaned_data.csv')
        df_preprocessed.to_csv(cleaned_data_output_path, index=False)
        st.success(f"Data cleaned and preprocessed. Saved to `{cleaned_data_output_path}`")
        st.write(df_preprocessed.head())

        # 4. Feature Engineering
        st.subheader("4. Feature Engineering")
        df_featured = create_features(df_preprocessed.copy())
        featured_data_output_path = os.path.join(PROCESSED_DATA_DIR, 'featured_data.csv')
        df_featured.to_csv(featured_data_output_path, index=False)
        st.success(f"Features engineered. Saved to `{featured_data_output_path}`")
        st.write(df_featured.head())

        # 5. Model Training & Evaluation (Updated Section)
        st.subheader("5. Model Training & Evaluation")

        # Unpack all 9 values returned by train_model
        model, scaler, feature_cols_trained_on, accuracy, precision, recall, f1, roc_auc, conf_matrix = train_model(df_featured.copy())

        st.success("Model trained and evaluated. Model and scaler saved.")
        st.write("---")
        st.subheader("Model Performance Metrics (on Test Set):")

        st.write(f"**Accuracy:** `{accuracy:.4f}`")
        st.write(f"**Precision:** `{precision:.4f}`")
        st.write(f"**Recall:** `{recall:.4f}`")
        st.write(f"**F1-Score:** `{f1:.4f}`")
        st.write(f"**ROC-AUC:** `{roc_auc:.4f}`")
        st.write("**Confusion Matrix:**")
        st.dataframe(pd.DataFrame(conf_matrix, index=['Actual No Churn', 'Actual Churn'], columns=['Predicted No Churn', 'Predicted Churn']))

        st.subheader("Pipeline Execution Complete!")
        st.info("You can now navigate to 'Upload & Predict New Data' to use the trained model and view reports.")


elif page == "Upload & Predict New Data":
    st.header("Upload New Customer Data & Get Predictions")
    st.write("Upload a CSV file containing new customer data (similar format to the original Telco dataset) to get churn predictions.")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            new_customer_df_raw = pd.read_csv(uploaded_file)
            st.write("Raw uploaded data preview:")
            st.write(new_customer_df_raw.head())

            st.subheader("Processing New Data...")

            # 1. Load Model and Scaler
            try:
                # FIX: load_model_and_scaler only returns 2 values (model and scaler), 
                # so we only unpack 2 variables to avoid the ValueError.
                model_loaded, scaler_loaded = load_model_and_scaler()

                # Load the full featured data used for training to get the *exact* column structure
                # This ensures alignment for prediction
                train_full_featured_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'featured_data.csv'))
                # IMPORTANT: Only drop 'Churn' if it exists in train_full_featured_df
                if 'Churn' in train_full_featured_df.columns:
                    training_features_df_for_prediction = train_full_featured_df.drop('Churn', axis=1)
                else:
                    training_features_df_for_prediction = train_full_featured_df.copy() # If no Churn, use all features

            except FileNotFoundError:
                st.error("Model or scaler not found. Please run the 'Full Pipeline' first to train the model.")
                st.stop() # Stop execution if model/scaler not found

            # 2. Data Cleaning & Preprocessing for new data
            st.info("Running data cleaning and preprocessing for uploaded data...")
            new_customer_df_cleaned = clean_data(new_customer_df_raw.copy())
            new_customer_df_preprocessed = preprocess_for_ml(new_customer_df_cleaned.copy())

            # 3. Feature Engineering for new data
            st.info("Running feature engineering for uploaded data...")
            new_customer_df_featured = create_features(new_customer_df_preprocessed.copy())

            # 4. Make Predictions
            st.info("Generating churn predictions...")
            predictions, probabilities = make_predictions(
                data_to_predict=new_customer_df_featured.copy(),
                model=model_loaded, # Use the loaded model
                scaler=scaler_loaded, # Use the loaded scaler
                training_features_df=training_features_df_for_prediction # Pass the DataFrame here
            )

            # 5. Display Results
            st.subheader("Prediction Results:")
            results_df = pd.DataFrame({
                # Ensure customerID exists, otherwise generate dummy IDs
                'customerID': new_customer_df_raw['customerID'].reset_index(drop=True) if 'customerID' in new_customer_df_raw.columns else [f"NewCust_{i}" for i in range(len(new_customer_df_raw))],
                'Predicted_Churn': ['Yes' if p == 1 else 'No' for p in predictions],
                'Churn_Probability': [f"{prob:.2f}" for prob in probabilities]
            })
            st.write(results_df)

            csv_output = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Churn Predictions CSV",
                data=csv_output,
                file_name="new_churn_predictions.csv",
                mime="text/csv",
            )
            st.success("Predictions generated successfully!")

            # Also save these predictions to the default report path
            save_predictions_report(
                original_df_with_ids=new_customer_df_raw.reset_index(drop=True),
                predictions=predictions,
                probabilities=probabilities,
                output_path=os.path.join(REPORTS_DIR, 'churn_predictions.csv')
            )
            st.info(f"Predictions for uploaded data also saved to `{os.path.join(REPORTS_DIR, 'churn_predictions.csv')}` for reporting.")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.exception(e) # Display full traceback for debugging


elif page == "View Reports":
    st.header("Generated Reports")
    st.write("Access the data profiling and prediction reports generated by the pipeline.")

    st.subheader("Data Profiling Report")
    profiling_report_path = os.path.join(REPORTS_DIR, 'profiling_report.html')
    if os.path.exists(profiling_report_path):
        st.success("Data Profiling Report found!")

        # Get the absolute path for webbrowser.open
        abs_profiling_report_path = os.path.abspath(profiling_report_path)
        
        # Pre-calculate the path replacement to avoid the SyntaxError in the f-string
        file_uri_path = abs_profiling_report_path.replace('\\', '/')

        # Create a Streamlit button
        if st.button("Open Data Profiling Report (HTML)", type="primary"):
            try:
                # webbrowser.open_new_tab(f"file:///{file_uri_path}") 
                st.success("Report should be opening in a new browser tab.")
            except Exception as e:
                st.error(f"Failed to open report: {e}. Please try manually accessing it.")
                # Fallback instructions if automated opening fails
                st.markdown(f"""
                    **To view the detailed Data Profiling Report manually:**
                    1.  **Copy and Paste the File URI:**
                        * Copy the following path:
                            `file:///{file_uri_path}` 
                        * Open a **new tab** in your web browser.
                        * Paste the copied path into the address bar and press `Enter`.

                    2.  **Open Directly from File Explorer:**
                        * Navigate to this location on your computer:
                            `{abs_profiling_report_path}`
                        * Double-click the `profiling_report.html` file to open it in your default browser.
                    """)

    else:
        st.warning("Data Profiling Report not found. Please run the 'Full Pipeline' first.")


    st.subheader("Churn Predictions Report")
    churn_predictions_path = os.path.join(REPORTS_DIR, 'churn_predictions.csv')
    if os.path.exists(churn_predictions_path):
        st.write("Preview of the last generated churn predictions (from pipeline run or last upload):")

        # Check if the file is truly empty or just has a header
        if os.path.getsize(churn_predictions_path) > 0:
            try:
                predictions_df = pd.read_csv(churn_predictions_path)
                # Check if DataFrame is empty after reading (e.g., only header)
                if not predictions_df.empty:
                    st.write(predictions_df.head(10))

                    csv_output = predictions_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Last Churn Predictions CSV",
                        data=csv_output,
                        file_name="last_churn_predictions.csv",
                        mime="text/csv",
                    )
                else:
                    st.warning("Churn Predictions Report found, but it is empty. Please run the 'Full Pipeline' or 'Upload & Predict New Data' to generate content.")
            except pd.errors.EmptyDataError:
                st.warning("Churn Predictions Report file exists but contains no data to parse. Please run the 'Full Pipeline' or 'Upload & Predict New Data' to generate content.")
            except Exception as e:
                st.error(f"An error occurred while reading the Churn Predictions Report: {e}")
        else:
            st.warning("Churn Predictions Report file exists but is empty. Please run the 'Full Pipeline' or 'Upload & Predict New Data' to generate content.")
    else:
        st.warning("Churn Predictions Report not found. Please run the 'Full Pipeline' or 'Upload & Predict New Data' first.")

st.sidebar.markdown("---")
st.sidebar.info("Developed as an end-to-end data science project example.")
