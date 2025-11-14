import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
# import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib
import os
import numpy as np

def train_model(df, target_col='Churn'):
    """
    Loops through a list of models, tunes their hyperparameters, evaluates them,
    and returns the best model and its metrics.
    """
    # Separate features (X) and target (y)
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Data split: Train={X_train.shape}, Test={X_test.shape}")

    # Scale numerical features
    numerical_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
    scaler = StandardScaler()
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
    print("Numerical features scaled.")

    # --- Model and Hyperparameter Definitions ---

    # Calculate scale_pos_weight for XGBoost
    scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]

    models_to_try = [
        {
            'name': 'RandomForest',
            'estimator': RandomForestClassifier(random_state=42, class_weight='balanced'),
            'param_grid': {
                'n_estimators': [100, 200, 300, 400, 500],
                'max_features': ['sqrt', 'log2', 0.6, 0.8],
                'max_depth': [8, 15, 25, 35, None],
                'min_samples_split': [2, 5, 10, 15],
                'min_samples_leaf': [1, 2, 4, 6],
                'bootstrap': [True, False]
            }
        },
        {
            'name': 'XGBoost',
            'estimator': XGBClassifier(random_state=42, scale_pos_weight=scale_pos_weight, use_label_encoder=False, eval_metric='logloss'),
            'param_grid': {
                'n_estimators': [100, 200, 300, 400, 500],
                'learning_rate': [0.005, 0.01, 0.05, 0.1, 0.15],
                'max_depth': [3, 5, 7, 9, 11],
                'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
                'gamma': [0, 0.1, 0.2, 0.3, 0.4]
            }
        },
        # {
        #     'name': 'LightGBM',
        #     'estimator': lgb.LGBMClassifier(random_state=42, is_unbalance=True),
        #     'param_grid': {
        #         'n_estimators': [100, 200, 300],
        #         'learning_rate': [0.01, 0.05, 0.1],
        #         'num_leaves': [20, 31, 40],
        #         'max_depth': [-1, 10, 15],
        #         'subsample': [0.7, 0.8, 0.9],
        #         'colsample_bytree': [0.7, 0.8, 0.9],
        #     }
        # }
        {
            'name': 'LogisticRegression',
            'estimator': LogisticRegression(random_state=42, solver='liblinear', class_weight='balanced'),
            'param_grid': {
                'C': np.logspace(-3, 3, 7), # Inverse of regularization strength
                'penalty': ['l1', 'l2']
            }
        },
        {
            'name': 'GradientBoosting',
            'estimator': GradientBoostingClassifier(random_state=42),
            'param_grid': {
                'n_estimators': [100, 200, 300, 400, 500],
                'learning_rate': [0.005, 0.01, 0.05, 0.1, 0.15],
                'max_depth': [3, 5, 7, 9, 11],
                'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                'max_features': ['sqrt', 'log2', None, 0.6, 0.8]
            }
        },
        {
            'name': 'AdaBoost',
            'estimator': AdaBoostClassifier(random_state=42),
            'param_grid': {
                'n_estimators': [50, 100, 150, 200, 250],
                'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
            }
        },
        {
            'name': 'HistGradientBoosting',
            'estimator': HistGradientBoostingClassifier(random_state=42),
            'param_grid': {
                'max_iter': [100, 200, 300, 400, 500],
                'learning_rate': [0.005, 0.01, 0.05, 0.1, 0.15, 0.2],
                'max_depth': [None, 3, 5, 7, 9, 11],
                'min_samples_leaf': [10, 20, 30, 40, 50, 60]
            }
        }
    ]

    results = []
    best_model_for_app = None
    
    for model_info in models_to_try:
        print(f"\n--- Tuning {model_info['name']} ---")
        
        random_search = RandomizedSearchCV(
            estimator=model_info['estimator'],
            param_distributions=model_info['param_grid'],
            n_iter=100,  # Increased for more extensive search
            cv=3,
            verbose=1,
            random_state=42,
            n_jobs=-1
        )
        
        random_search.fit(X_train, y_train)
        
        best_model = random_search.best_estimator_
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        conf_matrix = confusion_matrix(y_test, y_pred)

        results.append({
            'Model': model_info['name'],
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'ROC-AUC': roc_auc,
            'Confusion Matrix': conf_matrix,
            'Best Params': random_search.best_params_,
            'Model Object': best_model
        })

        print(f"--- Finished tuning {model_info['name']} ---")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  ROC-AUC: {roc_auc:.4f}")
        print(f"  Best Params: {random_search.best_params_}")

    # --- Display Results ---
    print("\n\n--- Model Comparison ---")
    results_df = pd.DataFrame(results)
    print(results_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']].to_string())
    
    # Find and save the best overall model
    best_model_stats = results_df.loc[results_df['Accuracy'].idxmax()]
    print(f"\n--- Best Performing Model: {best_model_stats['Model']} ---")
    
    final_model = best_model_stats['Model Object']
    
    # Save the final model and scaler
    os.makedirs('models', exist_ok=True)
    joblib.dump(final_model, 'models/churn_predictor_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    print("Best model and scaler saved to 'models/' directory.")

    # Return all the values the app expects
    return (
        final_model,
        scaler,
        X.columns.tolist(),
        best_model_stats['Accuracy'],
        best_model_stats['Precision'],
        best_model_stats['Recall'],
        best_model_stats['F1-Score'],
        best_model_stats['ROC-AUC'],
        best_model_stats['Confusion Matrix']
    )


if __name__ == "__main__":
    FE_DATA_PATH = 'data/processed/featured_data.csv'

    # Load data with engineered features
    df_featured = pd.read_csv(FE_DATA_PATH)

    # Run the model selection process
    train_model(df_featured.copy())
