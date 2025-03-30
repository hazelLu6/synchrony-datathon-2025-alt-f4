import pandas as pd
import numpy as np
import xgboost_models
import random

# Load the master dataset
print("Loading master dataset...")
try:
    df = pd.read_csv("exploratory_data_analysis/master_user_dataset.csv")
    print(f"Successfully loaded master dataset with {len(df)} rows and {len(df.columns)} columns")
except FileNotFoundError:
    print("Master dataset file not found. Looking for alternative locations...")
    try:
        df = pd.read_csv("data/master_user_dataset.csv")
        print(f"Successfully loaded master dataset from data/ with {len(df)} rows and {len(df.columns)} columns")
    except FileNotFoundError:
        print("Failed to find master dataset. Please ensure the file exists.")
        df = pd.DataFrame()  # Empty dataframe to prevent further errors

# If we have data, try to run the models
if not df.empty:
    # Check if quarterly fields exist, if not, run master.py first
    required_columns = ['spend_2024_q1', 'spend_2024_q2', 'spend_2024_q3']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Warning: Missing quarterly data columns: {missing_columns}")
        print("To run Q3 validation, first update the master dataset with quarterly data.")
        print("You can run the updated master.py script to regenerate the dataset.")
    else:
        print("\n=== Testing Q3 Validation Model ===")
        try:
            val_model, df_with_validation = xgboost_models.model1_q3_validation(df)
            print("Q3 Validation model executed successfully!")
        except Exception as e:
            print(f"Error executing Q3 Validation model: {e}")
    
    # Add necessary columns for testing - only if they don't exist
    # For Model 2 (Account Segmentation)
    if 'segment_label' not in df.columns:
        print("Adding segment_label column for testing...")
        # Create a random distribution for testing purposes
        df['segment_label'] = np.random.choice([0, 1, 2, 3], size=len(df), p=[0.4, 0.3, 0.2, 0.1])
    
    # For Model 3 (Risk Flagging)
    if 'risk_flag' not in df.columns:
        print("Adding risk_flag column for testing...")
        df['risk_flag'] = np.random.choice([0, 1], size=len(df), p=[0.8, 0.2])
    
    # For Model 4 (CLI Recommendation)
    if 'CLI_target_amount' not in df.columns:
        print("Adding CLI_target_amount column for testing...")
        df['CLI_target_amount'] = np.random.uniform(0, 5000, size=len(df))
    
    print("\n=== Testing Model 1: Q4 Spend Prediction ===")
    try:
        model1, df_with_pred = xgboost_models.model1_q4_spend_prediction(df)
        print("Model 1 executed successfully!")
    except Exception as e:
        print(f"Error executing Model 1: {e}")
    
    print("\n=== Testing Model 2: Account Segmentation ===")
    try:
        model2, df_with_segments = xgboost_models.model2_account_segmentation(df)
        print("Model 2 executed successfully!")
    except Exception as e:
        print(f"Error executing Model 2: {e}")
    
    print("\n=== Testing Model 3: Risk Flagging ===")
    try:
        model3, df_with_risk = xgboost_models.model3_risk_flagging(df)
        print("Model 3 executed successfully!")
    except Exception as e:
        print(f"Error executing Model 3: {e}")
    
    print("\n=== Testing Model 4: CLI Recommendation ===")
    try:
        model4, df_with_cli = xgboost_models.model4_cli_recommendation(df)
        print("Model 4 executed successfully!")
    except Exception as e:
        print(f"Error executing Model 4: {e}")
else:
    print("No data loaded. Cannot test models.")

print("\nTest script completed.") 