import pandas as pd
import numpy as np
import os
import sys

def check_missing_columns():
    """Check which columns from XGBoost models are missing in the master_user_dataset.csv"""
    try:
        # Try to read just the header row to get column names
        csv_path = "master_user_dataset.csv"
        if not os.path.exists(csv_path):
            print(f"Error: {csv_path} not found. Please run master.py first.")
            return
            
        df_header = pd.read_csv(csv_path, nrows=0)
        columns = df_header.columns.tolist()
        
        # Define key features used in XGBoost models
        key_features = [
            'is_high_income',
            'delinquency_12mo',
            'delinquency_24mo',
            'risk_flag',
            'risk_probability',
            'segment_label',
            'credit_score',
            'utilization_pct',
            'total_spend_2024',
            'total_spend_2025YTD',
            'credit_line',
            'current_balance',
            'avg_monthly_spend_2024'
        ]
        
        # Find which columns are missing
        missing_columns = [col for col in key_features if col not in columns]
        
        # Print the results
        print(f"CSV file contains {len(columns)} columns")
        
        if missing_columns:
            print(f"Missing columns ({len(missing_columns)}):")
            for col in missing_columns:
                print(f"  - {col}")
        else:
            print("All required XGBoost columns are present!")
            
        print("\nAll columns in CSV:")
        print(sorted(columns))
        
        # If possible, print first few rows of data to verify content
        try:
            df_sample = pd.read_csv(csv_path, nrows=5)
            if not missing_columns:
                print("\nSample data for key XGBoost columns:")
                print(df_sample[key_features].head())
        except Exception as e:
            print(f"Error reading data sample: {e}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_missing_columns() 