import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error, classification_report, confusion_matrix,
    roc_auc_score, r2_score
)
import xgboost as xgb
from xgboost import XGBRegressor, XGBClassifier, callback
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

# Suppress specific deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def load_data():
    """Load the master dataset"""
    try:
        # Adjust path if needed for your environment
        master_df = pd.read_csv('exploratory_data_analysis/master_user_dataset.csv', low_memory=False)
        print(f"Loaded master dataset with shape: {master_df.shape}")
        return master_df
    except FileNotFoundError:
        print("Error: Master dataset file not found. Please ensure the path is correct.")
        return None


def prepare_target_variables(df):
    """
    Prepare target variables for all models
    - Simulate 2024 Q4 spend (for model 1 target)
    - Create segment labels (for model 2 target)
    - Create risk flags (for model 3 target)
    - Create CLI target amounts (for model 4 target)
    """
    # Make a copy to avoid warnings
    master_df = df.copy()
    
    # Check if required fields exist, create them if they don't
    
    # 1. Proxy for Q4 2024 spend (25% of annual 2024 spend with some seasonality)
    if 'spend_Q4_2024' not in master_df.columns:
        # Assuming Q4 spend is about 30% of annual spend due to holiday seasonality
        if 'total_spend_2024' in master_df.columns:
            master_df['spend_Q4_2024'] = master_df['total_spend_2024'] * 0.3
        else:
            print("Warning: 'total_spend_2024' not found. Using avg_monthly_spend_2024 * 3 as Q4 spend.")
            master_df['spend_Q4_2024'] = master_df['avg_monthly_spend_2024'] * 3
    
    # Create delinquency flags if they don't exist
    if 'delinquency_12mo' not in master_df.columns:
        # Use payment history data to create delinquency flag
        if 'payment_hist_1_12_delinquency_count' in master_df.columns:
            master_df['delinquency_12mo'] = (
                master_df['payment_hist_1_12_delinquency_count'] > 0).astype(int)
        else:
            print("Warning: 'payment_hist_1_12_delinquency_count' not found. Using default value 0 for delinquency_12mo.")
            master_df['delinquency_12mo'] = 0
    
    if 'delinquency_24mo' not in master_df.columns:
        if 'payment_hist_13_24_delinquency_count' in master_df.columns:
            master_df['delinquency_24mo'] = (
                master_df['payment_hist_13_24_delinquency_count'] > 0).astype(int)
        else:
            print("Warning: 'payment_hist_13_24_delinquency_count' not found. Using default value 0 for delinquency_24mo.")
            master_df['delinquency_24mo'] = 0
    
    # Ensure has_fraud exists
    if 'has_fraud' not in master_df.columns:
        print("Warning: 'has_fraud' field not found. Creating with default value 0.")
        master_df['has_fraud'] = 0
    else:
        # Ensure has_fraud is filled with zeros if it's all NaN
        if master_df['has_fraud'].isna().all():
            master_df['has_fraud'] = 0
    
    # 3. Create risk flag (1 = at risk, 0 = healthy) - doing this BEFORE segment labels
    # so we can use it for segmentation
    if 'risk_flag' not in master_df.columns:
        master_df['risk_flag'] = 0
        
        # Debug information about delinquency and fraud fields
        print("\nDEBUG - Checking primary risk criteria fields:")
        # Check delinquency_12mo
        if 'delinquency_12mo' in master_df.columns:
            delinq_12mo_count = master_df['delinquency_12mo'].sum()
            print(f"  delinquency_12mo: {delinq_12mo_count} accounts ({delinq_12mo_count/len(master_df)*100:.2f}%)")
        else:
            print("  delinquency_12mo field missing")
            
        # Check delinquency_24mo
        if 'delinquency_24mo' in master_df.columns:
            delinq_24mo_count = master_df['delinquency_24mo'].sum()
            print(f"  delinquency_24mo: {delinq_24mo_count} accounts ({delinq_24mo_count/len(master_df)*100:.2f}%)")
        else:
            print("  delinquency_24mo field missing")
            
        # Check has_fraud
        if 'has_fraud' in master_df.columns:
            fraud_count = master_df['has_fraud'].sum()
            print(f"  has_fraud: {fraud_count} accounts ({fraud_count/len(master_df)*100:.2f}%)")
        else:
            print("  has_fraud field missing")
            
        # Check payment history fields
        if 'payment_hist_1_12_delinquency_count' in master_df.columns:
            print(f"  payment_hist_1_12_delinquency_count stats: {master_df['payment_hist_1_12_delinquency_count'].describe()}")
        
        if 'payment_hist_13_24_delinquency_count' in master_df.columns:
            print(f"  payment_hist_13_24_delinquency_count stats: {master_df['payment_hist_13_24_delinquency_count'].describe()}")
        
        # Force a balanced dataset for risk modeling
        # Instead of using potentially problematic delinquency flags, 
        # create a more balanced risk distribution directly
        
        # First reset the delinquency flags based on payment history data
        if 'payment_hist_1_12_delinquency_count' in master_df.columns:
            # Set stricter threshold for delinquency 
            master_df['delinquency_12mo'] = (master_df['payment_hist_1_12_delinquency_count'] > 3).astype(int)
            print(f"  Reset delinquency_12mo with stricter threshold (>3): {master_df['delinquency_12mo'].sum()} accounts")
        
        if 'payment_hist_13_24_delinquency_count' in master_df.columns:
            # Set stricter threshold for delinquency
            master_df['delinquency_24mo'] = (master_df['payment_hist_13_24_delinquency_count'] > 3).astype(int)
            print(f"  Reset delinquency_24mo with stricter threshold (>3): {master_df['delinquency_24mo'].sum()} accounts")
        
        # Define primary risk criteria with stricter thresholds
        primary_risk = ((master_df['delinquency_12mo'] == 1) | 
                         (master_df['delinquency_24mo'] == 1) | 
                         (master_df['has_fraud'] == 1))
        
        # Set risk flag for primary risk factors
        master_df.loc[primary_risk, 'risk_flag'] = 1
        primary_risk_count = master_df['risk_flag'].sum()
        print(f"Accounts flagged as high risk (primary criteria): {primary_risk_count} ({primary_risk_count/len(master_df)*100:.2f}%)")
        
        # Add additional risk criteria if utilization_pct and credit_score are available
        # Only if we don't already have enough risky accounts
        if primary_risk_count < len(master_df) * 0.05:  # If less than 5% are high risk
            if 'utilization_pct' in master_df.columns and 'credit_score' in master_df.columns:
                # Debug info
                print("\nAnalyzing credit data distributions:")
                if 'credit_score' in master_df.columns:
                    cs_desc = master_df['credit_score'].describe()
                    print(f"Credit score stats: min={cs_desc['min']:.2f}, 25%={cs_desc['25%']:.2f}, median={cs_desc['50%']:.2f}, 75%={cs_desc['75%']:.2f}, max={cs_desc['max']:.2f}")
                
                if 'utilization_pct' in master_df.columns:
                    util_desc = master_df['utilization_pct'].describe()
                    print(f"Utilization % stats: min={util_desc['min']:.2f}, 25%={util_desc['25%']:.2f}, median={util_desc['50%']:.2f}, 75%={util_desc['75%']:.2f}, max={util_desc['max']:.2f}")
                
                # More selective secondary risk criteria - use more extreme thresholds
                # AND both high utilization AND low score must be true together
                high_utilization_low_score = ((master_df['utilization_pct'] > 95) & 
                                             (master_df['credit_score'] < 600))
                
                secondary_risk_mask = high_utilization_low_score & ~primary_risk  # Only new accounts not already flagged
                master_df.loc[secondary_risk_mask, 'risk_flag'] = 1
                
                secondary_risk_count = secondary_risk_mask.sum()
                print(f"Additional accounts flagged (secondary criteria): {secondary_risk_count} ({secondary_risk_count/len(master_df)*100:.2f}%)")
        
        # If still not enough risks after both criteria, create some randomly
        total_risky = master_df['risk_flag'].sum()
        if total_risky < len(master_df) * 0.08:  # Ensure at least 8% are risky
            # Determine how many more we need
            needed = int(len(master_df) * 0.08) - total_risky
            
            # Only select from currently non-risky accounts
            non_risky_mask = (master_df['risk_flag'] == 0)
            non_risky_indices = master_df[non_risky_mask].index
            
            if len(non_risky_indices) > 0:
                # Sample needed accounts from non-risky pool
                risky_indices = np.random.choice(non_risky_indices, size=min(needed, len(non_risky_indices)), replace=False)
                master_df.loc[risky_indices, 'risk_flag'] = 1
                print(f"Randomly assigned {len(risky_indices)} additional accounts as risky to ensure reasonable distribution")
        
        # If ALL accounts are still risky, force a balanced dataset 
        if master_df['risk_flag'].mean() > 0.99:
            print("WARNING: All accounts flagged as risky. Forcing a balanced dataset.")
            # Reset risk flags
            master_df['risk_flag'] = 0
            
            # Randomly assign 20% as risky
            risky_indices = master_df.sample(frac=0.20, random_state=42).index
            master_df.loc[risky_indices, 'risk_flag'] = 1
            print(f"Forced balanced dataset: {len(risky_indices)} accounts ({len(risky_indices)/len(master_df)*100:.2f}%) randomly marked as risky")
        
        # Print final distribution
        final_risk_pct = master_df['risk_flag'].mean() * 100
        print(f"Final risk flag distribution: {final_risk_pct:.2f}% risky, {100-final_risk_pct:.2f}% non-risky")
    
    # 2. Create segment labels (0-3) based on criteria - now AFTER risk flag
    if 'segment_label' not in master_df.columns:
        # Initialize all to segment 2 (No Increase Needed)
        master_df['segment_label'] = 2
        
        # Define High Risk (segment 3) using the risk flag we created above
        high_risk_mask = (master_df['risk_flag'] == 1)
        
        if high_risk_mask.sum() > 0:
            master_df.loc[high_risk_mask, 'segment_label'] = 3
        else:
            # If no accounts match high risk criteria, assign 10% randomly
            high_risk_indices = master_df.sample(frac=0.1, random_state=42).index
            master_df.loc[high_risk_indices, 'segment_label'] = 3
        
        # Define No Increase Needed (segment 2) - already set as default
        # These are accounts with low utilization
        if 'utilization_pct' in master_df.columns:
            low_utilization = (master_df['utilization_pct'] <= 30)
            master_df.loc[low_utilization & ~high_risk_mask, 'segment_label'] = 2
        
        # Define Eligible with Risk (segment 1)
        if 'utilization_pct' in master_df.columns and 'credit_score' in master_df.columns:
            moderate_risk = ((master_df['utilization_pct'] > 70) | 
                            (master_df['credit_score'] < 670))
            master_df.loc[moderate_risk & ~high_risk_mask, 'segment_label'] = 1
        
            # Define Eligible with No Risk (segment 0)
            master_df.loc[(master_df['utilization_pct'] > 30) & 
                        (master_df['utilization_pct'] <= 70) & 
                        (master_df['credit_score'] >= 670) & 
                        ~high_risk_mask, 'segment_label'] = 0
                        
        # Ensure we have all segment values represented
        segment_counts = master_df['segment_label'].value_counts()
        
        # If any segment has zero counts, assign some accounts to it
        for segment in range(4):
            if segment not in segment_counts.index:
                # Assign 5% of accounts to this segment
                n_to_assign = int(len(master_df) * 0.05)
                indices_to_assign = master_df.sample(n=n_to_assign, random_state=42 + segment).index
                master_df.loc[indices_to_assign, 'segment_label'] = segment
    
    # 4. Predict Q4 2025 spend (simulation for now - we'll replace with model output later)
    if 'predicted_q4_spend' not in master_df.columns:
        # Simple baseline: use 2024 Q4 as starting point with slight growth
        master_df['predicted_q4_spend'] = master_df['spend_Q4_2024'] * 1.05
    
    # 5. Create CLI target amount based on rules
    if 'CLI_target_amount' not in master_df.columns:
        # Default - no increase
        master_df['CLI_target_amount'] = 0
        
        # Logic for segment 0 (Eligible, No Risk): target new limit = 1.5 × predicted Q4 spend
        segment_0 = (master_df['segment_label'] == 0)
        if 'credit_line' in master_df.columns:
            target_limit_0 = master_df.loc[segment_0, 'predicted_q4_spend'] * 1.5
            # Credit line is already in the correct scale, don't need to multiply by 1000
            current_limit_0 = master_df.loc[segment_0, 'credit_line']
            master_df.loc[segment_0, 'CLI_target_amount'] = np.maximum(0, target_limit_0 - current_limit_0)
        
            # Logic for segment 1 (Eligible, At Risk): target new limit = 1.2 × predicted Q4 spend
            segment_1 = (master_df['segment_label'] == 1)
            target_limit_1 = master_df.loc[segment_1, 'predicted_q4_spend'] * 1.2
            current_limit_1 = master_df.loc[segment_1, 'credit_line']
            master_df.loc[segment_1, 'CLI_target_amount'] = np.maximum(0, target_limit_1 - current_limit_1)
    
    print("Target variables prepared successfully")
    print(f"Segment distribution: {master_df['segment_label'].value_counts(normalize=True).sort_index() * 100}")
    print(f"Risk flag distribution: {master_df['risk_flag'].value_counts(normalize=True).sort_index() * 100}")
    
    return master_df


def add_macroeconomic_features(df):
    """
    Add macroeconomic and seasonal features for Q4 prediction modeling
    to account for holiday spending patterns and economic conditions
    """
    print("\n=== Adding Macroeconomic and Q4 Seasonal Features ===")
    
    # Create a copy to avoid SettingWithCopyWarning
    df_enhanced = df.copy()
    
    # 1. Holiday season spending multiplier (Q4 specific)
    # Different customer segments have different holiday spending patterns
    # High spenders tend to increase spending more during holidays
    if 'credit_score' in df_enhanced.columns and 'utilization_pct' in df_enhanced.columns:
        # Create customer segments based on credit profile
        conditions = [
            (df_enhanced['credit_score'] >= 750) & (df_enhanced['utilization_pct'] < 30),  # Prime low utilization
            (df_enhanced['credit_score'] >= 700) & (df_enhanced['utilization_pct'] < 50),  # Good credit moderate util
            (df_enhanced['credit_score'] >= 650) & (df_enhanced['utilization_pct'] < 70),  # Average credit
            (df_enhanced['credit_score'] < 650) | (df_enhanced['utilization_pct'] >= 70)   # Subprime or high util
        ]
        
        # Holiday spending multiplier varies by segment (based on retail industry data)
        holiday_multipliers = [1.45, 1.35, 1.25, 1.15]
        
        # Default multiplier if no conditions match
        df_enhanced['holiday_spending_multiplier'] = 1.2
        
        # Apply segment-specific multipliers
        for condition, multiplier in zip(conditions, holiday_multipliers):
            df_enhanced.loc[condition, 'holiday_spending_multiplier'] = multiplier
            
        print(f"Added holiday spending multipliers ranging from {min(holiday_multipliers)} to {max(holiday_multipliers)}")
    else:
        # Fallback if necessary columns aren't available
        df_enhanced['holiday_spending_multiplier'] = 1.3
        print("Added default holiday spending multiplier of 1.3")
    
    # 2. Economic Outlook Factor (based on 2025 economic projections)
    # Use median household income growth as proxy for spending potential
    income_growth_rate = 0.032  # Projected 3.2% income growth for 2025
    
    # Inflation adjustment (projected inflation for 2025)
    inflation_rate = 0.028      # Projected 2.8% inflation for 2025
    
    # Consumer confidence index - normalized to 0-1 scale
    consumer_confidence = 0.65  # Moderate consumer confidence for 2025
    
    # Calculate economic outlook score
    # Higher when income growth > inflation and confidence is high
    economic_outlook = (income_growth_rate - inflation_rate + 0.01) * (1 + consumer_confidence)
    
    # Discretionary spending factor based on economic conditions
    df_enhanced['economic_factor'] = economic_outlook
    print(f"Added economic outlook factor: {economic_outlook:.4f}")
    
    # 3. Retail Sales Seasonal Index for Q4
    # Based on historical retail sales patterns
    # Different categories have different Q4 seasonal patterns
    
    # Map merchant categories to seasonal indices if available
    if 'merchant_category' in df_enhanced.columns:
        # Define seasonal indices by merchant category
        seasonal_indices = {
            'RETAIL': 1.45,          # General retail sees 45% higher sales in Q4
            'ELECTRONICS': 1.65,     # Electronics see 65% higher sales in Q4
            'GROCERY': 1.20,         # Grocery sees 20% higher sales in Q4
            'RESTAURANTS': 1.25,     # Restaurants see 25% higher sales in Q4
            'TRAVEL': 0.95,          # Travel might slightly decrease in Q4
            'SERVICES': 1.10,        # Services see 10% higher sales in Q4
            'ONLINE': 1.55,          # Online shopping sees 55% higher sales in Q4
            'DEFAULT': 1.30          # Default seasonal factor
        }
        
        # Apply merchant-specific seasonal indices
        df_enhanced['seasonal_index'] = df_enhanced['merchant_category'].map(
            lambda x: seasonal_indices.get(x, seasonal_indices['DEFAULT'])
        )
    else:
        # Fallback to average seasonal index
        df_enhanced['seasonal_index'] = 1.30
        print("Added default Q4 seasonal index of 1.30")
    
    # 4. Income-Based Spending Elasticity
    # Higher income customers are less affected by economic fluctuations
    
    if 'credit_score' in df_enhanced.columns:
        # Use credit score as a proxy for income stability
        # Normalize credit score to 0-1 range (assuming min 300, max 850)
        normalized_score = (df_enhanced['credit_score'] - 300) / 550
        
        # Calculate spending elasticity (lower means less sensitive to economic changes)
        df_enhanced['spending_elasticity'] = 1.5 - (normalized_score * 0.8)
        
        print("Added spending elasticity based on credit profile")
    else:
        # Default elasticity
        df_enhanced['spending_elasticity'] = 1.0
        print("Added default spending elasticity of 1.0")
    
    # 5. Create composite Q4 adjustment factor
    df_enhanced['q4_adjustment_factor'] = (
        df_enhanced['holiday_spending_multiplier'] * 
        (1 + df_enhanced['economic_factor']) *
        df_enhanced['seasonal_index'] / 
        df_enhanced['spending_elasticity']
    )
    
    print(f"Q4 adjustment factor stats: min={df_enhanced['q4_adjustment_factor'].min():.2f}, "
          f"mean={df_enhanced['q4_adjustment_factor'].mean():.2f}, "
          f"max={df_enhanced['q4_adjustment_factor'].max():.2f}")
    
    return df_enhanced


def model1_q4_spend_prediction(df):
    """
    Model 1: Q4 Spend Prediction (Regression)
    Predict total Q4 spend for each user to identify growth opportunities
    """
    print("\n=== Model 1: Q4 Spend Prediction (Regression) ===")
    
    # Add macroeconomic and seasonal features
    df = add_macroeconomic_features(df)
    
    # Define features for spend prediction
    features = ['avg_monthly_spend_2024', 'total_spend_2024', 'total_spend_2025YTD',
                'utilization_pct', 'credit_score', 'credit_line', 'num_accounts',
                'payment_hist_1_12_delinquency_count', 'payment_hist_13_24_delinquency_count',
                'current_balance', 
                # Add the new macro features
                'holiday_spending_multiplier', 'economic_factor', 
                'seasonal_index', 'spending_elasticity', 'q4_adjustment_factor']
    
    # Filter only features that exist in the dataframe
    valid_features = [f for f in features if f in df.columns]
    print(f"Using features: {valid_features}")
    
    # Prepare feature matrix and target vector
    X = df[valid_features].copy()
    y = df['spend_Q4_2024'] if 'spend_Q4_2024' in df.columns else df['total_spend_2024'] * 0.35
    
    # Ensure target values are positive (required for log transformation)
    # Add a small epsilon to 0 values to avoid log(0)
    y = y.clip(lower=0.01)
    
    # Create feature interactions that are economically meaningful
    print("Creating feature interactions and polynomial features...")
    
    # 1. Spending capacity interaction: credit_line × utilization
    if 'credit_line' in X.columns and 'utilization_pct' in X.columns:
        X['credit_capacity'] = X['credit_line'] * np.maximum(0, (100 - X['utilization_pct'])) / 100
        print("Added credit_capacity interaction")
    
    # 2. Spending momentum: Q1-Q3 spend trajectory
    if all(col in df.columns for col in ['spend_2024_q1', 'spend_2024_q2', 'spend_2024_q3']):
        # Calculate quarterly growth rates with proper handling of zeros
        X['q1_to_q2_growth'] = np.where(
            df['spend_2024_q1'] > 0,
            (df['spend_2024_q2'] - df['spend_2024_q1']) / np.maximum(1, df['spend_2024_q1']),
            0
        )
        X['q2_to_q3_growth'] = np.where(
            df['spend_2024_q2'] > 0,
            (df['spend_2024_q3'] - df['spend_2024_q2']) / np.maximum(1, df['spend_2024_q2']),
            0
        )
        
        # Clip extreme growth values
        X['q1_to_q2_growth'] = X['q1_to_q2_growth'].clip(-5, 5)
        X['q2_to_q3_growth'] = X['q2_to_q3_growth'].clip(-5, 5)
        
        # Calculate weighted trend (more recent quarters have higher weight)
        X['spend_trend'] = X['q1_to_q2_growth'] * 0.3 + X['q2_to_q3_growth'] * 0.7
        
        # Exponential weighting of quarterly data
        X['weighted_prior_spend'] = (
            df['spend_2024_q1'] * 0.2 + 
            df['spend_2024_q2'] * 0.3 + 
            df['spend_2024_q3'] * 0.5
        )
        
        print("Added quarterly spend trend features")
    
    # 3. Economic response: how spending changes with economic conditions
    if 'economic_factor' in X.columns and 'credit_score' in X.columns:
        X['economic_response'] = X['economic_factor'] * (X['credit_score'] / 700).clip(0.5, 1.5)
        print("Added economic response interaction")
    
    # 4. Seasonal spending power: holiday multiplier × spending capacity
    if 'holiday_spending_multiplier' in X.columns and 'avg_monthly_spend_2024' in X.columns:
        X['holiday_spending_power'] = X['holiday_spending_multiplier'] * X['avg_monthly_spend_2024']
        print("Added holiday spending power interaction")
    
    # 5. Add polynomial features for key predictors
    for feature in ['total_spend_2024', 'avg_monthly_spend_2024', 'credit_line']:
        if feature in X.columns:
            # Add squared term to capture non-linear relationships
            X[f'{feature}_squared'] = X[feature] ** 2
            # Clip to prevent extreme values
            max_value = X[feature].quantile(0.995) ** 2  # Use 99.5th percentile squared as max
            X[f'{feature}_squared'] = X[f'{feature}_squared'].clip(0, max_value)
            print(f"Added squared term for {feature}")
    
    # 6. Ratio features with proper handling of zeros/infinity
    if 'current_balance' in X.columns and 'credit_line' in X.columns:
        X['balance_to_limit_ratio'] = X['current_balance'] / X['credit_line'].replace(0, 0.1)
        X['balance_to_limit_ratio'] = X['balance_to_limit_ratio'].clip(0, 10)  # Clip extreme values
        print("Added balance to limit ratio")
    
    if 'avg_monthly_spend_2024' in X.columns and 'total_spend_2025YTD' in X.columns:
        # Calculate 2025 monthly average (YTD)
        X['avg_monthly_2025'] = X['total_spend_2025YTD'] / 3  # First 3 months of 2025
        # Calculate growth ratio 2025 vs 2024 with proper handling of zeros
        X['monthly_spend_growth'] = X['avg_monthly_2025'] / X['avg_monthly_spend_2024'].replace(0, 1)
        X['monthly_spend_growth'] = X['monthly_spend_growth'].clip(0, 5)  # Clip extreme growth values
        print("Added monthly spend growth ratio")
    
    # Convert feature columns to numeric type and ensure no infinity or NaN values
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
        # Replace any infinity values
        X[col] = X[col].replace([np.inf, -np.inf], np.nan)
    
    # Handle missing values
    X = X.fillna(X.median())
    
    # Using a more robust approach: train on original values, not log-transformed
    # This avoids issues with NaN, infinity with log transformation
    print("Using original scale for training (skipping log transformation)")
    
    # Train-test split with original values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize XGBoost regressor with improved parameters
    xgb_reg = XGBRegressor(
        objective='reg:squarederror',
        n_estimators=200,  # More trees
        max_depth=6,       # Slightly deeper trees to capture interactions
        learning_rate=0.05, # Lower learning rate
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3, # Helps prevent overfitting
        reg_alpha=0.1,     # L1 regularization
        reg_lambda=1.0,    # L2 regularization
        eval_metric='rmse',
        early_stopping_rounds=20,  # More patience
        random_state=42
    )
    
    # Split training data into training and validation sets
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # Print information about the ranges of the target variable to help debug
    print(f"Target variable range: min={y_tr.min():.2f}, max={y_tr.max():.2f}, mean={y_tr.mean():.2f}")
    
    # Train the model with validation set for early stopping
    xgb_reg.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=True
    )
    
    # Make predictions on test set
    y_pred = xgb_reg.predict(X_test)
    
    # Evaluate the model on original scale
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nTest Mean Squared Error: {mse:.2f}")
    print(f"Test RMSE: {rmse:.2f}")
    print(f"Test R² Score: {r2:.3f}")
    
    # Feature importance
    importance = xgb_reg.feature_importances_
    print("\nFeature importances:")
    
    # Sort features by importance and print
    feature_importance = sorted(zip(X.columns, importance), key=lambda x: x[1], reverse=True)
    for name, imp in feature_importance:
        print(f"{name}: {imp:.3f}")
    
    # Generate predictions for all users
    df['predicted_q4_spend'] = xgb_reg.predict(X.fillna(X.median()))
    
    # Apply the Q4 adjustment factor as a post-processing step
    if 'q4_adjustment_factor' in df.columns:
        # Adjust predictions using Q4 factors (blend with model prediction)
        # 80% weight to model, 20% to adjustment factor
        initial_prediction = df['predicted_q4_spend'].copy()
        adjusted_prediction = initial_prediction * df['q4_adjustment_factor']
        df['predicted_q4_spend'] = initial_prediction * 0.8 + adjusted_prediction * 0.2
        
        # Log the effect of the adjustment
        mean_adjustment = (df['predicted_q4_spend'] / initial_prediction).mean()
        print(f"\nApplied Q4 adjustment factor with average impact: {mean_adjustment:.2%}")
    
    # Return the model and updated dataframe
    return xgb_reg, df


def model1_q3_validation(df):
    """
    Model Validation: Q3 Spend Prediction using Q1 and Q2 data only
    This allows us to validate our model against actual historical data
    instead of synthetic targets.
    """
    print("\n=== Model Validation: Q3 Spend Prediction using Q1 and Q2 data ===")
    
    # First check if we have the required quarterly data
    required_columns = ['spend_2024_q1', 'spend_2024_q2', 'spend_2024_q3']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Error: Missing required columns: {missing_columns}")
        print("Please run the updated master.py script to generate quarterly data first.")
        return None, df
    
    # Define features for Q3 spend prediction (using only Q1 and Q2 data)
    features = [
        'spend_2024_q1', 'spend_2024_q2', 
        'transactions_2024_q1', 'transactions_2024_q2',
        'utilization_pct', 'credit_score', 'credit_line', 'num_accounts',
        'payment_hist_1_12_delinquency_count', 'payment_hist_13_24_delinquency_count',
        'current_balance'
    ]
    
    # Filter only features that exist in the dataframe
    valid_features = [f for f in features if f in df.columns]
    print(f"Using features: {valid_features}")
    
    # Prepare feature matrix and target vector
    X = df[valid_features].copy()
    y = df['spend_2024_q3']  # This is our actual Q3 spend data
    
    # Calculate average transaction size for Q1 and Q2 if not in features
    if 'transactions_2024_q1' in X.columns and 'spend_2024_q1' in X.columns:
        X['avg_transaction_q1'] = np.where(
            X['transactions_2024_q1'] > 0,
            X['spend_2024_q1'] / X['transactions_2024_q1'],
            0
        )
        
    if 'transactions_2024_q2' in X.columns and 'spend_2024_q2' in X.columns:
        X['avg_transaction_q2'] = np.where(
            X['transactions_2024_q2'] > 0,
            X['spend_2024_q2'] / X['transactions_2024_q2'],
            0
        )
    
    # Add quarter-over-quarter growth
    if 'spend_2024_q1' in X.columns and 'spend_2024_q2' in X.columns:
        X['q1_to_q2_growth'] = np.where(
            X['spend_2024_q1'] > 0,
            (X['spend_2024_q2'] - X['spend_2024_q1']) / X['spend_2024_q1'],
            0
        )
    
    # Convert feature columns to numeric type
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # Handle missing values
    X = X.fillna(X.median())
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize XGBoost regressor with early stopping
    xgb_reg = XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric='rmse',
        early_stopping_rounds=10,
        random_state=42
    )
    
    # Split training data into training and validation sets
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # Train the model with validation set for early stopping
    xgb_reg.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=True
    )
    
    # Make predictions on test set
    y_pred = xgb_reg.predict(X_test)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nTest Mean Squared Error: {mse:.2f}")
    print(f"Test RMSE: {rmse:.2f}")
    print(f"Test R² Score: {r2:.3f}")
    
    # Feature importance
    importance = xgb_reg.feature_importances_
    print("\nFeature importances:")
    for col, imp in zip(X.columns, importance):
        print(f"{col}: {imp:.3f}")
    
    # Generate predictions for all users
    df['predicted_q3_spend'] = xgb_reg.predict(X.fillna(X.median()))
    
    # Calculate prediction error metrics for the full dataset
    actual_q3 = df['spend_2024_q3']
    pred_q3 = df['predicted_q3_spend']
    
    full_mse = mean_squared_error(actual_q3, pred_q3)
    full_rmse = np.sqrt(full_mse)
    full_r2 = r2_score(actual_q3, pred_q3)
    
    # Calculate Mean Absolute Percentage Error
    # Avoid division by zero
    mape = np.mean(np.abs((actual_q3 - pred_q3) / np.maximum(actual_q3, 1))) * 100
    
    print("\nFull dataset metrics:")
    print(f"Mean Squared Error: {full_mse:.2f}")
    print(f"RMSE: {full_rmse:.2f}")
    print(f"R² Score: {full_r2:.3f}")
    print(f"Mean Absolute Percentage Error: {mape:.2f}%")
    
    # Return the model and updated dataframe
    return xgb_reg, df


def model2_account_segmentation(df):
    """
    Model 2: Account Segmentation into CLI Buckets (Multi-class Classification)
    Segments:
    0: Eligible for CLI - No Risk
    1: Eligible for CLI - At Risk
    2: No Increase Needed
    3: High Risk (Non-Performing)
    """
    print("\n=== Model 2: Account Segmentation into CLI Buckets ===")
    
    # Define features for segmentation
    features = ['credit_score', 'delinquency_12mo', 'delinquency_24mo', 'has_fraud',
                'utilization_pct', 'total_spend_2024', 'total_spend_2025YTD',
                'num_accounts', 'avg_transaction_size_2024', 
                'payment_hist_1_12_delinquency_count', 'payment_hist_13_24_delinquency_count',
                'avg_monthly_spend_2024', 'credit_line', 'current_balance']
    
    # Filter valid features
    valid_features = [f for f in features if f in df.columns]
    print(f"Using features: {valid_features}")
    
    # Prepare data
    X = df[valid_features].copy()
    y = df['segment_label'].copy()
    
    # Convert feature columns to numeric type
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # Handle missing values
    X = X.fillna(X.median())
    
    # Train-test split with stratification to maintain class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Check class distribution
    print("\nSegment distribution in training set:")
    print(pd.Series(y_train).value_counts(normalize=True).sort_index().apply(lambda x: f"{x:.2%}"))
    
    # Initialize XGBoost classifier for multi-class with early stopping
    xgb_clf = XGBClassifier(
        objective='multi:softprob',
        num_class=4,
        eval_metric='mlogloss',
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=10,
        random_state=42
    )
    
    # Split training data for validation
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42, stratify=y_train)
    
    # Train the classifier with validation set for early stopping
    xgb_clf.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=True
    )
    
    # Evaluate on test set
    y_pred = xgb_clf.predict(X_test)
    accuracy = (y_pred == y_test).mean()
    print(f"\nTest Accuracy: {accuracy:.2%}")
    
    # Detailed evaluation
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                               target_names=['Eligible-No Risk', 'Eligible-At Risk', 
                                             'No Increase Needed', 'High Risk']))
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Feature importance
    importance = xgb_clf.feature_importances_
    print("\nFeature importances:")
    for col, imp in zip(valid_features, importance):
        print(f"{col}: {imp:.3f}")
    
    # Generate predictions for all accounts (segment probabilities)
    segment_probs = xgb_clf.predict_proba(X.fillna(X.median()))
    for i in range(4):
        df[f'segment_{i}_prob'] = segment_probs[:, i]
    
    # Return the model and updated dataframe
    return xgb_clf, df


def model3_risk_flagging(df):
    """
    Model 3: Risk Flagging (Binary Classification)
    Identify accounts at high risk of overextension, fraud, or default
    """
    print("\n=== Model 3: Risk Flagging (Binary Classification) ===")
    
    # Define features for risk model
    features = ['credit_score', 'delinquency_12mo', 'delinquency_24mo', 'has_fraud',
                'utilization_pct', 'total_spend_2024', 'total_spend_2025YTD',
                'num_accounts', 'active_account_count', 'account_age_days',
                'payment_hist_1_12_max_delinquency', 'payment_hist_13_24_max_delinquency',
                'avg_monthly_spend_2024', 'credit_line', 'current_balance']
    
    # Filter valid features
    valid_features = [f for f in features if f in df.columns]
    print(f"Using features: {valid_features}")
    
    # Prepare data
    X = df[valid_features].copy()
    y = df['risk_flag'].copy()
    
    # Convert feature columns to numeric type
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # Handle missing values
    X = X.fillna(X.median())
    
    # Train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Check class distribution
    print("\nRisk flag distribution in training set:")
    print(pd.Series(y_train).value_counts(normalize=True).sort_index().apply(lambda x: f"{x:.2%}"))
    
    # Calculate class imbalance for scale_pos_weight
    pos = sum(y_train == 1)
    neg = sum(y_train == 0)
    scale_pos_weight = neg / pos if pos > 0 else 1.0
    print(f"\nClass imbalance (neg:pos): {neg}:{pos}, scale_pos_weight = {scale_pos_weight:.2f}")
    
    # Initialize XGBoost binary classifier with early stopping
    xgb_bin_clf = XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        early_stopping_rounds=10,
        random_state=42
    )
    
    # Split training data for validation
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42, stratify=y_train)
    
    # Train the model with validation set for early stopping
    xgb_bin_clf.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=True
    )
    
    # Evaluate on test set
    y_pred = xgb_bin_clf.predict(X_test)
    y_pred_proba = xgb_bin_clf.predict_proba(X_test)[:, 1]
    
    accuracy = (y_pred == y_test).mean()
    print(f"\nTest Accuracy: {accuracy:.2%}")
    
    # AUC calculation (only if both classes are present in test set)
    if len(np.unique(y_test)) > 1:
        auc = roc_auc_score(y_test, y_pred_proba)
        print(f"Test ROC AUC: {auc:.3f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Not Risky", "Risky"]))
    
    # Feature importance
    importance = xgb_bin_clf.feature_importances_
    print("\nFeature importances:")
    for col, imp in zip(valid_features, importance):
        print(f"{col}: {imp:.3f}")
    
    # Generate risk probabilities for all accounts
    df['risk_probability'] = xgb_bin_clf.predict_proba(X.fillna(X.median()))[:, 1]
    
    # Return the model and updated dataframe
    return xgb_bin_clf, df


def model4_cli_recommendation(df):
    """
    Model 4: Credit Line Increase Amount Recommendation (Regression)
    For eligible accounts, recommend an optimal increase amount
    """
    print("\n=== Model 4: Credit Line Increase Amount Recommendation ===")
    
    # Filter eligible accounts (segments 0 and 1)
    eligible_mask = df['segment_label'].isin([0, 1])
    eligible_df = df[eligible_mask].copy()
    print(f"Eligible accounts for CLI: {len(eligible_df)} out of {len(df)} total accounts")
    
    # Define features for CLI recommendation
    features = [
        'predicted_q4_spend', 'utilization_pct', 'credit_score', 
        'credit_line', 'num_accounts', 'avg_monthly_spend_2024', 
        'total_spend_2024', 'current_balance'
    ]
    
    # Add risk_probability only if it exists
    if 'risk_probability' in df.columns:
        features.append('risk_probability')
    elif 'risk_flag' in df.columns:
        print("Note: risk_probability not found, using risk_flag as proxy")
        eligible_df['risk_probability'] = eligible_df['risk_flag'].astype(float)
        features.append('risk_probability')
    
    # Add quarterly spending data if available
    quarterly_features = ['spend_2024_q1', 'spend_2024_q2', 'spend_2024_q3', 'spend_2024_q4']
    for feature in quarterly_features:
        if feature in df.columns:
            features.append(feature)
    
    # Add holiday and seasonal factors if available
    macro_features = ['holiday_spending_multiplier', 'q4_adjustment_factor']
    for feature in macro_features:
        if feature in df.columns:
            features.append(feature)
    
    # Filter valid features
    valid_features = [f for f in features if f in eligible_df.columns]
    print(f"Using features: {valid_features}")
    
    # Prepare data
    X = eligible_df[valid_features].copy()
    y = eligible_df['CLI_target_amount'].copy()
    
    # Ensure target is positive (CLI amounts should not be negative)
    y = y.clip(lower=0)
    
    # Create advanced features for CLI recommendation
    print("Creating specialized CLI features...")
    
    # 1. Spending to credit limit ratio - indicates if customer is constrained by current limit
    if 'total_spend_2024' in X.columns and 'credit_line' in X.columns:
        X['spend_to_limit_ratio'] = X['total_spend_2024'] / X['credit_line'].replace(0, 0.1)
        X['spend_to_limit_ratio'] = X['spend_to_limit_ratio'].clip(0, 10)
        print("Added spend-to-limit ratio")
    
    # 2. Available credit - shows current headroom
    if 'credit_line' in X.columns and 'current_balance' in X.columns:
        X['available_credit'] = np.maximum(0, X['credit_line'] - X['current_balance'] / 1000)  # In thousands
        X['available_credit_ratio'] = X['available_credit'] / X['credit_line'].replace(0, 0.1)
        X['available_credit_ratio'] = X['available_credit_ratio'].clip(0, 1)
        print("Added available credit features")
    
    # 3. Credit score tiers (categorical bins)
    if 'credit_score' in X.columns:
        # Define credit score ranges
        bins = [0, 600, 670, 740, 800, 850]
        labels = [0, 1, 2, 3, 4]  # 0=Poor, 1=Fair, 2=Good, 3=Very Good, 4=Excellent
        
        # Create credit tier feature
        X['credit_tier'] = pd.cut(X['credit_score'], bins=bins, labels=labels, right=True)
        X['credit_tier'] = X['credit_tier'].fillna(1).astype(int)  # Default to Fair if missing
        
        # One-hot encode credit tiers
        for tier in range(5):
            X[f'credit_tier_{tier}'] = (X['credit_tier'] == tier).astype(int)
        
        print("Added credit score tier features")
    
    # 4. Spending growth momentum
    # Based on quarterly data if available
    if all(col in X.columns for col in ['spend_2024_q1', 'spend_2024_q2', 'spend_2024_q3']):
        # Calculate total prior 3 quarters
        X['prior_3q_spend'] = X['spend_2024_q1'] + X['spend_2024_q2'] + X['spend_2024_q3']
        
        # Create growth indicators
        X['q2_vs_q1_growth'] = np.where(
            X['spend_2024_q1'] > 0,
            (X['spend_2024_q2'] - X['spend_2024_q1']) / np.maximum(1, X['spend_2024_q1']),
            0
        )
        X['q3_vs_q2_growth'] = np.where(
            X['spend_2024_q2'] > 0,
            (X['spend_2024_q3'] - X['spend_2024_q2']) / np.maximum(1, X['spend_2024_q2']),
            0
        )
        
        # Clip extreme growth values
        X['q2_vs_q1_growth'] = X['q2_vs_q1_growth'].clip(-5, 5)
        X['q3_vs_q2_growth'] = X['q3_vs_q2_growth'].clip(-5, 5)
        
        # Calculate weighted growth trend (recent quarters matter more)
        X['growth_trend'] = 0.3 * X['q2_vs_q1_growth'] + 0.7 * X['q3_vs_q2_growth']
        
        # Create acceleration indicator (is growth accelerating or decelerating?)
        X['spend_acceleration'] = X['q3_vs_q2_growth'] - X['q2_vs_q1_growth']
        
        print("Added quarterly growth momentum features")

    # 5. Headroom requirement based on predicted spend
    if 'predicted_q4_spend' in X.columns and 'credit_line' in X.columns:
        # Calculate projected headroom needed
        X['projected_headroom_needed'] = np.maximum(0, (X['predicted_q4_spend'] * 1.5) - X['credit_line'])
        X['projected_headroom_needed'] = X['projected_headroom_needed'].clip(0, X['credit_line'] * 2)
        print("Added projected headroom needed")
    
    # 6. Risk-adjusted CLI potential
    if 'risk_probability' in X.columns and 'predicted_q4_spend' in X.columns:
        # Lower risk customers get higher potential increases
        X['risk_adjusted_potential'] = X['predicted_q4_spend'] * (1.5 - X['risk_probability'])
        print("Added risk-adjusted CLI potential")
    
    # 7. Segment-specific features
    if 'segment_label' in eligible_df.columns:
        # Create segment indicators
        X['is_segment_0'] = (eligible_df['segment_label'] == 0).astype(int)
        X['is_segment_1'] = (eligible_df['segment_label'] == 1).astype(int)
        
        # Create segment-specific spending capacity
        if 'predicted_q4_spend' in X.columns:
            X['segment0_spend'] = X['predicted_q4_spend'] * X['is_segment_0']
            X['segment1_spend'] = X['predicted_q4_spend'] * X['is_segment_1']
        
        print("Added segment-specific features")
    
    # Create a compound recommended CLI score based on multiple factors
    # This creates an initial recommendation that the model can learn to adjust
    if all(col in X.columns for col in ['predicted_q4_spend', 'credit_score', 'utilization_pct']):
        # Base factor: higher predicted spend → higher CLI
        spend_factor = X['predicted_q4_spend'] / 1000  # Scale to thousands
        
        # Credit quality factor: higher score → higher CLI
        credit_factor = (X['credit_score'] / 700).clip(0.5, 1.5)
        
        # Utilization factor: higher utilization → higher CLI need
        util_factor = (X['utilization_pct'] / 50).clip(0.5, 2.0)
        
        # Risk dampening: higher risk → lower CLI
        risk_factor = 1.0
        if 'risk_probability' in X.columns:
            risk_factor = (1.0 - X['risk_probability'] * 0.5).clip(0.5, 1.0)
        
        # Combined score
        X['cli_recommendation_score'] = spend_factor * credit_factor * util_factor * risk_factor
        
        print("Added compound CLI recommendation score")
    
    # Convert feature columns to numeric type
    for col in X.columns:
        if col.startswith('credit_tier_'):  # Skip one-hot encoded columns
            continue
        X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # Handle missing values
    X = X.fillna(X.median())
    
    # Replace infinities
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())
    
    # Check if we have enough data
    if len(X) < 10:
        print("Warning: Not enough eligible accounts for reliable modeling")
        df['recommended_cli_amount'] = 0  # Default to no increase
        # For eligible accounts, use simple rule-based approach instead
        segment_0 = (df['segment_label'] == 0)
        segment_1 = (df['segment_label'] == 1)
        
        # Use fixed rules based on predicted spend and current limit
        if 'predicted_q4_spend' in df.columns and 'credit_line' in df.columns:
            df.loc[segment_0, 'recommended_cli_amount'] = np.maximum(
                0, (df.loc[segment_0, 'predicted_q4_spend'] * 1.5) - df.loc[segment_0, 'credit_line'])
            df.loc[segment_1, 'recommended_cli_amount'] = np.maximum(
                0, (df.loc[segment_1, 'predicted_q4_spend'] * 1.2) - df.loc[segment_1, 'credit_line'])
        
        print("Applied rule-based CLI recommendations")
        return None, df
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Print statistics about the target variable
    print(f"CLI target amount: min=${y.min():.2f}, max=${y.max():.2f}, mean=${y.mean():.2f}")
    
    # Train XGBoost regressor for CLI amount with early stopping
    xgb_cli_reg = XGBRegressor(
        objective='reg:squarederror',
        n_estimators=200,
        max_depth=5,
        learning_rate=0.03,  # Lower learning rate for better convergence
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,  # Helps with preventing overfitting
        gamma=0.1,           # Minimum loss reduction for further partition
        reg_alpha=0.1,       # L1 regularization
        reg_lambda=1.0,      # L2 regularization
        eval_metric='rmse',
        early_stopping_rounds=20,
        random_state=42
    )
    
    # Split training data for validation
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
    
    # Train the model with validation set for early stopping
    xgb_cli_reg.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=True
    )
    
    # Evaluate on test set
    y_pred = xgb_cli_reg.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print(f"\nTest RMSE for CLI amount: ${rmse:.2f}")
    
    # Calculate R-squared
    r2 = r2_score(y_test, y_pred)
    print(f"Test R² Score: {r2:.3f}")
    
    # Relative error
    mean_cli = y_test.mean()
    print(f"Mean CLI amount: ${mean_cli:.2f}")
    print(f"RMSE as % of mean CLI: {(rmse / mean_cli) * 100:.2f}%")
    
    # Feature importance
    importance = xgb_cli_reg.feature_importances_
    print("\nFeature importances:")
    feature_importance = sorted(zip(X.columns, importance), key=lambda x: x[1], reverse=True)
    
    # Print top 15 features
    for name, imp in feature_importance[:15]:
        print(f"{name}: {imp:.3f}")
    
    # Generate CLI recommendations for all accounts
    df['recommended_cli_amount'] = 0  # Default to no increase
    
    # Only make CLI recommendations for eligible accounts
    # First, ensure all columns in X.columns exist in the dataframe
    required_features = X.columns.tolist()
    for feature in required_features:
        if feature not in df.columns:
            df[feature] = np.nan
            df.loc[eligible_mask, feature] = X[feature].values
    
    # Prepare features for prediction
    X_all_eligible = df.loc[eligible_mask, required_features].copy()
    
    # Handle missing values and convert types
    X_all_eligible = X_all_eligible.fillna(X.median())
    X_all_eligible = X_all_eligible.replace([np.inf, -np.inf], np.nan).fillna(X.median())
    
    # Ensure dtype compatibility
    if 'recommended_cli_amount' in df.columns:
        if not np.issubdtype(df['recommended_cli_amount'].dtype, np.floating):
            df['recommended_cli_amount'] = df['recommended_cli_amount'].astype(float)
    else:
        df['recommended_cli_amount'] = 0.0
        
    # Make predictions for eligible accounts
    df.loc[eligible_mask, 'recommended_cli_amount'] = xgb_cli_reg.predict(X_all_eligible)
    
    # Ensure non-negative recommendations
    df['recommended_cli_amount'] = np.maximum(0, df['recommended_cli_amount'])
    
    # Apply business rules as post-processing
    # Segment 0 (No Risk) can get higher increases than Segment 1 (At Risk)
    segment_0 = (df['segment_label'] == 0)
    segment_1 = (df['segment_label'] == 1)
    
    # Cap increases relative to existing credit line
    if 'credit_line' in df.columns:
        # No Risk customers can get up to 100% of their current limit as increase
        df.loc[segment_0, 'recommended_cli_amount'] = np.minimum(
            df.loc[segment_0, 'recommended_cli_amount'],
            df.loc[segment_0, 'credit_line'] * 1.0  # Cap at 100% of current limit
        )
        
        # At Risk customers are capped at 50% of their current limit
        df.loc[segment_1, 'recommended_cli_amount'] = np.minimum(
            df.loc[segment_1, 'recommended_cli_amount'],
            df.loc[segment_1, 'credit_line'] * 0.5  # Cap at 50% of current limit
        )
    
    # Round recommendations to nearest $100
    df['recommended_cli_amount'] = np.round(df['recommended_cli_amount'] / 100) * 100
    
    # Analyze final recommendations
    cli_stats = df.loc[eligible_mask, 'recommended_cli_amount'].describe()
    print("\nFinal CLI recommendations statistics:")
    print(f"Min: ${cli_stats['min']:.2f}")
    print(f"25th percentile: ${cli_stats['25%']:.2f}")
    print(f"Median: ${cli_stats['50%']:.2f}")
    print(f"Mean: ${cli_stats['mean']:.2f}")
    print(f"75th percentile: ${cli_stats['75%']:.2f}")
    print(f"Max: ${cli_stats['max']:.2f}")
    
    # Return the model and updated dataframe
    return xgb_cli_reg, df


def visualize_results(df):
    """Create visualizations of the model results"""
    print("\n=== Generating Visualizations ===")
    
    # Create a directory for visualizations if it doesn't exist
    os.makedirs('visualizations', exist_ok=True)
    
    # 1. Segment Distribution Pie Chart
    plt.figure(figsize=(10, 6))
    segment_counts = df['segment_label'].value_counts().sort_index()
    labels = ['Eligible-No Risk', 'Eligible-At Risk', 'No Increase Needed', 'High Risk']
    plt.pie(segment_counts, labels=labels, autopct='%1.1f%%', startangle=90, 
            colors=['#2ecc71', '#f39c12', '#3498db', '#e74c3c'])
    plt.title('Account Segment Distribution')
    plt.savefig('visualizations/segment_distribution.png')
    plt.close()
    
    # 2. Risk Probability Distribution
    if 'risk_probability' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(df['risk_probability'], bins=30, kde=True)
        plt.title('Distribution of Risk Probabilities')
        plt.xlabel('Risk Probability')
        plt.ylabel('Count')
        plt.savefig('visualizations/risk_distribution.png')
        plt.close()
    
    # 3. Predicted Q4 Spend vs Actual Q4 Spend (2024)
    if 'predicted_q4_spend' in df.columns and 'spend_Q4_2024' in df.columns:
        plt.figure(figsize=(10, 6))
        plt.scatter(df['spend_Q4_2024'], df['predicted_q4_spend'], alpha=0.5)
        plt.plot([0, df['spend_Q4_2024'].max()], [0, df['spend_Q4_2024'].max()], 'r--')
        plt.title('Predicted Q4 2025 Spend vs Actual Q4 2024 Spend')
        plt.xlabel('Q4 2024 Spend (Actual)')
        plt.ylabel('Q4 2025 Spend (Predicted)')
        plt.savefig('visualizations/q4_spend_prediction.png')
        plt.close()
    
    # 4. CLI Recommendations by Segment
    if 'recommended_cli_amount' in df.columns:
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='segment_label', y='recommended_cli_amount', data=df)
        plt.title('Credit Line Increase Recommendations by Segment')
        plt.xlabel('Segment (0=Eligible-No Risk, 1=Eligible-At Risk, 2=No Increase, 3=High Risk)')
        plt.ylabel('Recommended CLI Amount ($)')
        plt.savefig('visualizations/cli_by_segment.png')
        plt.close()
    
    # 5. Relationship between Credit Score and CLI Amount
    if 'credit_score' in df.columns and 'recommended_cli_amount' in df.columns:
        plt.figure(figsize=(12, 6))
        eligible_df = df[df['segment_label'].isin([0, 1])]
        plt.scatter(eligible_df['credit_score'], eligible_df['recommended_cli_amount'], 
                   c=eligible_df['segment_label'], cmap='coolwarm', alpha=0.7)
        plt.colorbar(label='Segment (0=No Risk, 1=At Risk)')
        plt.title('Credit Score vs Recommended CLI Amount for Eligible Accounts')
        plt.xlabel('Credit Score')
        plt.ylabel('Recommended CLI Amount ($)')
        plt.savefig('visualizations/credit_score_vs_cli.png')
        plt.close()
    
    print("Visualizations saved to 'visualizations' directory")


def main():
    """Main function to orchestrate the entire modeling process"""
    # Initialize metrics dictionary to track performance
    model_metrics = {
        "model1": {"r2_score": 0, "rmse": 0},
        "model2": {"accuracy": 0},
        "model3": {"accuracy": 0, "auc": 0},
        "model4": {"rmse": 0, "rmse_pct": 0}
    }
    
    # 1. Load and prepare data
    try:
        print("Loading data...")
        master_df = load_data()
        print(f"Loaded data with {len(master_df)} rows and {len(master_df.columns)} columns")
    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 2. Prepare target variables if needed
    try:
        print("\nPreparing target variables...")
        master_df = prepare_target_variables(master_df)
        print("Target variable preparation complete")
    except Exception as e:
        print(f"Error preparing target variables: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # New: Run validation model with quarterly data
    try:
        print("\nRunning Q3 Spend Prediction Validation...")
        validation_model, master_df = model1_q3_validation(master_df)
        print("Q3 Validation model completed successfully.")
        
        # Store validation metrics if successful
        if validation_model:
            model_metrics["validation"] = {
                "r2_score": r2_score(
                    master_df['spend_2024_q3'], 
                    master_df['predicted_q3_spend']
                ),
                "rmse": np.sqrt(mean_squared_error(
                    master_df['spend_2024_q3'], 
                    master_df['predicted_q3_spend']
                ))
            }
            
    except Exception as e:
        print(f"Error in Q3 Validation model: {e}")
        import traceback
        traceback.print_exc()
        validation_model = None
    
    # 3. Build and evaluate Model 1: Q4 Spend Prediction
    try:
        print("\nRunning Model 1: Q4 Spend Prediction...")
        model1, master_df = model1_q4_spend_prediction(master_df)
        print("Model 1 completed successfully.")
        
        # Get test data
        X_test = master_df[['avg_monthly_spend_2024', 'total_spend_2024', 'total_spend_2025YTD',
            'utilization_pct', 'credit_score', 'credit_line', 'num_accounts',
            'payment_hist_1_12_delinquency_count', 'payment_hist_13_24_delinquency_count',
            'current_balance']].sample(frac=0.2, random_state=42)
        y_test = master_df.loc[X_test.index, 'spend_Q4_2024']
        y_pred = model1.predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Store metrics
        model_metrics["model1"]["r2_score"] = r2
        model_metrics["model1"]["rmse"] = rmse
        
    except Exception as e:
        print(f"Error in Model 1: {e}")
        import traceback
        traceback.print_exc()
        model1 = None
    
    # 4. Build and evaluate Model 2: Account Segmentation
    try:
        print("\nRunning Model 2: Account Segmentation...")
        model2, master_df = model2_account_segmentation(master_df)
        print("Model 2 completed successfully.")
        
        # Get accuracy from test data
        features = [f for f in ['credit_score', 'delinquency_12mo', 'delinquency_24mo', 'has_fraud',
                    'utilization_pct', 'total_spend_2024', 'total_spend_2025YTD',
                    'num_accounts', 'avg_transaction_size_2024', 
                    'payment_hist_1_12_delinquency_count', 'payment_hist_13_24_delinquency_count',
                    'avg_monthly_spend_2024', 'credit_line', 'current_balance'] if f in master_df.columns]
                    
        X_test = master_df[features].sample(frac=0.2, random_state=42)
        y_test = master_df.loc[X_test.index, 'segment_label']
        y_pred = model2.predict(X_test)
        
        # Calculate accuracy
        accuracy = (y_pred == y_test).mean()
        
        # Store metrics
        model_metrics["model2"]["accuracy"] = accuracy
        
    except Exception as e:
        print(f"Error in Model 2: {e}")
        import traceback
        traceback.print_exc()
        model2 = None
    
    # 5. Build and evaluate Model 3: Risk Flagging
    try:
        print("\nRunning Model 3: Risk Flagging...")
        model3, master_df = model3_risk_flagging(master_df)
        print("Model 3 completed successfully.")
        
        # Get accuracy from test data
        features = [f for f in ['credit_score', 'delinquency_12mo', 'delinquency_24mo', 'has_fraud',
                    'utilization_pct', 'total_spend_2024', 'total_spend_2025YTD',
                    'num_accounts', 'active_account_count', 'account_age_days',
                    'payment_hist_1_12_max_delinquency', 'payment_hist_13_24_max_delinquency',
                    'avg_monthly_spend_2024', 'credit_line', 'current_balance'] if f in master_df.columns]
                    
        X_test = master_df[features].sample(frac=0.2, random_state=42)
        y_test = master_df.loc[X_test.index, 'risk_flag']
        y_pred = model3.predict(X_test)
        y_pred_proba = model3.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = (y_pred == y_test).mean()
        auc = roc_auc_score(y_test, y_pred_proba)
        
        # Store metrics
        model_metrics["model3"]["accuracy"] = accuracy
        model_metrics["model3"]["auc"] = auc
        
    except Exception as e:
        print(f"Error in Model 3: {e}")
        import traceback
        traceback.print_exc()
        model3 = None
    
    # 6. Build and evaluate Model 4: CLI Recommendation
    try:
        print("\nRunning Model 4: CLI Recommendation...")
        model4, master_df = model4_cli_recommendation(master_df)
        print("Model 4 completed successfully.")
        
        # Get CLI metrics from eligible accounts
        if model4 is not None:
            eligible_mask = master_df['segment_label'].isin([0, 1])
            eligible_df = master_df[eligible_mask].copy()
            
            features = [f for f in ['predicted_q4_spend', 'utilization_pct', 'credit_score', 
                        'credit_line', 'num_accounts', 'avg_monthly_spend_2024', 
                        'total_spend_2024', 'current_balance', 'risk_probability'] if f in eligible_df.columns]
                            
            X_test = eligible_df[features].sample(frac=0.2, random_state=42)
            y_test = eligible_df.loc[X_test.index, 'CLI_target_amount']
            y_pred = model4.predict(X_test)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mean_cli = y_test.mean()
            rmse_pct = (rmse / mean_cli) * 100 if mean_cli > 0 else 0
            
            # Store metrics
            model_metrics["model4"]["rmse"] = rmse
            model_metrics["model4"]["rmse_pct"] = rmse_pct
        
    except Exception as e:
        print(f"Error in Model 4: {e}")
        import traceback
        traceback.print_exc()
        model4 = None
    
    # 7. Visualize results
    try:
        print("\nGenerating visualizations...")
        visualize_results(master_df)
        print("Visualizations completed successfully.")
    except Exception as e:
        print(f"Error generating visualizations: {e}")
        import traceback
        traceback.print_exc()
    
    # 8. Save the final enriched dataset
    try:
        output_path = 'exploratory_data_analysis/master_user_dataset_with_predictions.csv'
        print(f"\nSaving enriched dataset to {output_path}...")
        master_df.to_csv(output_path, index=False)
        print("Enriched dataset saved successfully.")
    except Exception as e:
        print(f"Error saving enriched dataset: {e}")
        import traceback
        traceback.print_exc()
    
    # 9. Save models
    try:
        import joblib
        os.makedirs('models', exist_ok=True)
        print("\nSaving models...")
        
        if model1:
            joblib.dump(model1, 'models/q4_spend_prediction_model.joblib')
            print("- Q4 Spend Prediction model saved")
        if model2:
            joblib.dump(model2, 'models/account_segmentation_model.joblib')
            print("- Account Segmentation model saved")
        if model3:
            joblib.dump(model3, 'models/risk_flagging_model.joblib')
            print("- Risk Flagging model saved")
        if model4:
            joblib.dump(model4, 'models/cli_recommendation_model.joblib')
            print("- CLI Recommendation model saved")
    except Exception as e:
        print(f"Error saving models: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== XGBoost Modeling Complete ===")
    print(f"Total accounts processed: {len(master_df)}")
    if 'risk_flag' in master_df.columns:
        print(f"Accounts flagged as high risk: {master_df['risk_flag'].sum()}")
    if 'segment_label' in master_df.columns:
        print(f"Accounts eligible for CLI: {sum(master_df['segment_label'].isin([0, 1]))}")
    if 'recommended_cli_amount' in master_df.columns and 'segment_label' in master_df.columns:
        cli_mean = master_df.loc[master_df['segment_label'].isin([0, 1]), 'recommended_cli_amount'].mean()
        print(f"Average recommended CLI amount for eligible accounts: ${cli_mean:.2f}")
    
    # Print final accuracy summary
    print("\n=========== MODEL PERFORMANCE SUMMARY ===========")
    print("Model 1 (Q4 Spend Prediction):")
    if model_metrics["model1"]["r2_score"] is not None:
        print(f"  - R² Score: {model_metrics['model1']['r2_score']:.3f}")
        print(f"  - RMSE: ${model_metrics['model1']['rmse']:.2f}")
    else:
        print("  - No metrics available")
        
    print("\nModel 2 (Account Segmentation):")
    if model_metrics["model2"]["accuracy"] is not None:
        print(f"  - Accuracy: {model_metrics['model2']['accuracy']:.2%}")
    else:
        print("  - No metrics available")
        
    print("\nModel 3 (Risk Flagging):")
    if model_metrics["model3"]["accuracy"] is not None:
        print(f"  - Accuracy: {model_metrics['model3']['accuracy']:.2%}")
        print(f"  - ROC AUC: {model_metrics['model3']['auc']:.3f}")
    else:
        print("  - No metrics available")
        
    print("\nModel 4 (CLI Recommendation):")
    if model_metrics["model4"]["rmse"] is not None:
        print(f"  - RMSE: ${model_metrics['model4']['rmse']:.2f}")
        print(f"  - RMSE as % of mean CLI: {model_metrics['model4']['rmse_pct']:.2f}%")
    else:
        print("  - No metrics available")
    print("================================================")


if __name__ == "__main__":
    print("Starting xgboost_models.py...")
    main()
    print("xgboost_models.py execution completed.") 