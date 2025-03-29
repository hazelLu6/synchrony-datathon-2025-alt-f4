# Synchrony Credit Line Increase Prediction System

## Overview
This system provides a comprehensive solution for credit line increase (CLI) recommendations based on account performance, spending patterns, and risk assessment. It uses XGBoost models to predict customer spending, segment accounts, identify risky accounts, and recommend optimal credit line increases.

## File Structure

### Main Files
- `master.py`: Data preparation and feature engineering
- `xgboost_models.py`: Machine learning models implementation
- `test_models.py`: Test script to validate model performance

### Data Directory
- Contains raw data files including transaction facts, account dimensions, etc.

### Output Directories
- `exploratory_data_analysis/`: Stores processed datasets
- `models/`: Stores trained model files
- `visualizations/`: Stores model performance visualizations

## Key Functions

### master.py
- **clean_and_aggregate_data()**: Loads raw data files, processes them, and creates a master dataset with engineered features
- **compare_fraud_accounts()**: Generates comparison reports between fraudulent and non-fraudulent accounts

### xgboost_models.py
- **load_data()**: Loads the master dataset for modeling
- **prepare_target_variables()**: Creates target variables for all models:
  - Q4 spend target
  - Account segment labels (0-3)
  - Risk flags (0/1)
  - CLI target amounts
- **add_macroeconomic_features()**: Adds economic indicators and seasonal patterns for Q4 predictions
- **model1_q4_spend_prediction()**: Predicts customer spending for Q4
- **model1_q3_validation()**: Validates prediction approach using historical Q3 data
- **model2_account_segmentation()**: Classifies accounts into 4 segments
- **model3_risk_flagging()**: Identifies high-risk accounts
- **model4_cli_recommendation()**: Recommends optimal CLI amounts for eligible accounts
- **visualize_results()**: Creates visualizations of model results
- **main()**: Orchestrates the entire modeling process

### test_models.py
- Main script that tests all models with random data when needed

## How to Run

### 1. Data Preparation
```bash
python master.py
```
This creates the master dataset at `exploratory_data_analysis/master_user_dataset.csv`.

### 2. Model Training and Prediction
```bash
python xgboost_models.py
```
This trains all models and saves:
- Trained models to `models/` directory
- Visualizations to `visualizations/` directory
- Enriched dataset with predictions to `exploratory_data_analysis/master_user_dataset_with_predictions.csv`

### 3. Model Testing
```bash
python test_models.py
```
This validates model performance using test data.

## Data Merging Logic

### Indexes Used
- **current_account_nbr**: Primary key for account-level data
- **user_id**: Primary key for user-level data (one user can have multiple accounts)
- **transaction_date**: Used for temporal filtering and aggregation

### Merging Process
1. **Account to User Mapping**:
   - `df_syf_id` maps `current_account_nbr` to `user_id`
   - Aggregation counts unique accounts per user

2. **Account Dimension Data**:
   - Merged with user mapping on `current_account_nbr`

3. **Transaction Data**:
   - Both regular and world store transactions are combined
   - Temporal flags identify transactions in different time periods
   - Aggregated at account level with various spending metrics

4. **Fraud Data**:
   - Merged on `current_account_nbr` to flag accounts with fraud
   - Additional fraud transaction details included when available

5. **Financial Data (RAMS)**:
   - Merged on `current_account_nbr` for credit line, balance, and score information

6. **User-Level Aggregation**:
   - All account-level data is aggregated to user level
   - Different aggregation functions used for different metrics:
     - Sum for transaction counts, spending amounts
     - Min for credit scores
     - Max for risk flags and delinquency indicators

## Model Outputs

### Model 1: Q4 Spend Prediction
- **Output**: Predicted Q4 spending amount for each account
- **Performance**: R² Score = 0.969, RMSE = $6,830

### Model 2: Account Segmentation
- **Output**: Segment classification (0-3)
  - 0: Eligible for CLI - No Risk
  - 1: Eligible for CLI - At Risk
  - 2: No Increase Needed
  - 3: High Risk (Non-Performing)
- **Performance**: Accuracy = 94.8%

### Model 3: Risk Flagging
- **Output**: Binary risk classification (0/1)
- **Performance**: Accuracy = 91.3%, ROC AUC = 0.71

### Model 4: CLI Recommendation
- **Output**: Recommended CLI amount for eligible accounts
- **Performance**: RMSE = $3,200 (improved from original $24,800)

## Enhancements
The system includes advanced features such as:
- Macroeconomic indicators for Q4 spending patterns
- Seasonal adjustment factors
- Customer-specific spending elasticity
- Risk-adjusted recommendation amounts
- Business rule constraints for CLI limits