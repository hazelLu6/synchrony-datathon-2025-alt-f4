import pandas as pd
import numpy as np
import os

def clean_and_aggregate_data():
    # Load CSV files from the data/ directory
    df_account_dim = pd.read_csv("data/account_dim_20250325.csv")
    df_syf_id = pd.read_csv("data/syf_id_20250325.csv")
    df_statement = pd.read_csv("data/statement_fact_20250325.csv")
    df_trans = pd.read_csv("data/transaction_fact_20250325.csv")
    df_wrld_trans = pd.read_csv("data/wrld_stor_tran_fact_20250325.csv")
    df_fraud_case = pd.read_csv("data/fraud_claim_case_20250325.csv")
    df_fraud_tran = pd.read_csv("data/fraud_claim_tran_20250325.csv")  # available if needed
    df_rams = pd.read_csv("data/rams_batch_cur_20250325.csv")
    
    print("Loaded account_dim:", df_account_dim.shape)
    print("Loaded syf_id:", df_syf_id.shape)
    print("Loaded statement_fact:", df_statement.shape)
    print("Loaded transaction_fact:", df_trans.shape)
    print("Loaded wrld_stor_tran_fact:", df_wrld_trans.shape)
    print("Loaded fraud_claim_case:", df_fraud_case.shape)
    print("Loaded fraud_claim_tran:", df_fraud_tran.shape)
    print("Loaded rams_batch_cur:", df_rams.shape)
    
    # Rename syf_id columns so we can merge on current_account_nbr and get user_id
    df_syf_id.rename(columns={"account_nbr_pty": "current_account_nbr", "ds_id": "user_id"}, inplace=True)
    
    # Create a mapping of accounts to users before any merges
    # This will be used later to get the correct account count per user
    account_to_user_map = df_syf_id[["current_account_nbr", "user_id"]].copy()
    unique_account_counts = account_to_user_map.groupby("user_id")["current_account_nbr"].nunique().reset_index()
    unique_account_counts.rename(columns={"current_account_nbr": "num_accounts"}, inplace=True)
    
    # First merge account dimension with user_id mapping
    df_merged = pd.merge(df_account_dim, df_syf_id[["current_account_nbr", "user_id"]],
                         how="inner", on="current_account_nbr")
    print("After merging account_dim and syf_id:", df_merged.shape)
    
    # Rename is_employee to is_high_income if it exists
    if "is_employee" in df_merged.columns:
        df_merged.rename(columns={"is_employee": "is_high_income"}, inplace=True)
    
    # Merge statement fact (if available) on current_account_nbr
    if df_statement.shape[0] > 0:
        # Pre-aggregate statement data to avoid duplicating rows
        df_statement_agg = df_statement.groupby("current_account_nbr", as_index=False).agg({
            "return_check_cnt_total": "sum" if "return_check_cnt_total" in df_statement.columns else "count"
        })
        df_merged = pd.merge(df_merged, df_statement_agg, how="left", on="current_account_nbr")
    
    # Combine transaction facts and convert dates (do not drop rows with NaT)
    df_trans_all = pd.concat([df_trans, df_wrld_trans], ignore_index=True)
    df_trans_all["transaction_date"] = pd.to_datetime(df_trans_all["transaction_date"], errors="coerce")
    
    # Define time period functions - add quarterly breakdowns
    def is_2024(date_val):
        return (date_val >= pd.Timestamp("2024-01-01")) & (date_val <= pd.Timestamp("2024-12-31"))
    
    def is_2025_ytd(date_val):
        return (date_val >= pd.Timestamp("2025-01-01")) & (date_val <= pd.Timestamp("2025-03-24"))
    
    # Add quarterly period functions
    def is_2024_q1(date_val):
        return (date_val >= pd.Timestamp("2024-01-01")) & (date_val <= pd.Timestamp("2024-03-31"))
    
    def is_2024_q2(date_val):
        return (date_val >= pd.Timestamp("2024-04-01")) & (date_val <= pd.Timestamp("2024-06-30"))
    
    def is_2024_q3(date_val):
        return (date_val >= pd.Timestamp("2024-07-01")) & (date_val <= pd.Timestamp("2024-09-30"))
    
    def is_2024_q4(date_val):
        return (date_val >= pd.Timestamp("2024-10-01")) & (date_val <= pd.Timestamp("2024-12-31"))
    
    # Apply time period flags
    df_trans_all["in_2024"] = is_2024(df_trans_all["transaction_date"])
    df_trans_all["in_2025_ytd"] = is_2025_ytd(df_trans_all["transaction_date"])
    
    # Add quarterly flags
    df_trans_all["in_2024_q1"] = is_2024_q1(df_trans_all["transaction_date"])
    df_trans_all["in_2024_q2"] = is_2024_q2(df_trans_all["transaction_date"])
    df_trans_all["in_2024_q3"] = is_2024_q3(df_trans_all["transaction_date"])
    df_trans_all["in_2024_q4"] = is_2024_q4(df_trans_all["transaction_date"])
    
    df_trans_all["is_purchase"] = df_trans_all["transaction_type"].eq("SALE")
    df_trans_all["is_return"] = df_trans_all["transaction_type"].eq("RETURN")
    df_trans_all["signed_amount"] = np.where(df_trans_all["is_return"],
                                             -df_trans_all["transaction_amt"],
                                             df_trans_all["transaction_amt"])
    
    # Pre-aggregate transaction data by account to avoid duplicating rows
    trans_agg = df_trans_all.groupby("current_account_nbr").apply(lambda g: pd.Series({
        # Original aggregations
        "total_spend_2024": g.loc[g["in_2024"], "signed_amount"].sum(),
        "total_transactions_2024": g.loc[g["in_2024"] & g["is_purchase"]].shape[0],
        "total_spend_2025YTD": g.loc[g["in_2025_ytd"], "signed_amount"].sum(),
        "total_transactions_2025YTD": g.loc[g["in_2025_ytd"] & g["is_purchase"]].shape[0],
        
        # Add quarterly aggregations
        "spend_2024_q1": g.loc[g["in_2024_q1"], "signed_amount"].sum(),
        "transactions_2024_q1": g.loc[g["in_2024_q1"] & g["is_purchase"]].shape[0],
        "spend_2024_q2": g.loc[g["in_2024_q2"], "signed_amount"].sum(),
        "transactions_2024_q2": g.loc[g["in_2024_q2"] & g["is_purchase"]].shape[0],
        "spend_2024_q3": g.loc[g["in_2024_q3"], "signed_amount"].sum(),
        "transactions_2024_q3": g.loc[g["in_2024_q3"] & g["is_purchase"]].shape[0],
        "spend_2024_q4": g.loc[g["in_2024_q4"], "signed_amount"].sum(),
        "transactions_2024_q4": g.loc[g["in_2024_q4"] & g["is_purchase"]].shape[0],
    })).reset_index()
    
    # Use left merge to avoid duplicating rows
    df_merged = pd.merge(df_merged, trans_agg, how="left", on="current_account_nbr")
    
    # Process fraud case data
    if df_fraud_case.shape[0] > 0:
        df_fraud_case["has_fraud"] = 1
        df_fraud_case_agg = df_fraud_case.groupby("current_account_nbr", as_index=False)["has_fraud"].max()
        df_merged = pd.merge(df_merged, df_fraud_case_agg, how="left", on="current_account_nbr")
        # Explicitly fill NaN values with 0 for accounts with no fraud
        df_merged["has_fraud"] = df_merged["has_fraud"].fillna(0).astype(int)
        print(f"Fraud accounts: {df_merged['has_fraud'].sum()} out of {len(df_merged)} total accounts")
    else:
        # If no fraud cases, ensure the column exists with all zeros
        df_merged["has_fraud"] = 0
        print("No fraud accounts found in the data")
    
    # Process fraud transaction data
    if df_fraud_tran.shape[0] > 0:
        # Convert transaction amount to numeric and transaction date to datetime
        df_fraud_tran["transaction_am"] = pd.to_numeric(df_fraud_tran["transaction_am"], errors="coerce")
        df_fraud_tran["transaction_dt"] = pd.to_datetime(df_fraud_tran["transaction_dt"], errors="coerce")
        
        # Aggregate fraud transaction data by account
        df_fraud_tran_agg = df_fraud_tran.groupby("current_account_nbr", as_index=False).agg({
            "transaction_am": ["count", "sum", "mean", "max", "min"],
            "case_id": "nunique"  # Count unique fraud cases per account
        })
        
        # Flatten the column names
        df_fraud_tran_agg.columns = ['_'.join(col).strip('_') for col in df_fraud_tran_agg.columns.values]
        
        # Rename columns for clarity
        df_fraud_tran_agg.rename(columns={
            "current_account_nbr": "current_account_nbr",
            "transaction_am_count": "fraud_transaction_count",
            "transaction_am_sum": "fraud_transaction_sum",
            "transaction_am_mean": "fraud_transaction_avg",
            "transaction_am_max": "fraud_transaction_max",
            "transaction_am_min": "fraud_transaction_min",
            "case_id_nunique": "fraud_case_count"
        }, inplace=True)
        
        # Merge with main dataframe
        df_merged = pd.merge(df_merged, df_fraud_tran_agg, how="left", on="current_account_nbr")
        
        # Fill NaN values with 0 for accounts with no fraud transactions
        fraud_transaction_cols = [
            "fraud_transaction_count", "fraud_transaction_sum", 
            "fraud_transaction_avg", "fraud_transaction_max", 
            "fraud_transaction_min", "fraud_case_count"
        ]
        for col in fraud_transaction_cols:
            if col in df_merged.columns:
                df_merged[col] = df_merged[col].fillna(0)
    
    # Merge RAMS data, renaming key fields for clarity
    if df_rams.shape[0] > 0:
        df_rams.rename(columns={
            "cu_account_nbr": "current_account_nbr",
            "cu_crd_line": "credit_line",
            "cu_cur_balance": "current_balance",
            "cu_bhv_scr": "behavior_score",
            "cu_crd_bureau_scr": "credit_score"
        }, inplace=True)
        df_merged = pd.merge(df_merged, df_rams, how="left", on="current_account_nbr")
    
    # Clean payment history fields
    def parse_payment_history(history_string):
        if pd.isna(history_string) or not isinstance(history_string, str):
            return {
                'delinquency_count': 0,
                'max_delinquency': 0,
                'zero_balance_months': 0,
                'credit_balance_months': 0,
                'normal_months': 0
            }
        
        # Clean the string
        history = history_string.strip().upper()
        
        # Define delinquency mapping
        delinq_map = {
            'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7,
            'J': 1, 'K': 2, 'L': 3, 'M': 4, 'N': 5, 'O': 6, 'P': 7,
            '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7
        }
        
        delinq_values = [delinq_map.get(ch, 0) for ch in history]
        credit_balance = sum(1 for ch in history if ch in ['%', '#', '+', '-'])
        zero_balance = sum(1 for ch in history if ch == 'Z')
        normal_months = sum(1 for ch in history if ch in ['A', 'I', 'Q'])
        
        return {
            'delinquency_count': sum(1 for x in delinq_values if x > 0),
            'max_delinquency': max(delinq_values, default=0),
            'zero_balance_months': zero_balance,
            'credit_balance_months': credit_balance,
            'normal_months': normal_months
        }

    for period in ['1_12', '13_24']:
        field = f'payment_hist_{period}_mths'
        if field in df_merged.columns:
            history_features = df_merged[field].apply(parse_payment_history).apply(pd.Series)
            history_features = history_features.add_prefix(f'payment_hist_{period}_')
            df_merged = pd.concat([df_merged, history_features], axis=1)
            # Drop original payment history column to save memory
            df_merged = df_merged.drop(columns=[field])
    
    # Before final aggregation, add these lines after merging all data
    def prepare_numeric_features(df):
        # Convert dates to numeric features - ensure consistent timezone handling
        reference_date = pd.Timestamp('2025-03-25', tz='UTC')
        
        # More robust datetime conversion for open_date
        try:
            # First convert to datetime if needed
            df['open_date'] = pd.to_datetime(df['open_date'], errors='coerce')
            
            # Check if we have any valid datetime values before attempting to use .dt accessor
            if df['open_date'].notna().any():
                # Now localize timezone only if there are valid values and they don't already have timezone
                if hasattr(df['open_date'], 'dt') and df['open_date'].dt.tz is None:
                    df['open_date'] = df['open_date'].dt.tz_localize('UTC')
            
            # Calculate account age only for non-null values
            if hasattr(df['open_date'], 'dt'):
                df['account_age_days'] = (reference_date - df['open_date']).dt.days
            else:
                df['account_age_days'] = pd.NA
        except Exception as e:
            print(f"Error processing open_date: {e}")
            df['account_age_days'] = pd.NA
        
        # Similar robust handling for card_activation_date
        if 'card_activation_date' in df.columns:
            try:
                df['card_activation_date'] = pd.to_datetime(df['card_activation_date'], errors='coerce')
                
                # Only proceed with datetime operations if we have valid datetime values
                if df['card_activation_date'].notna().any():
                    # Now try to localize timezone
                    if hasattr(df['card_activation_date'], 'dt') and df['card_activation_date'].dt.tz is None:
                        df['card_activation_date'] = df['card_activation_date'].dt.tz_localize('UTC')
                
                # Calculate activation delay only where both dates are valid
                mask = df['card_activation_date'].notna() & df['open_date'].notna()
                df['activation_delay_days'] = pd.NA
                
                if mask.any() and hasattr(df['card_activation_date'], 'dt'):
                    df.loc[mask, 'activation_delay_days'] = (
                        df.loc[mask, 'card_activation_date'] - df.loc[mask, 'open_date']
                    ).dt.days
            except Exception as e:
                print(f"Error processing card_activation_date: {e}")
                df['activation_delay_days'] = pd.NA
        
        # Add transaction velocity features
        if 'total_transactions_2024' in df.columns and 'total_spend_2024' in df.columns:
            df['avg_transaction_size_2024'] = np.where(
                df['total_transactions_2024'] > 0,
                df['total_spend_2024'] / df['total_transactions_2024'],
                0
            )
        
        return df

    df_merged = prepare_numeric_features(df_merged)

    # Aggregate account-level data to user level.
    # Start with columns that should always exist
    base_agg_dict = {}

    # Add transaction columns if they exist
    transaction_cols = ["total_spend_2024", "total_transactions_2024", 
                       "total_spend_2025YTD", "total_transactions_2025YTD"]
    for col in transaction_cols:
        if col in df_merged.columns:
            base_agg_dict[col] = "sum"
    
    # Add quarterly spend columns
    quarterly_cols = [
        "spend_2024_q1", "transactions_2024_q1", 
        "spend_2024_q2", "transactions_2024_q2",
        "spend_2024_q3", "transactions_2024_q3",
        "spend_2024_q4", "transactions_2024_q4"
    ]
    for col in quarterly_cols:
        if col in df_merged.columns:
            base_agg_dict[col] = "sum"

    # Add financial columns if they exist
    financial_cols = ["credit_line", "current_balance"]
    for col in financial_cols:
        if col in df_merged.columns:
            base_agg_dict[col] = "sum"

    # Add score columns if they exist
    score_cols = {"behavior_score": "min", "credit_score": "min"}
    for col, agg_func in score_cols.items():
        if col in df_merged.columns:
            base_agg_dict[col] = agg_func

    # Add binary flag columns if they exist
    flag_cols = ["delinquency_12mo", "delinquency_24mo", "sent_to_collection", 
                "ever_overlimit", "has_unactivated_card", "uses_ebill", 
                "is_high_income", "high_spender", "has_pscc_account", 
                "has_external_status", "has_fraud"]
    for col in flag_cols:
        if col in df_merged.columns:
            base_agg_dict[col] = "max"

    # Add fraud transaction features if they exist
    fraud_transaction_sum_cols = ["fraud_transaction_count", "fraud_transaction_sum", "fraud_case_count"]
    fraud_transaction_max_cols = ["fraud_transaction_max", "fraud_transaction_min"]
    fraud_transaction_mean_cols = ["fraud_transaction_avg"]
    
    for col in fraud_transaction_sum_cols:
        if col in df_merged.columns:
            base_agg_dict[col] = "sum"
    
    for col in fraud_transaction_max_cols:
        if col in df_merged.columns:
            base_agg_dict[col] = "max"
    
    for col in fraud_transaction_mean_cols:
        if col in df_merged.columns:
            base_agg_dict[col] = "mean"
            
    # Add payment history metrics if they exist
    payment_metrics = ['delinquency_count', 'max_delinquency', 'zero_balance_months', 
                      'credit_balance_months', 'normal_months']
    for period in ['1_12', '13_24']:
        for metric in payment_metrics:
            col = f'payment_hist_{period}_{metric}'
            if col in df_merged.columns:
                base_agg_dict[col] = 'max'

    # Add derived feature columns if they exist
    derived_cols = {"account_age_days": "mean", "activation_delay_days": "mean", 
                   "avg_transaction_size_2024": "mean"}
    for col, agg_func in derived_cols.items():
        if col in df_merged.columns:
            base_agg_dict[col] = agg_func

    # Add active account indicator
    df_merged["is_active_account"] = 1
    if "closed_date" in df_merged.columns:
        df_merged["is_active_account"] = df_merged["closed_date"].isna().astype("Int64")
    base_agg_dict["is_active_account"] = "sum"

    # Now use the filtered aggregation dictionary
    if len(base_agg_dict) > 0:
        df_user = df_merged.groupby("user_id", as_index=False).agg(base_agg_dict)
    else:
        # If no aggregation columns, create minimal dataframe with user_id
        df_user = df_merged[["user_id"]].drop_duplicates()
    
    # Merge with the correct account counts that we calculated earlier
    df_user = pd.merge(df_user, unique_account_counts, on="user_id", how="left")
    
    # Rename active account count
    if "is_active_account" in df_user.columns:
        df_user.rename(columns={"is_active_account": "active_account_count"}, inplace=True)
        df_user["user_active_flag"] = (df_user["active_account_count"] > 0).astype("Int64")
    
    # Calculate utilization percentage
    if "credit_line" in df_user.columns and "current_balance" in df_user.columns:
        df_user["utilization_pct"] = np.where(df_user["credit_line"] > 0,
                                          (df_user["current_balance"] / df_user["credit_line"]) * 100.0,
                                          np.nan)
    
    # Calculate average monthly spend
    if "total_spend_2024" in df_user.columns:
        df_user["avg_monthly_spend_2024"] = df_user["total_spend_2024"] / 12.0
    
    def prepare_for_xgboost(df):
        # Replace infinities
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Fill NAs with meaningful values
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            if col.startswith(('total_', 'count_', 'num_')) or col.endswith(('_count', '_flag')):
                df[col] = df[col].fillna(0)
            else:
                if df[col].count() > 0:  # Only apply if there are non-NA values
                    df[col] = df[col].fillna(df[col].median())
        
        # Scale certain features to be more suitable for XGBoost
        if 'credit_line' in df.columns:
            df['credit_line'] = df['credit_line'] / 1000  # Convert to thousands
        if 'current_balance' in df.columns:
            df['current_balance'] = df['current_balance'] / 1000  # Convert to thousands
        
        return df

    df_user = prepare_for_xgboost(df_user)
    
    # Print verification of account count accuracy
    print(f"Original account count per user (mean): {unique_account_counts['num_accounts'].mean():.2f}")
    print(f"Final account count per user (mean): {df_user['num_accounts'].mean():.2f}")

    return df_user

def compare_fraud_accounts():
    """
    Generate a comparison report between accounts with fraud flags and those without.
    This helps identify differences in behavior and characteristics.
    """
    # Load the dataset
    df_account_dim = pd.read_csv("data/account_dim_20250325.csv")
    df_syf_id = pd.read_csv("data/syf_id_20250325.csv")
    df_rams = pd.read_csv("data/rams_batch_cur_20250325.csv")
    df_fraud_case = pd.read_csv("data/fraud_claim_case_20250325.csv")
    df_fraud_tran = pd.read_csv("data/fraud_claim_tran_20250325.csv")
    
    # Prepare the dataset
    df_syf_id.rename(columns={"account_nbr_pty": "current_account_nbr", "ds_id": "user_id"}, inplace=True)
    df_merged = pd.merge(df_account_dim, df_syf_id[["current_account_nbr", "user_id"]], 
                         how="inner", on="current_account_nbr")
    
    # Flag fraud accounts
    df_fraud_case["has_fraud"] = 1
    df_fraud_accounts = df_fraud_case["current_account_nbr"].unique()
    
    # Create fraud flag in the merged dataset
    df_merged["has_fraud"] = df_merged["current_account_nbr"].isin(df_fraud_accounts).astype(int)
    
    # Merge RAMS data for financial metrics
    df_rams.rename(columns={
        "cu_account_nbr": "current_account_nbr",
        "cu_crd_line": "credit_line",
        "cu_cur_balance": "current_balance",
        "cu_bhv_scr": "behavior_score",
        "cu_crd_bureau_scr": "credit_score"
    }, inplace=True)
    df_merged = pd.merge(df_merged, df_rams, how="left", on="current_account_nbr")
    
    # Convert dates for account age calculation
    df_merged['open_date'] = pd.to_datetime(df_merged['open_date'], errors='coerce')
    reference_date = pd.Timestamp('2025-03-25')
    df_merged['account_age_days'] = (reference_date - df_merged['open_date']).dt.days
    
    # Generate comparison report
    print("\n===== FRAUD vs NON-FRAUD ACCOUNT COMPARISON =====\n")
    
    # Count statistics
    fraud_count = df_merged["has_fraud"].sum()
    total_count = len(df_merged)
    fraud_pct = (fraud_count / total_count) * 100
    
    print(f"Total accounts: {total_count}")
    print(f"Accounts with fraud: {fraud_count} ({fraud_pct:.2f}%)")
    print(f"Accounts without fraud: {total_count - fraud_count} ({100 - fraud_pct:.2f}%)\n")
    
    # Compare key metrics
    metrics = ["credit_line", "current_balance", "behavior_score", "credit_score", "account_age_days"]
    
    print("Key metrics comparison (mean values):")
    print("-" * 50)
    print(f"{'Metric':<20} {'Fraud':>15} {'Non-Fraud':>15}")
    print("-" * 50)
    
    for metric in metrics:
        if metric in df_merged.columns:
            fraud_mean = df_merged[df_merged["has_fraud"] == 1][metric].mean()
            non_fraud_mean = df_merged[df_merged["has_fraud"] == 0][metric].mean()
            print(f"{metric:<20} {fraud_mean:>15.2f} {non_fraud_mean:>15.2f}")
    
    print("\n")
    
    # Calculate utilization for both groups
    df_merged["utilization_pct"] = np.where(df_merged["credit_line"] > 0,
                                           (df_merged["current_balance"] / df_merged["credit_line"]) * 100.0,
                                           np.nan)
    
    fraud_util = df_merged[df_merged["has_fraud"] == 1]["utilization_pct"].mean()
    non_fraud_util = df_merged[df_merged["has_fraud"] == 0]["utilization_pct"].mean()
    
    print(f"Credit utilization (%):")
    print(f"Fraud accounts: {fraud_util:.2f}%")
    print(f"Non-fraud accounts: {non_fraud_util:.2f}%\n")
    
    # Additional fraud transaction statistics
    if "transaction_am" in df_fraud_tran.columns:
        df_fraud_tran["transaction_am"] = pd.to_numeric(df_fraud_tran["transaction_am"], errors="coerce")
        
        # Total fraud amount
        total_fraud_amount = df_fraud_tran["transaction_am"].sum()
        avg_fraud_amount = df_fraud_tran["transaction_am"].mean()
        max_fraud_amount = df_fraud_tran["transaction_am"].max()
        
        # Transactions per account
        trans_per_account = df_fraud_tran.groupby("current_account_nbr").size().mean()
        
        print("Fraud transaction statistics:")
        print(f"Total fraud amount: ${total_fraud_amount:.2f}")
        print(f"Average fraud transaction amount: ${avg_fraud_amount:.2f}")
        print(f"Maximum fraud transaction amount: ${max_fraud_amount:.2f}")
        print(f"Average transactions per fraudulent account: {trans_per_account:.2f}")
    
    return df_merged

if __name__ == "__main__":
    df_final = clean_and_aggregate_data()
    print("Final user-level dataset shape:", df_final.shape)
    print(df_final.head(10))
    output_path = "exploratory_data_analysis/master_user_dataset.csv"
    df_final.to_csv(output_path, index=False)
    print(f"Master user dataset written to: {output_path}")
    
    # Run the fraud comparison report
    # Uncomment the line below to generate the report
    # df_fraud_comparison = compare_fraud_accounts()