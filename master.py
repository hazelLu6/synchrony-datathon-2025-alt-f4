import pandas as pd
import numpy as np

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
    df_merged = pd.merge(df_account_dim, df_syf_id[["current_account_nbr", "user_id"]],
                         how="inner", on="current_account_nbr")
    print("After merging account_dim and syf_id:", df_merged.shape)
    
    # Merge statement fact (if available) on current_account_nbr
    if "return_check_cnt_total" in df_statement.columns:
        df_statement_agg = df_statement.groupby("current_account_nbr", as_index=False).agg({
            "return_check_cnt_total": "sum"
        })
        df_merged = pd.merge(df_merged, df_statement_agg, how="left", on="current_account_nbr")
    
    # Combine transaction facts and convert dates (do not drop rows with NaT)
    df_trans_all = pd.concat([df_trans, df_wrld_trans], ignore_index=True)
    df_trans_all["transaction_date"] = pd.to_datetime(df_trans_all["transaction_date"], errors="coerce")
    
    def is_2024(date_val):
        return (date_val >= pd.Timestamp("2024-01-01")) & (date_val <= pd.Timestamp("2024-12-31"))
    def is_2025_ytd(date_val):
        return (date_val >= pd.Timestamp("2025-01-01")) & (date_val <= pd.Timestamp("2025-03-24"))
    
    df_trans_all["in_2024"] = is_2024(df_trans_all["transaction_date"])
    df_trans_all["in_2025_ytd"] = is_2025_ytd(df_trans_all["transaction_date"])
    df_trans_all["is_purchase"] = df_trans_all["transaction_type"].eq("SALE")
    df_trans_all["is_return"] = df_trans_all["transaction_type"].eq("RETURN")
    df_trans_all["signed_amount"] = np.where(df_trans_all["is_return"],
                                             -df_trans_all["transaction_amt"],
                                             df_trans_all["transaction_amt"])
    
    trans_agg = df_trans_all.groupby("current_account_nbr").apply(lambda g: pd.Series({
        "total_spend_2024": g.loc[g["in_2024"], "signed_amount"].sum(),
        "total_transactions_2024": g.loc[g["in_2024"] & g["is_purchase"]].shape[0],
        "total_spend_2025YTD": g.loc[g["in_2025_ytd"], "signed_amount"].sum(),
        "total_transactions_2025YTD": g.loc[g["in_2025_ytd"] & g["is_purchase"]].shape[0],
    })).reset_index()
    df_merged = pd.merge(df_merged, trans_agg, how="left", on="current_account_nbr")
    
    # Merge fraud claims (using fraud_claim_case)
    df_fraud_case["has_fraud"] = 1
    df_fraud_case_agg = df_fraud_case.groupby("current_account_nbr", as_index=False)["has_fraud"].max()
    df_merged = pd.merge(df_merged, df_fraud_case_agg, how="left", on="current_account_nbr")
    
    # Merge RAMS data, renaming key fields for clarity
    df_rams.rename(columns={
        "cu_account_nbr": "current_account_nbr",
        "cu_crd_line": "credit_line",
        "cu_cur_balance": "current_balance",
        "cu_bhv_scr": "behavior_score",
        "cu_crd_bureau_scr": "credit_score"
    }, inplace=True)
    df_merged = pd.merge(df_merged, df_rams, how="left", on="current_account_nbr")
    
    # Clean payment history fields
    for field in ["payment_hist_1_12_mths", "payment_hist_13_24_mths"]:
        if field in df_merged.columns:
            df_merged[field] = df_merged[field].fillna("").str.replace(r'\\', '', regex=True).str.replace('"', '', regex=True)
    
    def has_delinquency(history_string):
        return any(ch in set("BJC1234567H") for ch in history_string)
    
    df_merged["delinquency_12mo"] = (df_merged["payment_hist_1_12_mths"].apply(has_delinquency)
                                     .astype("Int64") if "payment_hist_1_12_mths" in df_merged.columns else np.nan)
    df_merged["delinquency_24mo"] = (df_merged["payment_hist_13_24_mths"].apply(has_delinquency)
                                     .astype("Int64") if "payment_hist_13_24_mths" in df_merged.columns else np.nan)
    
    # Create additional flags
    df_merged["sent_to_collection"] = (df_merged["date_in_collection"].notna().astype("Int64")
                                         if "date_in_collection" in df_merged.columns else np.nan)
    df_merged["ever_overlimit"] = (df_merged["overlimit_type_flag"].apply(lambda x: 1 if pd.notna(x) and x > 0 else 0)
                                   .astype("Int64") if "overlimit_type_flag" in df_merged.columns else np.nan)
    df_merged["uses_ebill"] = (df_merged["ebill_ind"].apply(lambda x: 1 if pd.notna(x) and str(x).strip() in ["B", "E", "L"] else 0)
                               .astype("Int64") if "ebill_ind" in df_merged.columns else np.nan)
    df_merged["has_unactivated_card"] = (df_merged["card_activation_flag"].apply(lambda x: 1 if pd.notna(x) and x in [7, 8] else 0)
                                          .astype("Int64") if "card_activation_flag" in df_merged.columns else np.nan)
    df_merged["is_employee"] = (df_merged["employee_code"].apply(lambda x: 1 if pd.notna(x) and str(x).upper() in ["1", "Y"] else 0)
                                 .astype("Int64") if "employee_code" in df_merged.columns else np.nan)
    df_merged["high_spender"] = (df_merged["employee_code"].apply(lambda x: 1 if pd.notna(x) and str(x).upper() == "H" else 0)
                                  .astype("Int64") if "employee_code" in df_merged.columns else np.nan)
    df_merged["has_pscc_account"] = (df_merged["pscc_ind"].apply(lambda x: 1 if pd.notna(x) and x == 1 else 0)
                                     .astype("Int64") if "pscc_ind" in df_merged.columns else np.nan)
    df_merged["has_external_status"] = (df_merged["external_status_reason_code"].apply(
        lambda x: 1 if pd.notna(x) and str(x).strip() not in ["0", ""] else 0).astype("Int64")
                                        if "external_status_reason_code" in df_merged.columns else np.nan)
    
    # Aggregate account-level data to user level.
    agg_dict = {
        "total_spend_2024": "sum",
        "total_transactions_2024": "sum",
        "total_spend_2025YTD": "sum",
        "total_transactions_2025YTD": "sum",
        "credit_line": "sum",
        "current_balance": "sum",
        "delinquency_12mo": "max",
        "delinquency_24mo": "max",
        "sent_to_collection": "max",
        "ever_overlimit": "max",
        "has_unactivated_card": "max",
        "uses_ebill": "max",
        "is_employee": "max",
        "high_spender": "max",
        "has_pscc_account": "max",
        "has_external_status": "max",
        "has_fraud": "max"
    }
    if "behavior_score" in df_merged.columns:
        agg_dict["behavior_score"] = "min"
    if "credit_score" in df_merged.columns:
        agg_dict["credit_score"] = "min"
    agg_dict["current_account_nbr"] = "count"
    df_merged["is_active_account"] = 1
    if "closed_date" in df_merged.columns:
        df_merged["is_active_account"] = df_merged["closed_date"].isna().astype("Int64")
    agg_dict["is_active_account"] = "sum"
    
    df_user = df_merged.groupby("user_id", as_index=False).agg(agg_dict)
    df_user.rename(columns={"current_account_nbr": "num_accounts",
                            "is_active_account": "active_account_count"}, inplace=True)
    df_user["user_active_flag"] = (df_user["active_account_count"] > 0).astype("Int64")
    df_user["utilization_pct"] = np.where(df_user["credit_line"] > 0,
                                          (df_user["current_balance"] / df_user["credit_line"]) * 100.0,
                                          np.nan)
    df_user["avg_monthly_spend_2024"] = df_user["total_spend_2024"] / 12.0
    
    return df_user

if __name__ == "__main__":
    df_final = clean_and_aggregate_data()
    print("Final user-level dataset shape:", df_final.shape)
    print(df_final.head(10))
    output_path = "exploratory_data_analysis/master_user_dataset.csv"
    df_final.to_csv(output_path, index=False)
    print(f"Master user dataset written to: {output_path}")