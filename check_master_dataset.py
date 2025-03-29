import pandas as pd
import numpy as np

# Load the master dataset
print("Loading master dataset...")
df = pd.read_csv("exploratory_data_analysis/master_user_dataset.csv")

# Analyze the num_accounts field
print("\n=== Analysis of num_accounts in Master Dataset ===")
print(f"Total users in master dataset: {len(df)}")
print(f"Mean num_accounts per user: {df['num_accounts'].mean():.2f}")
print(f"Median num_accounts per user: {df['num_accounts'].median():.2f}")
print(f"Min num_accounts per user: {df['num_accounts'].min()}")
print(f"Max num_accounts per user: {df['num_accounts'].max()}")

# Calculate distribution
print("\n=== Frequency Distribution of num_accounts ===")
value_counts = df['num_accounts'].value_counts().sort_index()
for accounts, count in value_counts.items():
    percentage = (count / len(df)) * 100
    print(f"{accounts} accounts: {count} users ({percentage:.2f}%)")

# Check users with high account counts
high_accounts = df[df['num_accounts'] > 3]
if len(high_accounts) > 0:
    print("\n=== Sample of Users with High Account Counts ===")
    for idx, row in high_accounts.head(10).iterrows():
        print(f"User with {row['num_accounts']} accounts, credit line: {row['credit_line']:.2f}, balance: {row['current_balance']:.2f}")
else:
    print("\nNo users with more than 3 accounts found.")

# Compare with original mapping data
print("\n=== Comparing with Original Mapping Data ===")
df_syf_id = pd.read_csv("data/syf_id_20250325.csv")
df_syf_id.rename(columns={"account_nbr_pty": "current_account_nbr", "ds_id": "user_id"}, inplace=True)
accounts_per_user_original = df_syf_id.groupby("user_id")["current_account_nbr"].count()

print(f"Original data - Mean accounts per user: {accounts_per_user_original.mean():.2f}")
print(f"Original data - Max accounts per user: {accounts_per_user_original.max()}")
print(f"Master dataset - Mean accounts per user: {df['num_accounts'].mean():.2f}")
print(f"Master dataset - Max accounts per user: {df['num_accounts'].max()}")

if df['num_accounts'].max() > accounts_per_user_original.max():
    print("\n*** DISCREPANCY DETECTED ***")
    print("The master dataset shows higher account counts than the original mapping data.") 