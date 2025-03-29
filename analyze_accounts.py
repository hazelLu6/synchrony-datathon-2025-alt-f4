import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the syf_id data
df_syf_id = pd.read_csv("data/syf_id_20250325.csv")
df_syf_id.rename(columns={"account_nbr_pty": "current_account_nbr", "ds_id": "user_id"}, inplace=True)

# Calculate accounts per user
accounts_per_user = df_syf_id.groupby("user_id")["current_account_nbr"].count()

# Generate statistics
print("=== Account Distribution Statistics ===")
print(f"Total users: {len(accounts_per_user)}")
print(f"Total accounts: {len(df_syf_id)}")
print(f"Mean accounts per user: {accounts_per_user.mean():.2f}")
print(f"Median accounts per user: {accounts_per_user.median():.2f}")
print(f"Min accounts per user: {accounts_per_user.min()}")
print(f"Max accounts per user: {accounts_per_user.max()}")
print("\n=== Frequency Distribution ===")
value_counts = accounts_per_user.value_counts().sort_index()
for accounts, count in value_counts.items():
    percentage = (count / len(accounts_per_user)) * 100
    print(f"{accounts} accounts: {count} users ({percentage:.2f}%)")

# Print the top 10 users with most accounts
print("\n=== Top 10 Users with Most Accounts ===")
top_10 = accounts_per_user.sort_values(ascending=False).head(10)
for user_id, count in top_10.items():
    print(f"User {user_id}: {count} accounts") 