import pandas as pd
import numpy as np

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('exploratory_data_analysis/master_user_dataset_with_predictions.csv')

# Overall credit line statistics
print("\nCREDIT LINE STATISTICS (before increases):")
stats = df['credit_line'].describe()
print(f"Count: {stats['count']:,.0f}")
print(f"Mean: ${stats['mean']*1000:,.2f}")
print(f"Std Dev: ${stats['std']*1000:,.2f}")
print(f"Min: ${stats['min']*1000:,.2f}")
print(f"25th Percentile: ${stats['25%']*1000:,.2f}")
print(f"Median: ${stats['50%']*1000:,.2f}")
print(f"75th Percentile: ${stats['75%']*1000:,.2f}")
print(f"Max: ${stats['max']*1000:,.2f}")
print(f"Total Credit Extended: ${df['credit_line'].sum()*1000:,.2f}")

# Credit line by segment
print("\nAVERAGE CREDIT LINE BY SEGMENT:")
for segment in sorted(df['segment_label'].unique()):
    segment_df = df[df["segment_label"] == segment]
    print(f"Segment {segment}: ${segment_df['credit_line'].mean()*1000:,.2f} (count: {len(segment_df):,})")

# Credit line by income level
print("\nAVERAGE CREDIT LINE BY INCOME LEVEL:")
high_income = df[df["is_high_income"] == 1]
reg_income = df[df["is_high_income"] == 0]
print(f"High Income: ${high_income['credit_line'].mean()*1000:,.2f} (count: {len(high_income):,})")
print(f"Regular Income: ${reg_income['credit_line'].mean()*1000:,.2f} (count: {len(reg_income):,})")

# Credit utilization statistics
if 'utilization_pct' in df.columns:
    print("\nCREDIT UTILIZATION STATISTICS:")
    util_stats = df['utilization_pct'].describe()
    print(f"Mean: {util_stats['mean']:.2f}%")
    print(f"Median: {util_stats['50%']:.2f}%")
    
    print("\nAVERAGE UTILIZATION BY SEGMENT:")
    for segment in sorted(df['segment_label'].unique()):
        segment_df = df[df["segment_label"] == segment]
        print(f"Segment {segment}: {segment_df['utilization_pct'].mean():.2f}%")

# Compare with recommended CLI amounts
if 'recommended_cli_amount' in df.columns:
    print("\nCOMPARISON WITH RECOMMENDED CLI AMOUNTS:")
    eligible_df = df[df['segment_label'].isin([0, 1])]
    current_credit = eligible_df['credit_line'].sum() * 1000
    recommended_cli = eligible_df['recommended_cli_amount'].sum()
    print(f"Current Total Credit Line (eligible accounts): ${current_credit:,.2f}")
    print(f"Total Recommended CLI: ${recommended_cli:,.2f}")
    print(f"Percentage Increase: {(recommended_cli/current_credit)*100:.2f}%")
    
    print("\nAVERAGE CLI AS PERCENTAGE OF CURRENT CREDIT LINE:")
    for segment in [0, 1]:
        segment_df = eligible_df[eligible_df["segment_label"] == segment]
        avg_credit = segment_df['credit_line'].mean() * 1000
        avg_cli = segment_df['recommended_cli_amount'].mean()
        print(f"Segment {segment}: {(avg_cli/avg_credit)*100:.2f}%") 