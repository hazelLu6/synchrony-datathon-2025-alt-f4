import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

# Ensure visualizations directory exists
os.makedirs('visualizations', exist_ok=True)

# Set dark style
plt.style.use('dark_background')
sns.set_context("talk")

# Custom dark mode colors
DARK_BG = '#121212'
TEXT_COLOR = '#FFFFFF'
GRID_COLOR = '#333333'
COLORS = ['#ff9d9d', '#9dff9d', '#9d9dff', '#ffff9d', '#ff9dff', '#9dffff']
CMAP = 'viridis'

print("Generating dark mode visualizations...")

# Model 1: Q4 Spend Prediction
# ----------------------------

# 1. Feature Importance Visualization
print("Creating Q4 Spend Feature Importance chart...")
features = ['holiday_spending_power', 'avg_monthly_spend_2024', 
            'total_spend_2024', 'spend_2024_q3', 'credit_line', 
            'current_balance', 'is_high_income', 'utilization_pct']
importance = [0.603, 0.154, 0.114, 0.068, 0.028, 0.016, 0.009, 0.008]

# Create DataFrame
feat_imp_df = pd.DataFrame({'Feature': features, 'Importance': importance})
feat_imp_df = feat_imp_df.sort_values('Importance', ascending=False)

# Create visualization
plt.figure(figsize=(10, 6), facecolor=DARK_BG)
ax = sns.barplot(x='Importance', y='Feature', data=feat_imp_df, 
                palette='magma')

# Add value labels
for i, v in enumerate(feat_imp_df['Importance']):
    ax.text(v + 0.01, i, f"{v:.3f}", va='center', color=TEXT_COLOR)

plt.title('Feature Importance: Q4 Spend Prediction Model', fontsize=16, color=TEXT_COLOR)
plt.xlabel('Relative Importance', fontsize=12, color=TEXT_COLOR)
plt.ylabel('', color=TEXT_COLOR)
ax.tick_params(colors=TEXT_COLOR)
plt.tight_layout()
plt.savefig('visualizations/q4_spend_feature_importance_dark.png', dpi=300, bbox_inches='tight', facecolor=DARK_BG)
plt.close()

# 2. Quarterly Spending Pattern Visualization
print("Creating Quarterly Spending Pattern chart...")
quarters = ['Q1', 'Q2', 'Q3', 'Q4 (Predicted)']
avg_spend = [2850, 3100, 3350, 4950]  # Example values
holiday_effect = [2850, 3100, 3350, 3450]  # Without holiday effect

plt.figure(figsize=(10, 6), facecolor=DARK_BG)
x = np.arange(len(quarters))
width = 0.35

plt.bar(x - width/2, avg_spend, width, label='With Holiday Effect', color='#ff6666')
plt.bar(x + width/2, holiday_effect, width, label='Projected Without Holiday', color='#6699ff')

# Add 25% increase arrow
plt.annotate('', xy=(3.1, 4200), xytext=(3.1, 3500),
            arrowprops=dict(facecolor='#ff9900', shrink=0.05, width=2))
plt.text(3.2, 3850, '+25%\nHoliday\nEffect', color='#ff9900', fontweight='bold')

plt.title('Quarterly Spending Patterns - 2024', fontsize=16, color=TEXT_COLOR)
plt.ylabel('Average Customer Spend ($)', fontsize=12, color=TEXT_COLOR)
plt.xticks(x, quarters, fontsize=12, color=TEXT_COLOR)
plt.yticks(color=TEXT_COLOR)
plt.legend(loc='upper left', facecolor=DARK_BG, labelcolor=TEXT_COLOR)
plt.grid(axis='y', linestyle='--', alpha=0.3, color=GRID_COLOR)
plt.tight_layout()
plt.savefig('visualizations/quarterly_spending_pattern_dark.png', dpi=300, bbox_inches='tight', facecolor=DARK_BG)
plt.close()

# Model 2: Customer Segmentation
# -----------------------------

# 1. Segment Distribution Pie Chart
print("Creating Segment Distribution chart...")
segments = ['Eligible-No Risk (0)', 'Eligible-At Risk (1)', 
            'No Increase Needed (2)', 'High Risk (3)']
sizes = [15, 25, 45, 15]  # Percentages
colors = ['#50fa7b', '#ffb86c', '#8be9fd', '#ff5555']
explode = (0.1, 0.05, 0, 0.05)  # Explode 1st slice

plt.figure(figsize=(10, 8), facecolor=DARK_BG)
wedges, texts, autotexts = plt.pie(sizes, explode=explode, labels=segments, colors=colors,
                                   autopct='%1.1f%%', shadow=True, startangle=90,
                                   textprops={'color': TEXT_COLOR})
for autotext in autotexts:
    autotext.set_color(DARK_BG)
    autotext.set_fontweight('bold')

plt.axis('equal')
plt.title('Customer Segmentation Distribution', fontsize=16, color=TEXT_COLOR)
plt.tight_layout()
plt.savefig('visualizations/segment_distribution_dark.png', dpi=300, bbox_inches='tight', facecolor=DARK_BG)
plt.close()

# 2. Segment Characteristic Comparison
print("Creating Segment Characteristics Radar chart...")
# Prepare data for radar chart
categories = ['Credit Score', 'Credit Line', 'Spending', 'Low Utilization', 'Low Delinquency']
segment0 = [0.9, 0.7, 0.8, 0.5, 0.95]  # Normalized values 
segment1 = [0.7, 0.6, 0.9, 0.3, 0.8]
segment2 = [0.8, 0.5, 0.3, 0.8, 0.95]
segment3 = [0.3, 0.4, 0.4, 0.2, 0.4]

# Set up radar chart
fig = plt.figure(figsize=(10, 8), facecolor=DARK_BG)
ax = fig.add_subplot(111, polar=True)
ax.set_facecolor(DARK_BG)

# Number of variables
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]  # Close the loop

# Add data
ax.plot(angles, segment0 + segment0[:1], 'o-', linewidth=2, label='Segment 0: Eligible-No Risk', color='#50fa7b')
ax.plot(angles, segment1 + segment1[:1], 'o-', linewidth=2, label='Segment 1: Eligible-At Risk', color='#ffb86c')
ax.plot(angles, segment2 + segment2[:1], 'o-', linewidth=2, label='Segment 2: No Increase Needed', color='#8be9fd')
ax.plot(angles, segment3 + segment3[:1], 'o-', linewidth=2, label='Segment 3: High Risk', color='#ff5555')
ax.fill(angles, segment0 + segment0[:1], alpha=0.1, color='#50fa7b')
ax.fill(angles, segment1 + segment1[:1], alpha=0.1, color='#ffb86c')
ax.fill(angles, segment2 + segment2[:1], alpha=0.1, color='#8be9fd')
ax.fill(angles, segment3 + segment3[:1], alpha=0.1, color='#ff5555')

# Add category labels and adjust appearance
plt.xticks(angles[:-1], categories, color=TEXT_COLOR)
plt.yticks(color=TEXT_COLOR, alpha=0)
ax.tick_params(axis='both', colors=TEXT_COLOR)
ax.spines['polar'].set_color(GRID_COLOR)
ax.grid(color=GRID_COLOR, alpha=0.3)

# Add title and legend
plt.title('Segment Characteristics Comparison', fontsize=16, color=TEXT_COLOR, pad=20)
legend = plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), facecolor=DARK_BG)
for text in legend.get_texts():
    text.set_color(TEXT_COLOR)

plt.tight_layout()
plt.savefig('visualizations/segment_characteristics_dark.png', dpi=300, bbox_inches='tight', facecolor=DARK_BG)
plt.close()

# Model 3: Risk Flagging
# ---------------------

# 1. ROC Curve Visualization
print("Creating ROC Curve chart...")
# Sample data for ROC curve
fpr = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
tpr = [0, 0.4, 0.55, 0.7, 0.8, 0.82, 0.85, 0.89, 0.91, 0.95, 0.98, 1.0]
auc = 0.647  # Area under curve value

plt.figure(figsize=(8, 8), facecolor=DARK_BG)
plt.plot(fpr, tpr, color='#ff7f0e', lw=2, label=f'ROC Curve (AUC = {auc:.3f})')
plt.plot([0, 1], [0, 1], color='#7f7f7f', lw=2, linestyle='--', label='Random Classifier')

# Highlight operating point
op_fpr, op_tpr = 0.2, 0.7  # Example operating point
plt.plot(op_fpr, op_tpr, 'ro', markersize=10)
plt.annotate('Operating Point\nPrecision: 0.22, Recall: 0.41',
            xy=(op_fpr, op_tpr), xytext=(op_fpr+0.15, op_tpr-0.15),
            arrowprops=dict(facecolor='#ffffff', shrink=0.05, alpha=0.5),
            color=TEXT_COLOR)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12, color=TEXT_COLOR)
plt.ylabel('True Positive Rate', fontsize=12, color=TEXT_COLOR)
plt.title('ROC Curve: Risk Flagging Model', fontsize=16, color=TEXT_COLOR)
plt.legend(loc="lower right", facecolor=DARK_BG, labelcolor=TEXT_COLOR)
plt.grid(alpha=0.2, color=GRID_COLOR)
plt.xticks(color=TEXT_COLOR)
plt.yticks(color=TEXT_COLOR)
plt.tight_layout()
plt.savefig('visualizations/risk_roc_curve_dark.png', dpi=300, bbox_inches='tight', facecolor=DARK_BG)
plt.close()

# 2. Risk Factor Importance
print("Creating Risk Factor Importance chart...")
# Sample risk factor importance data
risk_features = ['utilization_pct', 'payment_hist_13_24_max_delinquency', 
                 'credit_score', 'current_balance', 'delinquency_12mo',
                 'delinquency_24mo', 'payment_hist_1_12_max_delinquency',
                 'credit_line', 'total_spend_2025YTD', 'total_spend_2024']
risk_importance = [0.335, 0.109, 0.094, 0.085, 0.083, 
                   0.072, 0.064, 0.054, 0.044, 0.031]

# Create DataFrame
risk_imp_df = pd.DataFrame({'Risk Factor': risk_features, 'Importance': risk_importance})
risk_imp_df = risk_imp_df.sort_values('Importance', ascending=False)

# Create visualization
plt.figure(figsize=(10, 6), facecolor=DARK_BG)
ax = sns.barplot(x='Importance', y='Risk Factor', data=risk_imp_df, 
                palette='rocket')

# Add value labels
for i, v in enumerate(risk_imp_df['Importance']):
    ax.text(v + 0.01, i, f"{v:.3f}", va='center', color=TEXT_COLOR)

plt.title('Risk Factor Importance', fontsize=16, color=TEXT_COLOR)
plt.xlabel('Relative Importance', fontsize=12, color=TEXT_COLOR)
plt.ylabel('', color=TEXT_COLOR)
ax.tick_params(colors=TEXT_COLOR)
plt.tight_layout()
plt.savefig('visualizations/risk_factor_importance_dark.png', dpi=300, bbox_inches='tight', facecolor=DARK_BG)
plt.close()

# Model 4: CLI Recommendation
# -------------------------

# 1. CLI Amount Distribution by Segment
print("Creating CLI by Segment chart...")
# Sample CLI data
segments = ['Segment 0\nEligible-No Risk', 'Segment 1\nEligible-At Risk', 
           'Segment 2\nNo Increase', 'Segment 3\nHigh Risk']
avg_cli = [9850, 5320, 0, 0]
cli_counts = [2350, 3750, 6750, 2250]  # Account counts per segment

# Create visualization
plt.figure(figsize=(12, 6), facecolor=DARK_BG)
ax = plt.bar(segments, avg_cli, width=0.6, color=['#50fa7b', '#ffb86c', '#6272a4', '#ff5555'])

# Add count annotations
for i, v in enumerate(avg_cli):
    if v > 0:
        plt.text(i, v + 500, f"${v:,.0f}", ha='center', fontweight='bold', color=TEXT_COLOR)
    plt.text(i, 200, f"{cli_counts[i]:,} accounts", ha='center', color=DARK_BG, fontweight='bold')

plt.title('Average Credit Line Increase by Segment', fontsize=16, color=TEXT_COLOR)
plt.ylabel('Average CLI Amount ($)', fontsize=12, color=TEXT_COLOR)
plt.ylim(0, max(avg_cli) * 1.2)
plt.grid(axis='y', alpha=0.2, color=GRID_COLOR)
plt.xticks(color=TEXT_COLOR)
plt.yticks(color=TEXT_COLOR)
plt.tight_layout()
plt.savefig('visualizations/cli_by_segment_dark.png', dpi=300, bbox_inches='tight', facecolor=DARK_BG)
plt.close()

# 2. CLI Recommendation by Risk and Income
print("Creating CLI by Risk and Income chart...")
# Sample data grid for CLI recommendations
risk_levels = ['Very Low', 'Low', 'Moderate', 'High', 'Very High']
high_income = [10000, 7500, 5000, 0, 0]  # CLI amounts for high income
regular_income = [7500, 5000, 2500, 0, 0]  # CLI amounts for regular income

plt.figure(figsize=(12, 6), facecolor=DARK_BG)

x = np.arange(len(risk_levels))
width = 0.35

plt.bar(x - width/2, high_income, width, label='High Income', color='#8be9fd')
plt.bar(x + width/2, regular_income, width, label='Regular Income', color='#bd93f9')

plt.ylabel('Recommended CLI Amount ($)', fontsize=14, color=TEXT_COLOR)
plt.title('CLI Recommendations by Risk Level and Income', fontsize=16, color=TEXT_COLOR)
plt.xticks(x, risk_levels, fontsize=12, color=TEXT_COLOR)
plt.yticks(color=TEXT_COLOR)
legend = plt.legend(facecolor=DARK_BG)
for text in legend.get_texts():
    text.set_color(TEXT_COLOR)

# Add annotations for key insights
plt.annotate('No CLI for High Risk\nregardless of income', 
            xy=(3.5, 0), xytext=(3.5, 3000),
            arrowprops=dict(facecolor='#ff5555', shrink=0.05),
            ha='center', fontsize=10, color=TEXT_COLOR)

plt.annotate('Higher CLI for high income\nacross all eligible risk levels', 
            xy=(1, high_income[1]), xytext=(1, high_income[1] + 2000),
            arrowprops=dict(facecolor='#8be9fd', shrink=0.05),
            ha='center', fontsize=10, color=TEXT_COLOR)

plt.grid(axis='y', linestyle='--', alpha=0.2, color=GRID_COLOR)
plt.tight_layout()
plt.savefig('visualizations/cli_by_risk_income_dark.png', dpi=300, bbox_inches='tight', facecolor=DARK_BG)
plt.close()

# Integrated Model Pipeline Visualization
# -------------------------------------
print("Creating Integrated Model Pipeline diagram...")
# Create figure and axis
fig, ax = plt.subplots(figsize=(12, 8), facecolor=DARK_BG)

# Set background
ax.set_facecolor(DARK_BG)
ax.grid(False)
ax.set_xlim(0, 10)
ax.set_ylim(0, 7)
ax.axis('off')

# Draw boxes for each model
model1 = plt.Rectangle((1, 5.5), 3, 1, facecolor='#6272a4', alpha=0.8, edgecolor='#f8f8f2')
model2 = plt.Rectangle((1, 4), 3, 1, facecolor='#50fa7b', alpha=0.8, edgecolor='#f8f8f2')
model3 = plt.Rectangle((1, 2.5), 3, 1, facecolor='#ff5555', alpha=0.8, edgecolor='#f8f8f2')
model4 = plt.Rectangle((6, 4), 3, 1, facecolor='#ffb86c', alpha=0.8, edgecolor='#f8f8f2')

# Add boxes to plot
ax.add_patch(model1)
ax.add_patch(model2)
ax.add_patch(model3)
ax.add_patch(model4)

# Add text for model descriptions
ax.text(2.5, 6, "Model 1: Q4 Spend Prediction", ha='center', fontsize=12, fontweight='bold', color=TEXT_COLOR)
ax.text(2.5, 5.7, "XGBoost Regression", ha='center', fontsize=10, color=TEXT_COLOR)
ax.text(2.5, 4.5, "Model 2: Customer Segmentation", ha='center', fontsize=12, fontweight='bold', color=TEXT_COLOR)
ax.text(2.5, 4.2, "XGBoost Multi-class Classification", ha='center', fontsize=10, color=TEXT_COLOR)
ax.text(2.5, 3, "Model 3: Risk Flagging", ha='center', fontsize=12, fontweight='bold', color=TEXT_COLOR)
ax.text(2.5, 2.7, "XGBoost Binary Classification", ha='center', fontsize=10, color=TEXT_COLOR)
ax.text(7.5, 4.5, "Model 4: CLI Recommendation", ha='center', fontsize=12, fontweight='bold', color=TEXT_COLOR)
ax.text(7.5, 4.2, "XGBoost Regression", ha='center', fontsize=10, color=TEXT_COLOR)

# Add connectors (arrows)
arrow1 = plt.arrow(4.1, 6, 1.7, -1.5, head_width=0.2, head_length=0.2, 
                   fc='#f8f8f2', ec='#f8f8f2', length_includes_head=True, alpha=0.8)
arrow2 = plt.arrow(4.1, 4.5, 1.8, -0.1, head_width=0.2, head_length=0.2, 
                   fc='#f8f8f2', ec='#f8f8f2', length_includes_head=True, alpha=0.8)
arrow3 = plt.arrow(4.1, 3, 1.7, 1, head_width=0.2, head_length=0.2, 
                   fc='#f8f8f2', ec='#f8f8f2', length_includes_head=True, alpha=0.8)

# Add input/output labels
ax.text(1, 1.5, "MODEL INPUTS:", fontsize=10, fontweight='bold', color=TEXT_COLOR)
ax.text(1, 1.2, "• Transaction history", fontsize=9, color=TEXT_COLOR)
ax.text(1, 0.9, "• Account demographics", fontsize=9, color=TEXT_COLOR)
ax.text(1, 0.6, "• Credit scores & utilization", fontsize=9, color=TEXT_COLOR)
ax.text(1, 0.3, "• Payment history & fraud flags", fontsize=9, color=TEXT_COLOR)

ax.text(6, 3, "FINAL OUTPUT:", fontsize=10, fontweight='bold', color=TEXT_COLOR)
ax.text(6, 2.7, "• Personalized CLI amount", fontsize=9, color=TEXT_COLOR)
ax.text(6, 2.4, "• Segmentation label", fontsize=9, color=TEXT_COLOR)
ax.text(6, 2.1, "• Q4 spend prediction", fontsize=9, color=TEXT_COLOR)
ax.text(6, 1.8, "• Risk probability", fontsize=9, color=TEXT_COLOR)

# Add title
plt.suptitle("Integrated Model Pipeline for CLI Recommendation System", fontsize=16, y=0.95, color=TEXT_COLOR)

plt.tight_layout()
plt.savefig('visualizations/integrated_model_pipeline_dark.png', dpi=300, bbox_inches='tight', facecolor=DARK_BG)
plt.close()

print("All dark mode visualizations have been generated in the 'visualizations' folder!") 