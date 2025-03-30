# Synchrony Credit Line Increase Project: Data Processing Framework

## Data Cleaning and Merging Logic

The project builds a comprehensive credit line increase recommendation system using data from multiple sources. Here's how the data is processed:

### Key Files and Their Main Fields

#### 1. Account Dimension (`account_dim_20250325.csv`)
- `current_account_nbr`: Primary account identifier used across datasets
- `open_date`: Account opening date, used for calculating account age
- `card_activation_date`: Used to calculate activation delay (time between opening and activation)
- `closed_date`: Used to determine if account is still active
- `is_high_income`: Derived from `employee_code` field (H=1, Y or NaN=0)
- `payment_hist_1_12_mths`/`payment_hist_13_24_mths`: Payment history strings parsed into:
  - `delinquency_count`: Number of delinquent months
  - `max_delinquency`: Severity of worst delinquency
  - `zero_balance_months`: Months with zero balance
  - `credit_balance_months`: Months with credit balance
  - `normal_months`: Months with normal payment status

#### 2. User Mapping (`syf_id_20250325.csv`)
- `account_nbr_pty` → `current_account_nbr`: Account identifier
- `ds_id` → `user_id`: User identifier (one user can have multiple accounts)

#### 3. Transaction Data
Combined from two sources:
- Regular transactions (`transaction_fact_20250325.csv`)
- World store transactions (`wrld_stor_tran_fact_20250325.csv`)

Key fields:
- `current_account_nbr`: Account identifier
- `transaction_date`: Date of transaction (used for time-based aggregation)
- `transaction_amt`: Transaction amount
- `transaction_type`: Used to identify purchases ("SALE") vs returns ("RETURN")

Time-based transaction aggregations:
- Annual: `total_spend_2024`, `total_transactions_2024`
- Quarterly: `spend_2024_q1/q2/q3/q4`, `transactions_2024_q1/q2/q3/q4`
- YTD 2025: `total_spend_2025YTD`, `total_transactions_2025YTD`

#### 4. Fraud Data
From two sources:
- Case data (`fraud_claim_case_20250325.csv`)
- Transaction data (`fraud_claim_tran_20250325.csv`)

Key fields:
- `current_account_nbr`: Account identifier
- `has_fraud`: Binary flag (1=fraud account, 0=no fraud)
- `transaction_am` → Aggregated into:
  - `fraud_transaction_count`: Count of fraudulent transactions
  - `fraud_transaction_sum`: Total fraudulent amount
  - `fraud_transaction_avg`: Average fraudulent transaction
  - `fraud_transaction_max`/`min`: Maximum/minimum fraudulent amounts
  - `fraud_case_count`: Unique fraud cases per account

#### 5. Risk Assessment (`rams_batch_cur_20250325.csv`)
- `cu_account_nbr` → `current_account_nbr`: Account identifier
- `cu_crd_line` → `credit_line`: Credit limit (scaled to thousands for modeling)
- `cu_cur_balance` → `current_balance`: Current balance (scaled to thousands)
- `cu_bhv_scr` → `behavior_score`: Internal behavior score
- `cu_crd_bureau_scr` → `credit_score`: External credit bureau score

### Derived Features
- `account_age_days`: Days since account opening
- `activation_delay_days`: Days between account opening and card activation
- `avg_transaction_size_2024`: Average transaction amount in 2024
- `utilization_pct`: Current balance as percentage of credit line
- `avg_monthly_spend_2024`: Total 2024 spend divided by 12

### Aggregation Logic
Data is aggregated from account level to user level using specific functions:
- Sum for transaction metrics and amounts
- Min for score metrics
- Max for risk indicators and flags
- Mean for derived metrics like account age
- Counts for active accounts

### Feature Transformation
- Financial amounts (`credit_line`, `current_balance`) scaled to thousands
- Missing values imputed with:
  - Zeros for count fields
  - Medians for other numeric fields
- Infinity values replaced with NaN
- Payment history strings parsed into structured metrics

This detailed data processing pipeline creates a comprehensive user-level dataset that captures spending patterns, risk factors, account performance, and financial metrics - all essential for the credit line increase recommendation system.

## XGBoost Models

### Model 1: Q4 Spend Prediction

Model 1 is designed to predict customer spending for Q4 2024, which is critical for determining appropriate credit line increases. The model uses XGBoost regression to forecast spending based on historical patterns, customer behavior, and macroeconomic factors.

#### Validation Approach
Before building the main Q4 prediction model, a validation model is built to predict Q3 spend using only Q1 and Q2 data. This allows for validation against actual historical data rather than synthetic targets.

#### Macroeconomic Feature Engineering
The model incorporates sophisticated macroeconomic and seasonal features:

1. **Holiday Spending Multiplier**
   - Different customer segments have different holiday spending patterns
   - Four segments based on credit score and utilization
   - Multipliers range from 1.15 (subprime) to 1.45 (prime)
   - High-income customers receive an additional 15% boost

2. **Economic Outlook Factor**
   - Based on economic projections for 2025
   - Incorporates income growth rate (3.2%)
   - Adjusted for inflation rate (2.8%)
   - Consumer confidence index (0.65 on 0-1 scale)

3. **Retail Sales Seasonal Index**
   - Industry-specific Q4 seasonal patterns
   - Electronics: 65% higher sales in Q4
   - General retail: 45% higher sales in Q4
   - Online shopping: 55% higher sales in Q4
   - Grocery: 20% higher sales in Q4

4. **Income-Based Spending Elasticity**
   - Credit score used as proxy for income stability
   - Higher scores = lower elasticity (less sensitive to economic changes)
   - Normalized to 0-1 scale

5. **Composite Q4 Adjustment Factor**
   - Combines the above factors into a single multiplier
   - Formula: (Holiday Multiplier × (1 + Economic Factor) × Seasonal Index) ÷ Spending Elasticity

#### Advanced Feature Interactions
The model creates economically meaningful feature interactions:

1. **Credit Capacity**
   - Credit line × (100 - utilization percentage) / 100
   - Represents available spending power

2. **Spending Momentum**
   - Quarterly growth rates (Q1→Q2, Q2→Q3)
   - Weighted trend (70% weight to recent quarter)
   - Exponentially weighted prior spend

3. **Economic Response**
   - Economic factor × (credit score / 700)
   - Captures how different credit profiles respond to economic conditions

4. **Seasonal Spending Power**
   - Holiday multiplier × average monthly spend
   - Predicts holiday spending capability

5. **Polynomial Features**
   - Squared terms for key predictors
   - Captures non-linear relationships
   - Used for spend, monthly average, and credit line

6. **High Income Interactions**
   - Special interactions for high-income customers
   - High income × credit capacity
   - High income × spend profile
   - High income × seasonal effect

7. **Ratio Features**
   - Balance to limit ratio
   - Monthly spend growth (2025 vs 2024)

#### Model Configuration
The XGBoost model uses a carefully tuned configuration:
- Objective: Squared error regression
- Trees: 200 estimators
- Max depth: 6 (to capture complex interactions)
- Learning rate: 0.05 (slower learning for better generalization)
- Regularization: Alpha=0.1, Lambda=1.0
- Early stopping: 20 rounds of no improvement
- Cross-validation: 20% validation split

#### Post-Processing
The model applies a post-processing adjustment:
- Initial predictions are blended with the Q4 adjustment factor
- 80% weight to model prediction, 20% to adjustment factor
- This fine-tunes predictions with domain knowledge

#### Model Performance
- Test RMSE: ~$6,830
- R² Score: 0.969
- The model achieves strong predictive performance with high explanatory power

#### Key Predictors
Top features by importance:
1. Average monthly spend (and its squared term)
2. Total 2024 spend
3. Credit line
4. Holiday spending power
5. Credit score
6. Utilization percentage

This model provides a robust foundation for predicting customer Q4 spending, which is then used by downstream models to determine appropriate credit line increases and risk assessments.

### Model 2: Account Segmentation

Model 2 classifies accounts into four distinct segments to determine their eligibility for credit line increases based on risk level and credit utilization patterns. This multi-class classification model is essential for targeting the right accounts for CLI offers.

#### Segmentation Categories
The model segments accounts into four categories:
1. **Segment 0: Eligible for CLI - No Risk**
   - Moderate utilization (30-70%)
   - Good credit score (≥670)
   - No high-risk flags

2. **Segment 1: Eligible for CLI - At Risk**
   - High utilization (>70%) or
   - Lower credit score (<670)
   - No critical risk factors

3. **Segment 2: No Increase Needed**
   - Low utilization (≤30%)
   - Credit line already sufficient for spending patterns

4. **Segment 3: High Risk (Non-Performing)**
   - Accounts with delinquency
   - Fraud history
   - Other high-risk factors identified by Model 3

#### Feature Selection
The model uses a comprehensive set of risk and behavior indicators:
- Credit profile: `credit_score`, `utilization_pct`
- Risk history: `delinquency_12mo`, `delinquency_24mo`, `has_fraud`
- Payment patterns: `payment_hist_1_12_delinquency_count`, `payment_hist_13_24_delinquency_count`
- Spending behavior: `total_spend_2024`, `total_spend_2025YTD`, `avg_transaction_size_2024`
- Financial metrics: `credit_line`, `current_balance`
- Customer profile: `num_accounts`, `is_high_income`

#### Model Configuration
The XGBoost classifier is configured for multi-class classification:
- Objective: Multi-class softmax probability
- Number of classes: 4
- Evaluation metric: Multi-class log loss
- Trees: 100 estimators
- Max depth: 4 (prevents overfitting on this complex classification)
- Learning rate: 0.1
- Sampling: 80% of data, 80% of features per tree
- Early stopping: 10 rounds

#### Cross-Validation Approach
- Uses stratified sampling to maintain class distribution
- 90% of training data used for model building
- 10% reserved for validation and early stopping
- 20% of total data held out for testing

#### Model Output
In addition to the segment classification, the model produces probability scores for each segment:
- `segment_0_prob`: Probability of being eligible with no risk
- `segment_1_prob`: Probability of being eligible with some risk
- `segment_2_prob`: Probability of not needing an increase
- `segment_3_prob`: Probability of being high risk

These probability scores provide a confidence level for each classification and can be used for borderline cases.

#### Performance Metrics
- Test Accuracy: ~94.5%
- Detailed performance metrics by segment:
  - Precision: How many accounts classified into a segment truly belong there
  - Recall: How many accounts truly belonging to a segment were correctly classified
  - F1-Score: Harmonic mean of precision and recall

#### Key Predictors
The top features driving the segmentation:
1. Credit utilization percentage
2. Credit score
3. Delinquency counts
4. Current balance to credit line ratio
5. Spending patterns

#### Business Impact
This segmentation model drives critical business decisions:
- Determines which accounts are eligible for credit line increases
- Helps prioritize accounts for different marketing treatments
- Supports risk management by identifying high-risk accounts
- Informs optimization of credit line increases to maximize acceptance rates

The segmentation results are used as inputs for Model 4 (CLI Recommendation), which determines the optimal increase amount for eligible accounts.

### Model 3: Risk Flagging

Model 3 is a binary classification model that identifies accounts with elevated risk of delinquency, default, or fraud. This critical risk assessment component ensures that credit line increases are only offered to accounts with acceptable risk profiles.

#### Target Variable Definition
The risk flag (`risk_flag`) is created based on multiple criteria:

1. **Primary Risk Criteria** (strict thresholds)
   - Multiple delinquencies in the past 12 months (>3 instances)
   - Multiple delinquencies in the past 13-24 months (>3 instances)
   - Any fraud history (`has_fraud` = 1)

2. **Secondary Risk Criteria** (applied if primary flags <5% of accounts)
   - Extremely high utilization (>95%) AND
   - Very low credit score (<600)

3. **Distribution Balancing**
   - Ensures a balanced dataset for model training (approximately 8% risk flagged)
   - Uses random assignment if needed to reach target distribution
   - Maintains statistical power for minority class detection

#### Feature Selection
The model uses a comprehensive set of risk indicators:
- Credit quality: `credit_score`
- Delinquency history: `delinquency_12mo`, `delinquency_24mo`, `has_fraud`
- Payment severity: `payment_hist_1_12_max_delinquency`, `payment_hist_13_24_max_delinquency`
- Account characteristics: `account_age_days`, `num_accounts`, `active_account_count`
- Financial status: `utilization_pct`, `credit_line`, `current_balance`
- Spending patterns: `total_spend_2024`, `total_spend_2025YTD`, `avg_monthly_spend_2024`
- Customer profile: `is_high_income`

#### Class Imbalance Handling
The model incorporates specialized techniques to address the class imbalance inherent in risk modeling:
- Calculation of `scale_pos_weight` based on class distribution
- Weighted learning that gives more importance to minority class examples
- Stratified sampling for train/test splits to maintain class distributions
- Evaluation metrics suitable for imbalanced classification

#### Model Configuration
The XGBoost classifier is optimized for binary classification with imbalanced data:
- Objective: Binary logistic regression
- Evaluation metric: Area Under the ROC Curve (AUC)
- Trees: 100 estimators
- Max depth: 3 (shallower to prevent overfitting to majority class)
- Learning rate: 0.1
- Sampling: 80% of data, 80% of features per tree
- Early stopping: 10 rounds
- Scale positive weight: Calculated from class distribution

#### Cross-Validation Approach
- Stratified train/test split (80/20) to preserve class distribution
- Further stratified validation split (10% of training data)
- Early stopping based on AUC to optimize for ranking performance

#### Model Output
The model produces two key outputs:
1. `risk_flag`: Binary prediction (0=not risky, 1=risky)
2. `risk_probability`: Probability of being risky (0-1 range)

The probability score is particularly valuable as an input to the CLI recommendation model, allowing for risk-adjusted CLI amounts.

#### Performance Metrics
- Test Accuracy: ~91%
- ROC AUC: ~0.65-0.75
- Precision/Recall metrics for both classes:
  - Precision (Risky): How many flagged accounts are truly risky
  - Recall (Risky): How many truly risky accounts were successfully identified

#### Key Predictors
The top features driving risk identification:
1. Delinquency history (recent and historical)
2. Credit score
3. Utilization percentage
4. Payment history delinquency counts
5. Account age

#### Business Impact
This risk flagging model has significant business implications:
- Protects the financial institution from extending credit to high-risk accounts
- Reduces potential credit losses and fraud exposure
- Improves the targeting efficiency of CLI programs
- Provides risk probabilities that can be used to fine-tune CLI amounts
- Serves as a critical input to the account segmentation model

The model's risk probabilities are used both for determining segment assignments and for calibrating appropriate CLI amounts in Model 4.

### Model 4: CLI Recommendation

Model 4 is a regression model that recommends optimized credit line increase (CLI) amounts for eligible accounts. This model takes into account spending patterns, risk profiles, and business constraints to generate appropriate CLI recommendations.

#### Eligibility Criteria
Only accounts in segments 0 and 1 (as classified by Model 2) are eligible for CLI recommendations:
- **Segment 0:** Eligible - No Risk accounts
- **Segment 1:** Eligible - At Risk accounts

The model excludes accounts in segments 2 (No Increase Needed) and 3 (High Risk).

#### Feature Selection
The model uses a comprehensive set of predictors:
- **Spending patterns:** `predicted_q4_spend`, `avg_monthly_spend_2024`, `total_spend_2024`
- **Risk metrics:** `risk_probability` (from Model 3), `credit_score`
- **Account characteristics:** `credit_line`, `current_balance`, `utilization_pct`
- **Customer profile:** `is_high_income`, `num_accounts`
- **Seasonal factors:** `holiday_spending_multiplier`, `q4_adjustment_factor`

#### Advanced Feature Engineering
The model creates specialized CLI prediction features:
1. **Spending-to-limit ratio:** Indicates if customers are constrained by current limits
2. **Available credit metrics:** Current headroom and available credit ratio
3. **Credit score tiers:** Categorical bins for credit scores (Poor, Fair, Good, Very Good, Excellent)
4. **Risk-adjusted CLI potential:** Higher potential increases for lower-risk customers
5. **Segment-specific spending capacity:** Different handling for Segment 0 vs. Segment 1
6. **High-income specific features:** Special treatment for high-income customers with good credit
7. **Compound CLI recommendation score:** Initial recommendation based on spending, credit quality, utilization, and risk

#### Model Configuration
The XGBoost regressor is optimized for CLI amount prediction:
- Objective: Squared error regression
- Trees: 150 estimators
- Max depth: 5
- Learning rate: 0.05
- Regularization: Alpha=0.1, Lambda=1.0
- Early stopping: 15 rounds

#### Business Rules and Constraints
The model applies several business rules as post-processing:
1. **Minimum CLI amount:** $500 for eligible accounts
2. **Maximum CLI caps by segment:**
   - Segment 0 (No Risk): Up to 100% of current credit line
   - Segment 1 (At Risk): Up to 50% of current credit line
3. **High-income adjustments:**
   - High-income accounts in Segment 0 can receive up to 150% of current limit
   - High-income accounts in Segment 1 can receive up to 75% of current limit
4. **Round recommendations:** CLI amounts rounded to nearest $100

#### Performance Metrics
- Test RMSE: ~$6,240
- Mean CLI amount: ~$14,510
- RMSE as percentage of mean: ~43%

#### Key Predictors
The top features driving CLI recommendations:
1. Predicted Q4 spend
2. Current credit line
3. Compound CLI recommendation score
4. Credit score
5. Utilization percentage
6. Risk probability
7. Holiday spending multiplier

#### Business Impact
The CLI Recommendation model has significant business implications:
- Optimizes credit line increases to match customer spending patterns
- Balances growth opportunities against risk exposure
- Provides personalized CLI amounts rather than generic increases
- Adapts recommendations based on risk profiles and customer segments
- Incorporates business rules and constraints to ensure practical recommendations

The final recommendations are incorporated into the enriched dataset along with predictions from all previous models, creating a comprehensive view for decision-making.

## Model Output Workflow

The entire XGBoost modeling pipeline is designed to enrich a master dataset with predictions that support credit line increase decisions. The workflow follows a sequential process where each model builds upon the outputs of previous models.

### Master Dataset Enrichment
Each model adds new columns to the master dataset:
1. **Model 1 (Q4 Spend Prediction)** adds:
   - `predicted_q4_spend`: Forecasted spending for Q4 2025

2. **Model 2 (Account Segmentation)** adds:
   - `segment_label`: Classification into segments 0-3
   - `segment_0_prob` through `segment_3_prob`: Probability scores for each segment

3. **Model 3 (Risk Flagging)** adds:
   - `risk_flag`: Binary risk classification (0=not risky, 1=risky)
   - `risk_probability`: Probability of being classified as risky

4. **Model 4 (CLI Recommendation)** adds:
   - `recommended_cli_amount`: Dollar amount recommended for credit line increase

### Visualization Outputs
The modeling process generates several visualizations saved to the `visualizations` directory:

1. **Segment Distribution**: Pie chart showing the proportion of accounts in each segment
2. **Risk Distribution**: Histogram of risk probabilities across all accounts
3. **Q4 Spend Prediction**: Scatter plot comparing predicted vs. actual Q4 spending
4. **CLI by Segment**: Box plot showing CLI recommendation distributions by segment
5. **Credit Score vs. CLI**: Scatter plot visualizing the relationship between credit scores and recommended CLI amounts

### Model and Data Persistence
The following artifacts are saved at the end of the modeling process:

1. **Enriched Dataset**: 
   - Saved to: `exploratory_data_analysis/master_user_dataset_with_predictions.csv`
   - Contains all original features plus model predictions

2. **Trained Models**:
   - XGBoost model objects saved to the `models` directory:
     - `q4_spend_prediction_model.joblib`
     - `account_segmentation_model.joblib`
     - `risk_flagging_model.joblib`
     - `cli_recommendation_model.joblib`

3. **Final Metrics Summary**:
   - Overall performance metrics for each model are displayed after completion
   - Statistics on accounts processed, risk classifications, and CLI recommendations

### Using the Enriched Dataset
The final master dataset with predictions provides a comprehensive decision-support tool that can be used to:

1. **Identify Growth Opportunities**: Target accounts with high predicted Q4 spend in segments 0 and 1
2. **Manage Risk**: Exclude high-risk accounts (segment 3) from CLI offers
3. **Optimize CLI Amounts**: Use the recommended amounts for personalized CLI offers
4. **Segment Marketing**: Tailor marketing messages based on segment classification
5. **Monitor Performance**: Compare actual outcomes to predicted metrics for model refinement
