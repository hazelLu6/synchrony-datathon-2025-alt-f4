# Q4 Spend Prediction & CLI Optimization Project

## 1. Project Overview

This project addresses four key objectives for a financial institution:

1. Q4 Spending Prediction:  
   • Build a regression model that forecasts each customer's Q4 spend based on historical data (past Q4 trends, yearly spending) and the current year's last 8 months of data.

2. Customer Segmentation:  
   • Classify accounts into four segments:  
     0. Eligible for CLI (No Risk)  
     1. Eligible for CLI (At Risk)  
     2. No CLI Increase Needed  
     3. Non-Performing / High-Risk  

3. Risk Detection:  
   • Identify high-risk accounts (fraud, delinquencies, default risk).

4. Credit Line Increase (CLI) Recommendation:  
   • Suggest a personalized CLI amount per customer, balancing growth opportunities with potential default or fraud risk.

These objectives aim to boost revenue from increased credit lines while minimizing risk from high-risk accounts.

---

## 2. Repository Structure

Below is a sample structure of the repository and relevant files:
├── data/ # Raw CSV data files
├── cleaned_data/ # Cleaned/processed data files
├── exploratory_data_analysis/ # Intermediate or final datasets, metrics, logs
├── models/ # Model artifacts (joblib or pickle files)
├── master.py # Script to clean & aggregate data into a user-level dataset
├── xgboost_models.py # Contains XGBoost model definitions
├── run_xgboost_pipeline.py # Main script to run the entire modeling pipeline
├── README.md # Project instructions (this file)
```
---

## 3. Setup Instructions

1. **Clone or Download the Repository**  
   • Use Git or directly download the ZIP from your repository or team folder.

2. **Create & Activate a Python Environment (Python 3.8+ recommended)**  
   Example with Conda:
   ```
   conda create -n credit_ml python=3.9
   conda activate credit_ml
   ```
   Or use virtualenv, venv, etc.

3. **Install Required Packages**  
   If you have a requirements.txt file, run:
   ```
   pip install -r requirements.txt
   ```
   Common packages needed include:
   - pandas  
   - numpy  
   - scikit-learn  
   - xgboost  
   - joblib  
   - warnings (built into Python, but typically suppressed in the scripts)

4. **Place Data Files**  
   Ensure CSV files are placed in the correct directories:
   - data/ (account_dim_20250325.csv, transaction_fact_20250325.csv, world transaction data, etc.)
   - cleaned_data/ (if you have pre-cleaned or interim CSV files)

5. **(Optional) Adjust File Paths**  
   • If your data is in a different location, update paths in `master.py` or other scripts accordingly.

---

## 4. How to Run

1. **Data Cleaning & Aggregation**  
   • Run `master.py`:
     ```
     python master.py
     ```
   • This merges and cleans raw data, producing a single user-level dataset.  
   • By default, it writes out a file like `exploratory_data_analysis/master_user_dataset.csv`.

2. **Modeling Pipeline**  
   • Execute `run_xgboost_pipeline.py`:  
     ```
     python run_xgboost_pipeline.py
     ```
   • This script attempts to load the master dataset and then:  
     1. Initializes the data with required columns (correcting any missing fields for the models)  
     2. Runs Model 1: Q4 Spend Prediction  
     3. Runs Model 2: Customer Segmentation  
     4. Runs Model 3: Risk Flagging  
     5. Runs Model 4: CLI Recommendation  
   • Final output includes:  
     - A new CSV in `exploratory_data_analysis/`, e.g. `master_user_dataset_with_predictions.csv`, with predictions and recommended CLI amounts.  
     - Saved model artifacts (if no errors occur) in `models/`.

3. **Review Results**  
   - Check console logs for performance metrics (RMSE, accuracy, etc.).  
   - Explore the final CSV to see each user's predicted Q4 spend, assigned segment, risk flag, and CLI recommendation.

---

## 5. Features & Approach

1. **Feature Engineering**  
   • Aggregated spend for Q1, Q2, Q3, plus YTD totals  
   • Holiday multiplier for seasonal Q4 spike  
   • Fraud/delinquency flags from 12-24 months of history  
   • Credit line & current balance → utilization %  
   • Payment history columns to identify risk patterns

2. **Models**  
   A. **Model 1: Q4 Spend Prediction (Regression)**  
      - Predicts each user's Q4 spend.  
      - Key features: Past Q4 patterns, total 2024 spend, monthly average, holiday multiplier.

   B. **Model 2: Customer Segmentation (Multi-class Classification)**  
      - Segments: Eligible-No Risk (0), Eligible-At Risk (1), No Increase Needed (2), High-Risk (3).  
      - Key features: credit_score, utilization_pct, delinquency counts, fraud flags.

   C. **Model 3: Risk Flagging (Binary Classification)**  
      - Outputs a probability or a hard flag (0 or 1) indicating risk of default/fraud.  
      - Focuses on recall for the "risky" class, preventing missed high-risk accounts.

   D. **Model 4: CLI Recommendation (Regression)**  
      - Suggests a credit line increase amount.  
      - Inputs: credit_line, segment_label, risk_probability, is_high_income, Q4 spend potential.

---

## 6. Known Limitations & Future Enhancements

1. **Macro-economic data** (unemployment, inflation, consumer sentiment) could refine spending models.  
2. **Real-time streaming** might catch new fraud patterns more effectively.  
3. Additional **A/B testing** of CLI offers can tune acceptance rates and reduce default risk.

---

## 7. Example Results

• **Q4 Spend Model**:  
  - Achieves an R² close to 0.98 on training data (example figure).  
  - RMSE indicates robust predictive capability, with strong holiday season correlation.

• **Segmentation**:  
  - Overall multi-class accuracy ~94%.  
  - Segment 0 & 1 help identify CLI eligibility, while Segment 2 & 3 focus on restricting unnecessary or high-risk CLI.

• **Risk Model**:  
  - ~84% accuracy with emphasis on recall for risky customers.  
  - High utilization and delinquency are leading risk indicators.

• **CLI Model**:  
  - Recommends carefully calibrated increases for eligible accounts.  
  - Higher CLI for high-income + good credit. Minimal or no CLI if at risk.

---

## 8. Reproduce & Contribute

1. **Fork This Repository** or **Download** it.  
2. **Set Up** your Python environment & data paths.  
3. **Run** the cleaning script (`master.py`) and pipeline (`run_xgboost_pipeline.py`).  
4. **Inspect** the output CSV for final predictions and recommended CLI.  
5. **Pull Requests** are welcome to enhance feature engineering or improve the models.

---

## 9. Presentation Video

• We have prepared a short (< 7-minute) explanation video:  
  - [Unlisted YouTube Link Here] (Replace with final link)  
  - Google slide used: https://docs.google.com/presentation/d/17cAbNUgTXFelDzF7bCX9abowQR43-TFINMWHQ6a9k3o/edit?usp=sharing 
  - More detailed logic explained: 

In the video, we briefly walk through the business problem, data sources, featured models, and final insights. 

---

## 10. Submission Guidelines & Acknowledgments

• **Submission**: Zip all relevant code (including this README) and upload to your project folder / repository.  
• **Team**: [Alt F4: Tianyi, Hazel, Ben, Angel ].  
• **Thanks** to everyone who contributed to data generation and the domain SMEs who provided key insights.

---

**End of README**