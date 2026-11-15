# Brokerage Customer Churn — Analysis and Prediction

Original project by Louis Lu, Nov 2026. Built from a brokerage customer dataset provided by Louis — this is original work, not adapted from a tutorial.

## Business question

Which customers are most likely to churn, and what drives that risk, so retention effort can be targeted at the customers who need it most?

## Data

7,043 customers, 6 columns: account balance (CNY), days since last trade, last month's commission (CNY), total commission (CNY), tenure with the broker (years), and churned (0/1).

The raw dataset is **not included** in this repository. It contains no customer IDs or personal information, but it is Louis's own data. To reproduce the analysis, place the file as `data/brokerage_churn.xlsx` and run:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl
python churn_analysis.py
```

Data quality: no missing values in any column; 6 exact duplicate rows removed (7,043 → 7,037).

## Findings

- **Churn rate 26.5%** — 1,869 of 7,037 customers churned.
- **Tenure is the clearest univariate signal.** Churn is 47.4% for customers in their first year, falling steadily to 6.5% at 5 years.
- **Recency matters.** Churn is 17.0% for customers who traded within 30 days, rising to 45.0% when the last trade was 180+ days ago.
- **Balance matters.** Churn is 43.1% in the lowest balance quartile (≤ ¥56,000) vs 14.5% in the highest (> ¥361,000). Churned customers hold on average ¥158,000 vs ¥250,000 for retained customers.
- **A counterintuitive one:** churned customers paid *more* commission last month on average (¥372 vs ¥306), even though their lifetime commission is lower. High recent commission alone does not mean a safe customer.

## Models

Two models, 80/20 stratified train/test split (random_state=42), 5-fold cross-validation:

| Model | Test ROC AUC | CV ROC AUC |
| --- | --- | --- |
| Logistic regression | 0.832 | 0.788–0.827 |
| Random forest | 0.767 | 0.760–0.789 |

Logistic regression at a 0.5 threshold: accuracy 79.5%, churn precision 0.68, churn recall 0.43. It ranks risk well, but catches less than half of churners at 0.5 — so use the predicted probability as a risk score for targeting, not as a hard yes/no label.

Standardized logistic coefficients: last month's commission +1.50, days since last trade +0.68, balance −0.79, tenure −0.09. Random forest importance: last month's commission 0.39, balance 0.34, recency 0.19, tenure 0.09.

## A data-quality catch worth noting

`total_commission` is an exact linear function of balance: `total_commission = 0.0111111 × balance + 1777.78` (Pearson r = 1.0, max residual < 1e-11). The two columns carry identical information, so `total_commission` was dropped before modelling — otherwise the logistic coefficients for the two would not be separately interpretable. The check is in `check_collinearity()` in `churn_analysis.py`.

Related caveat: the features are correlated with each other (balance–tenure r = 0.82, recency–tenure r = −0.77), so read individual coefficients as directional, not as isolated causal effects.

## What is missing / next steps

- No customer ID, date, channel, or product information, so no time trend, cohort analysis, or targeting of specific customers.
- Next steps: tune the decision threshold for recall, try gradient boosting, and add a simple retention playbook (e.g. outreach to first-year customers inactive for 90+ days).

## Files

- `churn_analysis.py` — full analysis and modelling script.
- `charts/` — churn by tenure, churn by recency, balance distribution, ROC curves, feature importance.
- `outputs/` — summary tables (churn by tenure / recency / balance quartile) and `metrics.json`.

## Skills

Python, Pandas, scikit-learn (logistic regression, random forest), exploratory data analysis, churn modelling, data-quality checks.
