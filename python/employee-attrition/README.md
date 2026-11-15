# Employee Attrition — Analysis and Prediction

Original project by Louis Lu, Nov 2026. Built from an HR employee dataset provided by Louis — this is original work, not adapted from a tutorial.

## Business question

Who leaves, and what drives attrition, so HR can intervene before the employees most at risk walk out?

## Data

15,000 employees, 7 columns: salary band (low / medium / high), satisfaction (0–10), evaluation score (0–1), number of projects, monthly hours, tenure (years), and left (0/1).

The raw dataset is **not included** in this repository. It contains no employee IDs or personal information, but it is Louis's own data. To reproduce the analysis, place the file as `data/employee_attrition.xlsx` and run:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl
python attrition_analysis.py
```

## A data-quality catch first

The file contains **3,044 exact duplicate rows** — and they are not random. 46.8% of the rows in duplicate groups had left, vs 10.9% of unique rows; 1,613 of the 3,044 removed rows were leavers. So the raw attrition rate of 23.8% is inflated. After de-duplication the real rate is **16.4%** (1,959 of 11,956). Every rate and model in this project uses the de-duplicated data. No missing values in any column.

## Findings

- **Satisfaction is U-shaped.** Attrition is 43.6% at satisfaction 3–5, only 1.5% at 5–7, and 10.0% above 7. The safest employees are not the happiest — they are the moderately satisfied ones.
- **Monthly hours are U-shaped too.** Attrition is 25.5% below 150 hours, 8.0% at 150–200 hours, and 29.1% above 250 hours. Both underwork and overwork predict leaving.
- **Evaluation is U-shaped as well.** Attrition is 27.2% for scores ≤ 0.5 and 23.9% for scores above 0.9 — low performers leave, and so do top performers.
- **Tenure peaks at 5 years.** Attrition is 1.1% at 2 years, 16.5% at 3 years, 45.3% at 5 years, and 0% at 7+ years (282 employees, no leavers).
- **Projects are extreme at both ends.** Attrition is 53.5% for employees with 2 projects, 1.1% with 3 projects, and **100% with 7 projects (143 of 143 left)**.
- **Salary matters univariately, not in the model.** Low salary 20.2%, medium 14.5%, high 4.8% — but once satisfaction and the other factors are included, salary adds almost nothing (random forest importance ~0.01).

## Models

Two models, 80/20 stratified train/test split (random_state=42), 5-fold cross-validation:

| Model | Test ROC AUC | CV ROC AUC |
| --- | --- | --- |
| Logistic regression | 0.798 | 0.790–0.828 |
| Random forest | 0.975 | 0.969–0.992 |

Logistic regression catches only 16% of leavers at a 0.5 threshold (precision 0.43). The gap between the two models is the story: attrition here is strongly non-linear, which a linear model cannot capture. Random forest importance: satisfaction 0.35, tenure 0.18, projects 0.18, monthly hours 0.16, evaluation 0.13, salary ~0.01.

**A note on the 0.975:** that score is unusually high for attrition. This dataset has near-deterministic patterns (everyone with 7 projects left; no one with 7+ years tenure left), which a tree model exploits. Treat it as a property of this dataset, not as a claim that real-world attrition is 97% predictable.

## What is missing / next steps

- No department, hire date, promotion, or manager information, so no cohort or team-level analysis.
- Next steps: threshold tuning for the random forest, a simple at-risk list (e.g. satisfaction 3–5 with 5 years tenure and 250+ hours), and checking whether the 7-project and 7-year patterns hold in a fresh sample.

## Files

- `attrition_analysis.py` — full analysis and modelling script.
- `charts/` — attrition by satisfaction, hours, tenure, and projects, plus ROC curves and feature importance.
- `outputs/` — summary tables (attrition by satisfaction / hours / tenure / projects / salary) and `metrics.json`.

## Skills

Python, Pandas, scikit-learn (logistic regression, random forest), exploratory data analysis, attrition modelling, data-quality checks.
