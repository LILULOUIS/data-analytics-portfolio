# Employee Attrition — Analysis and Prediction
# Original project by Louis Lu, Nov 2026.
# Data: HR employee file provided by Louis (15,000 rows, 7 columns).
# The raw dataset is NOT included in this repository (it is Louis's
# own data). To reproduce: place the file as
# data/employee_attrition.xlsx and run
#   python attrition_analysis.py
# It writes charts to charts/ and summary tables to outputs/.
#
# Business question: who leaves, and what drives attrition, so HR
# can intervene before the employees most at risk walk out?
#
# Method
# - Exploratory analysis: attrition by satisfaction, monthly hours,
#   tenure, number of projects, salary band, and evaluation score.
# - Two models on an 80/20 stratified train/test split (random_state=42):
#   logistic regression (features standardized, salary one-hot encoded)
#   and random forest (300 trees). Evaluated with ROC AUC,
#   precision/recall, and 5-fold cross-validation.
#
# Key data-quality finding (checked in this script)
# - 3,044 exact duplicate rows removed (15,000 -> 11,956).
# - The duplicates were not random: 46.8% of rows in duplicate groups
#   had left, vs 10.9% of unique rows, and 1,613 of the 3,044 removed
#   rows were leavers. The raw attrition rate of 23.8% is therefore
#   inflated; the de-duplicated rate is 16.4% (1,959 of 11,956).
#   All rates and models in this project use the de-duplicated data.
# - No missing values in any column.
#
# Results (test set, de-duplicated data)
# - Attrition rate 16.4%.
# - Random forest ROC AUC 0.975 (CV 0.969-0.992), accuracy 98.2%,
#   attrition precision 0.98, recall 0.91.
# - Logistic regression ROC AUC 0.798 (CV 0.790-0.828), attrition
#   recall only 0.16 at a 0.5 threshold. The gap between the two
#   models is the story: attrition here is strongly non-linear
#   (U-shaped in satisfaction, hours, and evaluation), which a
#   linear model cannot capture.
# - Random forest importance: satisfaction 0.35, tenure 0.18,
#   projects 0.18, monthly hours 0.16, evaluation 0.13, salary ~0.01.
#   Salary matters univariately (low 20.2% vs high 4.8%) but adds
#   almost nothing once satisfaction and the other factors are in.
# - U-shaped patterns: satisfaction 3-5 -> 43.6% attrition, 5-7 ->
#   1.5%, 7-10 -> 10.0%; monthly hours <=150 -> 25.5%, 150-200 ->
#   8.0%, 250+ -> 29.1%; evaluation <=0.5 -> 27.2%, 0.9-1.0 -> 23.9%.
# - Tenure: 2 years 1.1%, 3 years 16.5%, 5 years 45.3%, 7+ years 0%
#   (n=282, no leavers). Projects: 2 projects 53.5%, 3 projects 1.1%,
#   7 projects 100% (143 of 143 left).
#
# Honest caveat on the random forest score
# - AUC 0.975 is unusually high for attrition. This dataset has
#   near-deterministic patterns (everyone with 7 projects left; no
#   one with 7+ years tenure left), which a tree model exploits.
#   Treat the score as a property of this dataset, not as a claim
#   that real-world attrition is 97% predictable.

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

plt.style.use("ggplot")
plt.rcParams["figure.figsize"] = (12, 8)

DATA_PATH = os.path.join("data", "employee_attrition.xlsx")
CHART_DIR = "charts"
OUTPUT_DIR = "outputs"

COLUMN_MAP = {
    "工资": "salary",
    "满意度": "satisfaction",
    "考核得分": "evaluation",
    "工程数量": "projects",
    "月工时": "monthly_hours",
    "工龄": "tenure",
    "离职": "left",
}


def main():
    os.makedirs(CHART_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    raw = pd.read_excel(DATA_PATH).rename(columns=COLUMN_MAP)
    print(f"Raw shape: {raw.shape}")
    print(f"Missing values:\n{raw.isna().sum().to_string()}")
    dup_mask = raw.duplicated(keep=False)
    print(
        f"Attrition in duplicate groups: {raw[dup_mask]['left'].mean():.3f} "
        f"vs unique rows: {raw[~dup_mask]['left'].mean():.3f}"
    )
    df = raw.drop_duplicates().copy()
    print(f"After dropping {len(raw) - len(df)} duplicate rows: {df.shape}")
    print(
        f"Attrition rate: raw {raw['left'].mean():.3f} -> "
        f"de-duplicated {df['left'].mean():.3f}"
    )

    # --- Exploratory analysis ----------------------------------------
    df["sat_bin"] = pd.cut(df["satisfaction"], [0, 3, 5, 7, 10], include_lowest=True)
    df["hrs_bin"] = pd.cut(df["monthly_hours"], [0, 150, 200, 250, 310], include_lowest=True)
    df["eval_bin"] = pd.cut(df["evaluation"], [0, 0.5, 0.7, 0.9, 1.0], include_lowest=True)
    tables = {
        "attrition-by-satisfaction": df.groupby("sat_bin", observed=True)["left"],
        "attrition-by-hours": df.groupby("hrs_bin", observed=True)["left"],
        "attrition-by-tenure": df.groupby("tenure")["left"],
        "attrition-by-projects": df.groupby("projects")["left"],
        "attrition-by-salary": df.groupby("salary")["left"],
    }
    for name, grouped in tables.items():
        table = grouped.agg(["mean", "count"]).round(4)
        print(f"\n{name}:\n{table.to_string()}")
        table.to_csv(os.path.join(OUTPUT_DIR, f"{name}.csv"))
    print("\nGroup means:\n", df.groupby("left").mean(numeric_only=True).round(2).to_string())
    df.groupby("left").mean(numeric_only=True).round(2).to_csv(
        os.path.join(OUTPUT_DIR, "group-means.csv")
    )

    # --- Models -------------------------------------------------------
    X = pd.get_dummies(
        df[["satisfaction", "evaluation", "projects", "monthly_hours", "tenure", "salary"]],
        columns=["salary"],
        drop_first=True,
        dtype=float,
    )
    X.columns = [
        "satisfaction",
        "evaluation",
        "projects",
        "monthly_hours",
        "tenure",
        "salary_low",
        "salary_high",
    ]
    y = df["left"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=1000, random_state=42).fit(X_train_s, y_train)
    rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1).fit(
        X_train, y_train
    )

    metrics = {
        "n_rows_raw": int(len(raw)),
        "n_rows_deduped": int(len(df)),
        "duplicates_removed": int(len(raw) - len(df)),
        "attrition_rate_raw": round(float(raw["left"].mean()), 4),
        "attrition_rate_deduped": round(float(y.mean()), 4),
    }
    for name, model, X_eval in [("logreg", lr, X_test_s), ("rf", rf, X_test)]:
        proba = model.predict_proba(X_eval)[:, 1]
        pred = (proba >= 0.5).astype(int)
        auc = roc_auc_score(y_test, proba)
        cv = cross_val_score(
            model,
            X_train_s if name == "logreg" else X_train,
            y_train,
            cv=5,
            scoring="roc_auc",
        )
        print(f"\n{name}: test AUC {auc:.4f}, CV AUC {cv.round(4)}")
        print(classification_report(y_test, pred, digits=3))
        print(confusion_matrix(y_test, pred))
        metrics[f"{name}_auc"] = round(float(auc), 4)
        metrics[f"{name}_cv_auc"] = [round(float(v), 4) for v in cv]

    print("\nLogistic coefficients (standardized):")
    metrics["logreg_coef"] = {}
    for feat, coef in zip(X.columns, lr.coef_[0]):
        print(f"  {feat}: {coef:.3f} (odds ratio {np.exp(coef):.3f})")
        metrics["logreg_coef"][feat] = round(float(coef), 3)
    print("Random forest importance:")
    metrics["rf_importance"] = {}
    for feat, imp in sorted(zip(X.columns, rf.feature_importances_), key=lambda t: -t[1]):
        print(f"  {feat}: {imp:.3f}")
        metrics["rf_importance"][feat] = round(float(imp), 3)

    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # --- Charts -------------------------------------------------------
    def bar_chart(data, title, xlabel, fname, color):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar([str(x) for x in data.index], data["mean"] * 100, color=color)
        ax.set_ylabel("Attrition rate (%)")
        ax.set_xlabel(xlabel)
        ax.set_title(title, weight="bold")
        for i, v in enumerate(data["mean"] * 100):
            ax.text(i, v + 1, f"{v:.1f}%", ha="center")
        plt.tight_layout()
        plt.savefig(os.path.join(CHART_DIR, fname), dpi=200)
        plt.close()

    bar_chart(
        tables["attrition-by-satisfaction"].agg(["mean", "count"]),
        "Attrition is U-shaped in satisfaction",
        "Satisfaction score",
        "attrition-by-satisfaction.png",
        "#8e44ad",
    )
    bar_chart(
        tables["attrition-by-hours"].agg(["mean", "count"]),
        "Attrition is U-shaped in monthly hours",
        "Monthly hours",
        "attrition-by-hours.png",
        "#e67e22",
    )
    bar_chart(
        tables["attrition-by-tenure"].agg(["mean", "count"]),
        "Attrition peaks at 5 years of tenure",
        "Tenure (years)",
        "attrition-by-tenure.png",
        "#c0392b",
    )
    bar_chart(
        tables["attrition-by-projects"].agg(["mean", "count"]),
        "Attrition by number of projects",
        "Number of projects",
        "attrition-by-projects.png",
        "#2e86c1",
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    for name, model, X_eval in [("logreg", lr, X_test_s), ("rf", rf, X_test)]:
        fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_eval)[:, 1])
        label = "Logistic Regression" if name == "logreg" else "Random Forest"
        ax.plot(fpr, tpr, label=f"{label} (AUC {metrics[f'{name}_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves — attrition prediction (test set)", weight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, "roc-curves.png"), dpi=200)
    plt.close()

    importance = sorted(
        zip(X.columns, rf.feature_importances_), key=lambda t: -t[1]
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(
        [x[0] for x in importance][::-1],
        [x[1] for x in importance][::-1],
        color="#2e86c1",
    )
    ax.set_xlabel("Random Forest feature importance")
    ax.set_title("What drives attrition? (Random Forest)", weight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, "feature-importance.png"), dpi=200)
    plt.close()

    print(f"\nDone. Charts in {CHART_DIR}/, tables in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
