# Brokerage Customer Churn — Analysis and Prediction
# Original project by Louis Lu, Nov 2026.
# Data: brokerage customer file provided by Louis (7,043 rows, 6 columns).
# The raw dataset is NOT included in this repository (no customer IDs
# or personal information in it, but it is Louis's own data).
# To reproduce: place the file as data/brokerage_churn.xlsx and run
#   python churn_analysis.py
# It writes charts to charts/ and summary tables to outputs/.
#
# Business question: which customers are most likely to churn, and
# what drives that risk, so retention effort can be targeted?
#
# Method
# - Exploratory analysis: churn rate by tenure, recency, and balance
#   quartile; group means; correlations.
# - Two models on an 80/20 stratified train/test split (random_state=42):
#   logistic regression (features standardized) and random forest
#   (300 trees). Evaluated with ROC AUC, precision/recall, and
#   5-fold cross-validation.
#
# Key data-quality finding (checked in this script)
# - 6 exact duplicate rows removed (7,043 -> 7,037).
# - No missing values in any column.
# - total_commission is an exact linear function of balance:
#     total_commission = 0.0111111 * balance + 1777.78
#   (Pearson r = 1.0, max residual < 1e-11). The two columns carry
#   identical information, so total_commission is dropped before
#   modelling; otherwise the logistic coefficients are not
#   separately interpretable. The check is in check_collinearity().
#
# Results (test set, after dropping total_commission)
# - Churn rate 26.5% (1,869 of 7,037 customers).
# - Logistic regression ROC AUC 0.832 (CV 0.788-0.827);
#   random forest ROC AUC 0.767 (CV 0.760-0.789).
# - Logistic regression at a 0.5 threshold: accuracy 79.5%,
#   churn precision 0.68, churn recall 0.43 — it ranks risk well
#   but catches less than half of churners at 0.5, so use the
#   predicted probability as a risk score, not a hard label.
# - Standardized logistic coefficients: last_month_commission +1.50,
#   recency_days +0.68, balance -0.79, tenure_years -0.09.
#   Random forest importance: last_month_commission 0.39, balance 0.34,
#   recency_days 0.19, tenure_years 0.09.
# - Univariate signals: churn 47.4% at 0 years tenure vs 6.5% at
#   5 years; 45.0% when last trade was 180+ days ago vs 17.0% within
#   30 days; 43.1% in the lowest balance quartile vs 14.5% in the
#   highest.
# - Caveat: the features are correlated (balance-tenure r = 0.82,
#   recency-tenure r = -0.77), so read individual coefficients as
#   directional, not as isolated causal effects.

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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

DATA_PATH = os.path.join("data", "brokerage_churn.xlsx")
CHART_DIR = "charts"
OUTPUT_DIR = "outputs"

COLUMN_MAP = {
    "账户资金（元）": "balance",
    "最后一次交易距今时间（天）": "recency_days",
    "上月交易佣金（元）": "last_month_commission",
    "累计交易佣金（元）": "total_commission",
    "本券商使用时长（年）": "tenure_years",
    "是否流失": "churned",
}


def check_collinearity(df):
    """Check whether total_commission is an exact linear function of balance."""
    b = df["balance"].to_numpy(dtype=float)
    t = df["total_commission"].to_numpy(dtype=float)
    r = np.corrcoef(b, t)[0, 1]
    a, c = np.linalg.lstsq(
        np.vstack([b, np.ones_like(b)]).T, t, rcond=None
    )[0]
    max_resid = float(np.abs(t - (a * b + c)).max())
    print(f"corr(balance, total_commission) = {r:.6f}")
    print(f"total_commission = {a:.7f} * balance + {c:.2f}, max residual {max_resid:.2e}")
    return r, max_resid


def main():
    os.makedirs(CHART_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_excel(DATA_PATH).rename(columns=COLUMN_MAP)
    print(f"Raw shape: {df.shape}")
    print(f"Missing values:\n{df.isna().sum().to_string()}")
    df = df.drop_duplicates()
    print(f"After dropping {7043 - len(df)} duplicate rows: {df.shape}")
    print(f"Churn rate: {df['churned'].mean():.3f}")

    check_collinearity(df)
    # Drop the redundant column before modelling (see header note).
    df_model = df.drop(columns=["total_commission"])

    # --- Exploratory analysis ----------------------------------------
    by_tenure = df.groupby("tenure_years")["churned"].agg(["mean", "count"])
    by_recency = df.groupby(
        pd.cut(df["recency_days"], [0, 30, 90, 180, 361], include_lowest=True),
        observed=True,
    )["churned"].agg(["mean", "count"])
    by_balance = df.groupby(
        pd.qcut(df["balance"], 4), observed=True
    )["churned"].agg(["mean", "count"])
    print("\nChurn by tenure:\n", by_tenure.round(4).to_string())
    print("\nChurn by recency:\n", by_recency.round(4).to_string())
    print("\nChurn by balance quartile:\n", by_balance.round(4).to_string())
    print("\nGroup means:\n", df.groupby("churned").mean(numeric_only=True).round(1).to_string())
    by_tenure.to_csv(os.path.join(OUTPUT_DIR, "churn-by-tenure.csv"))
    by_recency.to_csv(os.path.join(OUTPUT_DIR, "churn-by-recency.csv"))
    by_balance.to_csv(os.path.join(OUTPUT_DIR, "churn-by-balance-quartile.csv"))

    # --- Models -------------------------------------------------------
    features = ["balance", "recency_days", "last_month_commission", "tenure_years"]
    X = df_model[features]
    y = df_model["churned"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=1000, random_state=42).fit(X_train_s, y_train)
    rf = RandomForestClassifier(
        n_estimators=300, random_state=42, n_jobs=-1
    ).fit(X_train, y_train)

    metrics = {"n_rows": int(len(df)), "churn_rate": round(float(y.mean()), 4)}
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
    for feat, coef in zip(features, lr.coef_[0]):
        print(f"  {feat}: {coef:.3f} (odds ratio {np.exp(coef):.3f})")
        metrics["logreg_coef"][feat] = round(float(coef), 3)
    print("Random forest importance:")
    metrics["rf_importance"] = {}
    for feat, imp in sorted(
        zip(features, rf.feature_importances_), key=lambda t: -t[1]
    ):
        print(f"  {feat}: {imp:.3f}")
        metrics["rf_importance"][feat] = round(float(imp), 3)

    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # --- Charts -------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(by_tenure.index.astype(str), by_tenure["mean"] * 100, color="#c0392b")
    ax.set_ylabel("Churn rate (%)")
    ax.set_xlabel("Tenure with broker (years)")
    ax.set_title("Churn rate falls sharply with tenure", weight="bold")
    for i, v in enumerate(by_tenure["mean"] * 100):
        ax.text(i, v + 1, f"{v:.1f}%", ha="center")
    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, "churn-by-tenure.png"), dpi=200)
    plt.close()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([str(x) for x in by_recency.index], by_recency["mean"] * 100, color="#e67e22")
    ax.set_ylabel("Churn rate (%)")
    ax.set_xlabel("Days since last trade")
    ax.set_title("Churn rate rises with days since last trade", weight="bold")
    for i, v in enumerate(by_recency["mean"] * 100):
        ax.text(i, v + 1, f"{v:.1f}%", ha="center")
    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, "churn-by-recency.png"), dpi=200)
    plt.close()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(
        [df[df.churned == 0]["balance"], df[df.churned == 1]["balance"]],
        bins=40,
        label=["Retained", "Churned"],
        alpha=0.7,
    )
    ax.set_xscale("log")
    ax.set_xlabel("Account balance (CNY, log scale)")
    ax.set_ylabel("Customers")
    ax.set_title("Churned customers hold smaller balances", weight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, "balance-by-churn.png"), dpi=200)
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 6))
    for name, model, X_eval in [("logreg", lr, X_test_s), ("rf", rf, X_test)]:
        fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_eval)[:, 1])
        label = (
            "Logistic Regression" if name == "logreg" else "Random Forest"
        )
        ax.plot(fpr, tpr, label=f"{label} (AUC {metrics[f'{name}_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves — churn prediction (test set)", weight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, "roc-curves.png"), dpi=200)
    plt.close()

    importance = sorted(
        zip(features, rf.feature_importances_), key=lambda t: -t[1]
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh([x[0] for x in importance][::-1], [x[1] for x in importance][::-1], color="#2e86c1")
    ax.set_xlabel("Random Forest feature importance")
    ax.set_title("What drives churn? (Random Forest)", weight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, "feature-importance.png"), dpi=200)
    plt.close()

    print(f"\nDone. Charts in {CHART_DIR}/, tables in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
