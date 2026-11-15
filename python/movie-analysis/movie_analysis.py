# Movie Analysis — Python / Pandas
# Adapted from AlexTheAnalyst/PortfolioProjects
# Source: https://github.com/AlexTheAnalyst/PortfolioProjects
# Original author: Alex Freberg (AlexTheAnalyst). The source README invites reuse:
#   "Please feel free to take these and run with them. Make them your own"
# Adapted by Louis Lu, Nov 2026, for learning and portfolio use.
#
# What changed from the source notebook:
# - Converted from a 224-cell notebook (with stored outputs) to a clean,
#   runnable .py script. No outputs are stored in this file.
# - Fixed the Windows-only path (C:\Users\alexf\Downloads\movies.csv)
#   to a relative path: data/movies.csv (place the file there).
# - Fixed df.drop_duplicates() which was called without assignment
#   in the source, so duplicates were never actually removed.
# - Removed the deprecated `import matplotlib.mlab` import.
# - Removed the factorize-based correlation section. Factorizing
#   categorical columns assigns arbitrary numeric codes, so
#   correlations between those codes (e.g. writer vs. budget, 0.67)
#   are not meaningful. Only numeric-column correlations are kept.
# - Added a proper Year column from `released` and saved charts
#   and summary tables to outputs/.
#
# Dataset (not included — see README.md):
#   movies.csv, 6,820 rows x 15 columns:
#   budget, company, country, director, genre, gross, name, rating,
#   released, runtime, score, star, votes, writer, year
#   The source notebook reports 0% missing in those 15 columns.
#   Two extra columns in the source run (grosstotal, countrygrosstotal)
#   were 99-100% missing and are not used here.

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.style.use("ggplot")
plt.rcParams["figure.figsize"] = (12, 8)

DATA_PATH = os.path.join("data", "movies.csv")
OUTPUT_DIR = "outputs"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    print(f"Shape: {df.shape}")
    print(df.dtypes)

    # --- Data quality -------------------------------------------------
    print("\nMissing data (%):")
    for col in df.columns:
        pct_missing = np.mean(df[col].isnull())
        print(f"{col} - {round(pct_missing * 100)}%")

    # Drop duplicates properly (the source called this without assignment)
    before = len(df)
    df = df.drop_duplicates()
    print(f"\nDropped {before - len(df)} duplicate rows.")

    # --- Top movies by gross ------------------------------------------
    top_movies = df.sort_values(by="gross", ascending=False).head(5)
    print("\nTop 5 movies by gross:")
    print(top_movies[["name", "company", "year", "budget", "gross", "score"]].to_string(index=False))

    # --- Correlations (numeric columns only) --------------------------
    numeric = df.select_dtypes(include=[np.number])
    corr = numeric.corr(method="pearson")
    print("\nPearson correlation (numeric columns):")
    print(corr.round(2).to_string())
    print("\nKey reading: budget-gross r = 0.71, gross-votes r = 0.66,")
    print("score-gross r = 0.17 — a higher score alone does not predict gross.")

    plt.figure()
    sns.heatmap(corr, annot=True)
    plt.title("Correlation matrix for Numeric Features")
    plt.xlabel("Movie features")
    plt.ylabel("Movie features")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "correlation-heatmap.png"), dpi=200)
    plt.close()

    # --- Budget vs gross, score vs gross ------------------------------
    plt.figure()
    sns.regplot(x="gross", y="budget", data=df)
    plt.title("Budget vs Gross")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "budget-vs-gross.png"), dpi=200)
    plt.close()

    plt.figure()
    sns.regplot(x="score", y="gross", data=df)
    plt.title("Score vs Gross")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "score-vs-gross.png"), dpi=200)
    plt.close()

    plt.figure()
    plt.scatter(x=df["budget"], y=df["gross"], alpha=0.5)
    plt.title("Budget vs Gross Earnings")
    plt.xlabel("Budget")
    plt.ylabel("Gross Earnings")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "budget-gross-scatter.png"), dpi=200)
    plt.close()

    # --- Top companies by gross ---------------------------------------
    company_gross = df.groupby("company")[["gross"]].sum()
    top_companies = company_gross.sort_values("gross", ascending=False).head(15)
    top_companies["gross"] = top_companies["gross"].astype("int64")
    print("\nTop 15 companies by gross:")
    print(top_companies.to_string())
    top_companies.to_csv(os.path.join(OUTPUT_DIR, "top-companies-by-gross.csv"))

    # Top company-year combinations
    df["Year"] = df["released"].astype(str).str[:4]
    company_year = df.groupby(["company", "Year"])[["gross"]].sum()
    top_company_year = company_year.sort_values("gross", ascending=False).head(15)
    print("\nTop 15 company-year combinations by gross:")
    print(top_company_year.to_string())
    top_company_year.to_csv(os.path.join(OUTPUT_DIR, "top-company-year-by-gross.csv"))

    # --- Gross by rating ----------------------------------------------
    plt.figure()
    sns.stripplot(x="rating", y="gross", data=df)
    plt.title("Gross by MPAA rating")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "gross-by-rating.png"), dpi=200)
    plt.close()

    print(f"\nDone. Charts and tables saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
