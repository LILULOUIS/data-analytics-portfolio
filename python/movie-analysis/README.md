# Movie Analysis — Python / Pandas

Adapted from AlexTheAnalyst/PortfolioProjects (https://github.com/AlexTheAnalyst/PortfolioProjects).

## What this shows

Exploratory analysis of 6,820 movies with Pandas, Seaborn, and Matplotlib: data-quality checks, outliers, correlations, top movies and top companies by gross, and gross by rating.

## Files

- `movie_analysis.py` — cleaned, runnable version of the source notebook. Run it with `python movie_analysis.py` after placing the dataset in `data/movies.csv`. It writes charts and summary tables to `outputs/`.
- `charts/` — charts built from the findings reported in the source notebook (see below).

## Dataset

`movies.csv` is **not included** in this repository. The source notebook reads it from a local Downloads folder, and it is not in the source repository either.

Expected file: `data/movies.csv`, 6,820 rows x 15 columns — budget, company, country, director, genre, gross, name, rating, released, runtime, score, star, votes, writer, year.

The source notebook reports 0% missing in those 15 columns. Two extra columns that appeared in that run (`grosstotal`, `countrygrosstotal`) were 99–100% missing and are not used here.

## Findings reported in the source notebook

These numbers come from the outputs stored in the source notebook, on the 6,820-movie dataset. They are reported here, not re-run on a new dataset yet.

- **Top movies by gross:** Star Wars: The Force Awakens ($936.7M), Avatar ($760.5M), Titanic ($658.7M), Jurassic World ($652.3M), The Avengers ($623.4M).
- **Top companies by gross:** Warner Bros. ($21.32B), Universal Pictures ($19.43B), Paramount Pictures ($17.12B), Twentieth Century Fox ($14.79B), Walt Disney Pictures ($10.46B).
- **Top company-year:** Paramount Pictures, 2011 ($1.49B); Universal Pictures, 2015 ($1.41B); Warner Bros., 2008 ($1.33B).
- **Pearson correlations (numeric columns only):** budget–gross 0.71, gross–votes 0.66, budget–votes 0.50, runtime–score 0.40, score–gross 0.17. A higher score alone does not predict gross.

## What I changed from the source

- Converted a 224-cell notebook with stored outputs into a clean `.py` script.
- Fixed the Windows-only file path to a relative `data/movies.csv`.
- Fixed `df.drop_duplicates()`, which was called without assignment in the source, so duplicates were never actually removed.
- Removed the deprecated `matplotlib.mlab` import.
- Removed the factorize-based correlation section. Factorizing categorical columns assigns arbitrary numeric codes, so correlations between those codes are not meaningful. Only numeric-column correlations are kept.

## Skills

Python, Pandas, NumPy, Seaborn, Matplotlib, exploratory data analysis, data cleaning, correlation analysis.
