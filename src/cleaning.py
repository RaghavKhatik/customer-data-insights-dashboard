"""
cleaning.py
-----------
Reusable, importable data cleaning utilities.

Design goal (matches the resume bullet): these functions are written to be
reused across datasets/columns rather than hard-coded one-off scripts, which
is what actually cuts down repeat cleaning time on future projects.
"""

import numpy as np
import pandas as pd


def standardize_text_column(series: pd.Series) -> pd.Series:
    """Lowercase, strip whitespace, and title-case a messy categorical column."""
    return series.astype(str).str.strip().str.lower().replace("nan", np.nan).str.title()


def standardize_gender_column(series: pd.Series) -> pd.Series:
    mapping = {
        "m": "Male", "male": "Male",
        "f": "Female", "female": "Female",
    }
    cleaned = series.astype(str).str.strip().str.lower().map(mapping)
    return cleaned


def parse_mixed_dates(series: pd.Series) -> pd.Series:
    """Handle a column containing a mix of ISO and DD/MM/YYYY date formats."""
    def _parse(val):
        for fmt in ("%Y-%m-%d", "%d/%m/%Y"):
            try:
                return pd.to_datetime(val, format=fmt)
            except (ValueError, TypeError):
                continue
        return pd.NaT
    return series.apply(_parse)


def cap_outliers_iqr(series: pd.Series, k: float = 1.5) -> pd.Series:
    """
    Cap outliers using the IQR method (a defensible, explainable choice
    an interviewer will likely ask about vs. simple z-score trimming).
    """
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr
    return series.clip(lower=lower, upper=upper)


def fill_missing_numeric(series: pd.Series, strategy: str = "median") -> pd.Series:
    if strategy == "median":
        return series.fillna(series.median())
    elif strategy == "mean":
        return series.fillna(series.mean())
    raise ValueError(f"Unknown strategy: {strategy}")


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pipeline. Returns a cleaned copy; does not mutate input.
    Tracks and prints how many inconsistencies were fixed, to quantify impact.
    """
    df = df.copy()
    n_before_issues = (
        df.isna().sum().sum()
        + (df["age"].lt(0) | df["age"].gt(110)).sum()
    )

    # 1. Drop exact duplicates
    df = df.drop_duplicates()

    # 2. Fix invalid ages (data entry errors), then cap remaining outliers
    df.loc[(df["age"] < 0) | (df["age"] > 110), "age"] = np.nan
    df["age"] = fill_missing_numeric(df["age"], "median")
    df["age"] = cap_outliers_iqr(df["age"])

    # 3. Standardize categorical text fields
    df["gender"] = standardize_gender_column(df["gender"])
    df["gender"] = df["gender"].fillna("Unknown")

    df["category"] = standardize_text_column(df["category"])
    df["category"] = df["category"].fillna("Unknown")

    # 4. Handle numeric outliers + missing values
    df["annual_income"] = cap_outliers_iqr(df["annual_income"])
    df["annual_income"] = fill_missing_numeric(df["annual_income"], "median")

    df["purchase_amount"] = cap_outliers_iqr(df["purchase_amount"])
    df["purchase_amount"] = fill_missing_numeric(df["purchase_amount"], "median")

    df["satisfaction_score"] = fill_missing_numeric(df["satisfaction_score"], "median")

    # 5. Parse inconsistent date formats
    df["signup_date"] = parse_mixed_dates(df["signup_date"])

    n_after_issues = df.isna().sum().sum()

    reduction_pct = (
        0 if n_before_issues == 0
        else round((1 - n_after_issues / n_before_issues) * 100, 1)
    )
    print(f"Inconsistencies before cleaning: {int(n_before_issues)}")
    print(f"Inconsistencies after cleaning:  {int(n_after_issues)}")
    print(f"Reduction: ~{reduction_pct}%")

    return df


if __name__ == "__main__":
    raw = pd.read_csv("data/raw_customer_data.csv")
    cleaned = clean_dataset(raw)
    cleaned.to_csv("data/cleaned_customer_data.csv", index=False)
    print(f"\nSaved cleaned dataset: {cleaned.shape[0]} rows -> data/cleaned_customer_data.csv")
