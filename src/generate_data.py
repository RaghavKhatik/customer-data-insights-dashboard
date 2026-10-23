"""
generate_data.py
-----------------
Generates a synthetic "raw" e-commerce style dataset with realistic messiness:
missing values, outliers, inconsistent formatting, and duplicate-like noise.

This simulates the kind of raw dataset a Data Analyst would receive before
cleaning -- used to demonstrate the cleaning + visualization pipeline.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N = 10500  # 10,000+ records as per project spec


def generate_raw_dataset(n=N):
    customer_ids = np.arange(1, n + 1)

    ages = np.random.normal(35, 12, n).round()
    # Inject unrealistic outliers (data entry errors)
    outlier_idx = np.random.choice(n, size=int(n * 0.01), replace=False)
    ages[outlier_idx] = np.random.choice([-5, 0, 150, 200], size=len(outlier_idx))

    genders = np.random.choice(
        ["Male", "Female", "M", "F", "male", "female", np.nan],
        size=n,
        p=[0.28, 0.28, 0.1, 0.1, 0.1, 0.1, 0.04],
    )

    income = np.random.normal(55000, 20000, n)
    income[np.random.choice(n, size=int(n * 0.015), replace=False)] *= 8  # outliers
    income[np.random.choice(n, size=int(n * 0.05), replace=False)] = np.nan

    purchase_amount = np.abs(np.random.exponential(120, n))
    purchase_amount[np.random.choice(n, size=int(n * 0.02), replace=False)] = np.nan

    categories = np.random.choice(
        ["Electronics", "electronics", "Clothing", "Clothing ", "Grocery",
         "grocery", "Home & Kitchen", np.nan],
        size=n,
        p=[0.2, 0.05, 0.2, 0.05, 0.2, 0.05, 0.2, 0.05],
    )

    signup_dates = pd.to_datetime("2022-01-01") + pd.to_timedelta(
        np.random.randint(0, 900, n), unit="D"
    )
    # Introduce inconsistent date string formats for a subset
    signup_dates = signup_dates.astype(str)
    mixed_fmt_idx = np.random.choice(n, size=int(n * 0.1), replace=False)
    signup_dates = pd.Series(signup_dates)
    signup_dates.iloc[mixed_fmt_idx] = pd.to_datetime(
        signup_dates.iloc[mixed_fmt_idx]
    ).dt.strftime("%d/%m/%Y")

    satisfaction_score = np.random.randint(1, 6, n).astype(float)
    satisfaction_score[np.random.choice(n, size=int(n * 0.03), replace=False)] = np.nan

    df = pd.DataFrame({
        "customer_id": customer_ids,
        "age": ages,
        "gender": genders,
        "annual_income": income.round(2),
        "purchase_amount": purchase_amount.round(2),
        "category": categories,
        "signup_date": signup_dates,
        "satisfaction_score": satisfaction_score,
    })

    # Add a handful of fully duplicated rows (common real-world issue)
    dup_rows = df.sample(50, random_state=1)
    df = pd.concat([df, dup_rows], ignore_index=True)

    return df


if __name__ == "__main__":
    df = generate_raw_dataset()
    df.to_csv("data/raw_customer_data.csv", index=False)
    print(f"Generated raw dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Saved to data/raw_customer_data.csv")
