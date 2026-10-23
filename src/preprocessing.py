"""
preprocessing.py
-----------------
ML preprocessing utilities: encoding categorical variables and normalizing
numeric features, so the cleaned dataset is ready to feed into a model.
"""

import pandas as pd


def one_hot_encode(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """One-hot encode nominal categorical columns (e.g. category, gender)."""
    return pd.get_dummies(df, columns=columns, drop_first=True)


def min_max_normalize(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Normalize numeric columns to [0, 1] range."""
    df = df.copy()
    for col in columns:
        min_val, max_val = df[col].min(), df[col].max()
        if max_val > min_val:
            df[col] = (df[col] - min_val) / (max_val - min_val)
        else:
            df[col] = 0.0
    return df


def prepare_for_modeling(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full preprocessing pipeline: encode categoricals, normalize numerics.
    Returns a model-ready DataFrame.
    """
    df = df.copy()
    df = df.drop(columns=["customer_id", "signup_date"], errors="ignore")

    numeric_cols = ["age", "annual_income", "purchase_amount", "satisfaction_score"]
    categorical_cols = ["gender", "category"]

    df = min_max_normalize(df, numeric_cols)
    df = one_hot_encode(df, categorical_cols)

    return df


if __name__ == "__main__":
    cleaned = pd.read_csv("data/cleaned_customer_data.csv")
    model_ready = prepare_for_modeling(cleaned)
    model_ready.to_csv("data/model_ready_data.csv", index=False)
    print(f"Model-ready dataset shape: {model_ready.shape}")
    print(f"Columns: {list(model_ready.columns)}")
    print("Saved to data/model_ready_data.csv")
