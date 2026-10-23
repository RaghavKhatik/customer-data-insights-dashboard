"""
visualize.py
------------
Builds 10+ Matplotlib visualizations from the cleaned dataset to surface
trends and patterns: distributions, comparisons, and correlations.
"""

import matplotlib
matplotlib.use("Agg")  # render to files, no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT_DIR = "outputs"


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/{name}.png", dpi=130)
    plt.close(fig)
    print(f"Saved {OUT_DIR}/{name}.png")


def plot_age_histogram(df):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(df["age"], bins=25, color="#4C72B0", edgecolor="white")
    ax.set_title("Customer Age Distribution")
    ax.set_xlabel("Age")
    ax.set_ylabel("Count")
    save(fig, "01_age_histogram")


def plot_income_histogram(df):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(df["annual_income"], bins=30, color="#55A868", edgecolor="white")
    ax.set_title("Annual Income Distribution")
    ax.set_xlabel("Annual Income")
    ax.set_ylabel("Count")
    save(fig, "02_income_histogram")


def plot_purchase_histogram(df):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(df["purchase_amount"], bins=30, color="#C44E52", edgecolor="white")
    ax.set_title("Purchase Amount Distribution")
    ax.set_xlabel("Purchase Amount")
    ax.set_ylabel("Count")
    save(fig, "03_purchase_histogram")


def plot_category_bar(df):
    fig, ax = plt.subplots(figsize=(7, 5))
    counts = df["category"].value_counts()
    ax.bar(counts.index, counts.values, color="#8172B2")
    ax.set_title("Customer Count by Category")
    ax.set_ylabel("Number of Customers")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    save(fig, "04_category_bar")


def plot_gender_bar(df):
    fig, ax = plt.subplots(figsize=(6, 5))
    counts = df["gender"].value_counts()
    ax.bar(counts.index, counts.values, color="#CCB974")
    ax.set_title("Customer Count by Gender")
    ax.set_ylabel("Number of Customers")
    save(fig, "05_gender_bar")


def plot_avg_purchase_by_category(df):
    fig, ax = plt.subplots(figsize=(7, 5))
    avg = df.groupby("category")["purchase_amount"].mean().sort_values(ascending=False)
    ax.bar(avg.index, avg.values, color="#64B5CD")
    ax.set_title("Average Purchase Amount by Category")
    ax.set_ylabel("Avg Purchase Amount")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    save(fig, "06_avg_purchase_by_category")


def plot_satisfaction_by_category(df):
    fig, ax = plt.subplots(figsize=(7, 5))
    avg = df.groupby("category")["satisfaction_score"].mean().sort_values(ascending=False)
    ax.bar(avg.index, avg.values, color="#DD8452")
    ax.set_title("Average Satisfaction Score by Category")
    ax.set_ylabel("Avg Satisfaction (1-5)")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    save(fig, "07_satisfaction_by_category")


def plot_income_vs_purchase_scatter(df):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df["annual_income"], df["purchase_amount"], alpha=0.15, s=10, color="#4C72B0")
    ax.set_title("Annual Income vs. Purchase Amount")
    ax.set_xlabel("Annual Income")
    ax.set_ylabel("Purchase Amount")
    save(fig, "08_income_vs_purchase_scatter")


def plot_correlation_heatmap(df):
    numeric_df = df[["age", "annual_income", "purchase_amount", "satisfaction_score"]]
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=30, ha="right")
    ax.set_yticklabels(corr.columns)
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", color="black")
    ax.set_title("Correlation Heatmap (Numeric Features)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    save(fig, "09_correlation_heatmap")


def plot_age_by_gender_boxplot(df):
    fig, ax = plt.subplots(figsize=(7, 5))
    groups = [df.loc[df["gender"] == g, "age"] for g in df["gender"].unique()]
    ax.boxplot(groups, labels=df["gender"].unique())
    ax.set_title("Age Distribution by Gender")
    ax.set_ylabel("Age")
    save(fig, "10_age_by_gender_boxplot")


def plot_signups_over_time(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = pd.to_datetime(df["signup_date"]).dt.to_period("M").value_counts().sort_index()
    ax.plot(counts.index.astype(str), counts.values, marker="o", color="#4C72B0")
    ax.set_title("Customer Signups Over Time")
    ax.set_ylabel("New Signups")
    plt.setp(ax.get_xticklabels(), rotation=60, ha="right")
    save(fig, "11_signups_over_time")


def run_all(df):
    plot_age_histogram(df)
    plot_income_histogram(df)
    plot_purchase_histogram(df)
    plot_category_bar(df)
    plot_gender_bar(df)
    plot_avg_purchase_by_category(df)
    plot_satisfaction_by_category(df)
    plot_income_vs_purchase_scatter(df)
    plot_correlation_heatmap(df)
    plot_age_by_gender_boxplot(df)
    plot_signups_over_time(df)


if __name__ == "__main__":
    df = pd.read_csv("data/cleaned_customer_data.csv")
    run_all(df)
    print("\nAll 11 visualizations generated in outputs/")
