"""
Replicates Walmart_Data_Preprocessing.ipynb:
- load raw Walmart.csv
- clean / fix dtypes
- feature engineering (calendar, lag, rolling features)
- time-based train/test split
Outputs: data/train_features.csv, data/test_features.csv, data/full_features.csv
"""
import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

TEST_SPLIT_FRACTION = 0.2


def run():
    df = pd.read_csv(os.path.join(DATA_DIR, "Walmart.csv"))

    df = df.drop_duplicates()

    df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")
    df = df.sort_values(["Store", "Date"]).reset_index(drop=True)

    df["Weekly_Sales"] = df["Weekly_Sales"].clip(lower=0)

    # Calendar features
    df["month"] = df["Date"].dt.month
    df["week_of_year"] = df["Date"].dt.isocalendar().week.astype(int)
    df["year"] = df["Date"].dt.year
    df["quarter"] = df["Date"].dt.quarter

    # Lag features
    for lag in [1, 4, 8, 12]:
        df[f"sales_lag_{lag}"] = df.groupby("Store")["Weekly_Sales"].shift(lag)

    # Rolling features (shift(1) first so this week's own sales never leaks)
    for window in [4, 8, 12]:
        df[f"rolling_mean_{window}"] = (
            df.groupby("Store")["Weekly_Sales"].shift(1).rolling(window).mean().reset_index(level=0, drop=True)
        )
        df[f"rolling_std_{window}"] = (
            df.groupby("Store")["Weekly_Sales"].shift(1).rolling(window).std().reset_index(level=0, drop=True)
        )

    # Save the full engineered dataset (with NaNs) for the app to use for
    # per-store "latest known state" lookups before we drop early rows.
    df.to_csv(os.path.join(DATA_DIR, "full_features_raw.csv"), index=False)

    df_clean = df.dropna(subset=["sales_lag_12", "rolling_mean_12"])

    min_date, max_date = df_clean["Date"].min(), df_clean["Date"].max()
    split_date = min_date + (max_date - min_date) * (1 - TEST_SPLIT_FRACTION)

    train_df = df_clean[df_clean["Date"] < split_date]
    test_df = df_clean[df_clean["Date"] >= split_date]

    train_df.to_csv(os.path.join(DATA_DIR, "train_features.csv"), index=False)
    test_df.to_csv(os.path.join(DATA_DIR, "test_features.csv"), index=False)
    df_clean.to_csv(os.path.join(DATA_DIR, "full_features.csv"), index=False)

    print("Split date  :", split_date.date())
    print("Train rows  :", train_df.shape[0])
    print("Test rows   :", test_df.shape[0])
    print("Saved train_features.csv, test_features.csv, full_features.csv")


if __name__ == "__main__":
    run()
