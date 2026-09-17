"""
data_cleaning.py
-----------------
Cleans and transforms the raw supply chain dataset:
 - handles missing values
 - removes duplicates / irrelevant rows
 - engineers features used later for EDA and ML

Run:
    python data_cleaning.py
Input:  data/supply_chain_raw.csv
Output: data/supply_chain_clean.csv
"""

import pandas as pd
import numpy as np

RAW_PATH = "data/supply_chain_raw.csv"
CLEAN_PATH = "data/supply_chain_clean.csv"


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["order_date", "expected_delivery_date", "actual_delivery_date"])
    print(f"Loaded {len(df)} rows, {df.shape[1]} columns")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset="order_id", keep="first")
    print(f"Removed {before - len(df)} duplicate rows")

    # Impute missing numeric values with column median (robust to outliers)
    numeric_cols = [
        "supplier_reliability_score", "weather_risk_index", "geopolitical_risk_index",
        "port_congestion_index", "inventory_level_units", "demand_forecast_units",
        "actual_demand_units", "order_value_usd"
    ]
    for col in numeric_cols:
        if df[col].isna().any():
            median_val = df[col].median()
            n_missing = df[col].isna().sum()
            df[col] = df[col].fillna(median_val)
            print(f"Filled {n_missing} missing values in '{col}' with median={median_val:.3f}")

    # Drop rows with impossible/irrelevant values
    df = df[df["order_value_usd"] > 0]
    df = df[df["inventory_level_units"] >= 0]

    return df.reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    # Recompute delay from dates to guarantee consistency
    df["delay_days"] = (df["actual_delivery_date"] - df["expected_delivery_date"]).dt.days
    df["delay_days"] = df["delay_days"].clip(lower=0)

    # Demand volatility: how far actual demand deviated from forecast
    df["demand_variance_pct"] = (
        (df["actual_demand_units"] - df["demand_forecast_units"]) / df["demand_forecast_units"]
    ) * 100

    # Inventory-to-demand coverage ratio (days of supply proxy)
    df["inventory_coverage_ratio"] = df["inventory_level_units"] / df["actual_demand_units"]

    # Composite external risk score
    df["composite_risk_score"] = (
        0.4 * (1 - df["supplier_reliability_score"])
        + 0.25 * df["weather_risk_index"]
        + 0.20 * df["geopolitical_risk_index"]
        + 0.15 * df["port_congestion_index"]
    ).round(3)

    # Calendar features
    df["order_month"] = df["order_date"].dt.month
    df["order_weekday"] = df["order_date"].dt.day_name()

    # Risk tier bucket, useful for dashboards
    df["risk_tier"] = pd.cut(
        df["composite_risk_score"],
        bins=[-0.01, 0.33, 0.66, 1.0],
        labels=["Low", "Medium", "High"],
    )

    return df


if __name__ == "__main__":
    df = load_data(RAW_PATH)
    df = clean_data(df)
    df = engineer_features(df)
    df.to_csv(CLEAN_PATH, index=False)
    print(f"\nSaved cleaned dataset -> {CLEAN_PATH}")
    print(df.describe(include="all").T[["count", "mean", "std"]].head(10) if False else "")
    print(f"Final shape: {df.shape}")
