"""
eda_analysis.py
-----------------
Exploratory and statistical analysis of the cleaned supply chain data.
Generates summary stats, correlation matrix, and PNG charts used in the
report / dashboard.

Run:
    python eda_analysis.py
Input:  data/supply_chain_clean.csv
Output: outputs/*.png, outputs/eda_summary.txt
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

df = pd.read_csv("data/supply_chain_clean.csv", parse_dates=["order_date"])

summary_lines = []

def log(line=""):
    print(line)
    summary_lines.append(str(line))

log("=== SUPPLY CHAIN DISRUPTION RISK — EDA SUMMARY ===\n")

# 1. Overall disruption rate
disruption_rate = df["disruption_flag"].mean()
log(f"Overall disruption rate: {disruption_rate:.2%}")

# 2. Disruption rate by transport mode
by_mode = df.groupby("transport_mode")["disruption_flag"].mean().sort_values(ascending=False)
log("\nDisruption rate by transport mode:")
log(by_mode.round(3).to_string())

# 3. Disruption rate by region
by_region = df.groupby("region")["disruption_flag"].mean().sort_values(ascending=False)
log("\nDisruption rate by region:")
log(by_region.round(3).to_string())

# 4. Supplier reliability vs disruption correlation
corr_reliability = df["supplier_reliability_score"].corr(df["disruption_flag"])
log(f"\nCorrelation (supplier_reliability_score vs disruption_flag): {corr_reliability:.3f}")

# 5. Average delay by risk tier
by_tier = df.groupby("risk_tier", observed=True)["delay_days"].mean()
log("\nAverage delay (days) by composite risk tier:")
log(by_tier.round(2).to_string())

# 6. Top 10 highest-risk suppliers
top_suppliers = (
    df.groupby("supplier_id")
    .agg(orders=("order_id", "count"), disruption_rate=("disruption_flag", "mean"),
         avg_delay=("delay_days", "mean"))
    .query("orders >= 20")
    .sort_values("disruption_rate", ascending=False)
    .head(10)
)
log("\nTop 10 highest-risk suppliers (min 20 orders):")
log(top_suppliers.round(3).to_string())

with open("outputs/eda_summary.txt", "w") as f:
    f.write("\n".join(summary_lines))

# ---- Charts ----

# Correlation heatmap
numeric_cols = [
    "supplier_reliability_score", "weather_risk_index", "geopolitical_risk_index",
    "port_congestion_index", "inventory_coverage_ratio", "demand_variance_pct",
    "composite_risk_score", "delay_days", "disruption_flag"
]
plt.figure(figsize=(9, 7))
sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation matrix — risk factors vs disruption")
plt.tight_layout()
plt.savefig("outputs/correlation_heatmap.png", dpi=150)
plt.close()

# Disruption rate by transport mode
plt.figure(figsize=(7, 5))
by_mode.plot(kind="bar", color="#0F6E56")
plt.ylabel("Disruption rate")
plt.title("Disruption rate by transport mode")
plt.tight_layout()
plt.savefig("outputs/disruption_by_mode.png", dpi=150)
plt.close()

# Delay distribution
plt.figure(figsize=(7, 5))
sns.histplot(df["delay_days"], bins=30, kde=True, color="#D85A30")
plt.title("Distribution of delivery delay (days)")
plt.tight_layout()
plt.savefig("outputs/delay_distribution.png", dpi=150)
plt.close()

# Monthly disruption trend
monthly = df.groupby(df["order_date"].dt.to_period("M"))["disruption_flag"].mean()
plt.figure(figsize=(9, 5))
monthly.plot(marker="o", color="#185FA5")
plt.ylabel("Disruption rate")
plt.title("Monthly disruption rate trend")
plt.tight_layout()
plt.savefig("outputs/monthly_trend.png", dpi=150)
plt.close()

print("\nSaved charts to outputs/ and summary to outputs/eda_summary.txt")
