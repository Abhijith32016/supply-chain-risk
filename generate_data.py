"""
generate_data.py
-----------------
Generates a realistic synthetic supply chain dataset for the
Cloud-Based Supply Chain Disruption Risk Analytics project.

If you have a real dataset (e.g. from Kaggle's "Supply Chain Shipment
Pricing Data" or "DataCo Smart Supply Chain"), skip this script and
point data_cleaning.py at your CSV instead.

Run:
    python generate_data.py
Output:
    data/supply_chain_raw.csv
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

N = 6000  # number of orders

suppliers = [f"SUP-{i:03d}" for i in range(1, 41)]
supplier_reliability = {s: np.random.beta(6, 2) for s in suppliers}  # 0-1, skewed reliable
transport_modes = ["Road", "Rail", "Sea", "Air"]
regions = ["North America", "Europe", "Asia Pacific", "Latin America", "Middle East/Africa"]
product_categories = ["Electronics", "Automotive Parts", "Pharma", "Textiles", "FMCG", "Machinery"]

rows = []
start_date = datetime(2023, 1, 1)

for i in range(N):
    order_id = f"ORD-{100000+i}"
    supplier = np.random.choice(suppliers)
    reliability = supplier_reliability[supplier]
    region = np.random.choice(regions)
    category = np.random.choice(product_categories)
    mode = np.random.choice(transport_modes, p=[0.45, 0.20, 0.25, 0.10])

    order_date = start_date + timedelta(days=int(np.random.uniform(0, 640)))

    # base lead time depends on transport mode
    base_lead = {"Road": 4, "Rail": 7, "Sea": 21, "Air": 3}[mode]
    expected_lead_days = base_lead + np.random.randint(-1, 3)
    expected_delivery = order_date + timedelta(days=int(expected_lead_days))

    # risk factors
    weather_risk = np.clip(np.random.normal(0.3, 0.2), 0, 1)
    geopolitical_risk = np.clip(np.random.normal(0.2, 0.15), 0, 1)
    port_congestion = np.clip(np.random.normal(0.25, 0.2), 0, 1) if mode == "Sea" else np.random.uniform(0, 0.1)

    # inventory & demand
    inventory_level = max(0, np.random.normal(500, 150))
    demand_forecast = max(1, np.random.normal(450, 120))
    demand_shock = np.random.normal(0, 0.15)  # unexpected demand swing
    actual_demand = max(0, demand_forecast * (1 + demand_shock))

    # probability of delay increases with low reliability & high risk factors
    disruption_prob = (
        0.35 * (1 - reliability)
        + 0.25 * weather_risk
        + 0.20 * geopolitical_risk
        + 0.15 * port_congestion
        + 0.10 * min(1, abs(demand_shock) * 2)
    )
    disruption_prob = np.clip(disruption_prob + np.random.normal(0, 0.05), 0, 1)
    disrupted = np.random.rand() < disruption_prob

    if disrupted:
        delay_days = int(np.random.gamma(shape=2.0, scale=4.0)) + 1
    else:
        delay_days = max(0, int(np.random.normal(0, 1)))

    actual_delivery = expected_delivery + timedelta(days=delay_days)
    order_value_usd = round(np.random.lognormal(mean=8.5, sigma=0.6), 2)

    rows.append({
        "order_id": order_id,
        "supplier_id": supplier,
        "supplier_reliability_score": round(reliability, 3),
        "region": region,
        "product_category": category,
        "transport_mode": mode,
        "order_date": order_date.date().isoformat(),
        "expected_delivery_date": expected_delivery.date().isoformat(),
        "actual_delivery_date": actual_delivery.date().isoformat(),
        "delay_days": delay_days,
        "weather_risk_index": round(weather_risk, 3),
        "geopolitical_risk_index": round(geopolitical_risk, 3),
        "port_congestion_index": round(port_congestion, 3),
        "inventory_level_units": round(inventory_level, 1),
        "demand_forecast_units": round(demand_forecast, 1),
        "actual_demand_units": round(actual_demand, 1),
        "order_value_usd": order_value_usd,
        "disruption_flag": int(disrupted),
    })

df = pd.DataFrame(rows)

# inject some realistic messiness: missing values & duplicate rows
missing_idx = np.random.choice(df.index, size=int(0.02 * N), replace=False)
df.loc[missing_idx, "weather_risk_index"] = np.nan

missing_idx2 = np.random.choice(df.index, size=int(0.015 * N), replace=False)
df.loc[missing_idx2, "supplier_reliability_score"] = np.nan

dup_rows = df.sample(20, random_state=1)
df = pd.concat([df, dup_rows], ignore_index=True)

df.to_csv("data/supply_chain_raw.csv", index=False)
print(f"Generated {len(df)} rows -> data/supply_chain_raw.csv")
print(f"Disruption rate: {df['disruption_flag'].mean():.2%}")
