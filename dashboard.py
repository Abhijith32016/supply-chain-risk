"""
dashboard.py
-----------------
Interactive dashboard for the Cloud-Based Supply Chain Disruption Risk
Analytics project, built with Streamlit. Reads the cleaned dataset and
trained model, and shows KPIs, risk visualizations, and a live
"predict disruption risk" tool.

Run locally:
    pip install streamlit plotly
    streamlit run dashboard.py

Deploy to Azure: see README.md Step 6 (Azure App Service / Container Apps).
"""

import pandas as pd
import numpy as np
import joblib
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Supply Chain Disruption Risk", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("data/supply_chain_clean.csv", parse_dates=["order_date"])

@st.cache_resource
def load_model():
    return joblib.load("models/disruption_model.joblib")

df = load_data()
bundle = load_model()
model, encoders, features = bundle["model"], bundle["encoders"], bundle["features"]

st.title("Cloud-Based Supply Chain Disruption Risk Analytics")
st.caption("Interactive dashboard — supplier reliability, delays, and predictive risk scoring")

# ---- Sidebar filters ----
st.sidebar.header("Filters")
regions = st.sidebar.multiselect("Region", sorted(df["region"].unique()), default=list(df["region"].unique()))
modes = st.sidebar.multiselect("Transport mode", sorted(df["transport_mode"].unique()), default=list(df["transport_mode"].unique()))
filtered = df[df["region"].isin(regions) & df["transport_mode"].isin(modes)]

# ---- KPI row ----
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total orders", f"{len(filtered):,}")
col2.metric("Disruption rate", f"{filtered['disruption_flag'].mean():.1%}")
col3.metric("Avg delay (days)", f"{filtered['delay_days'].mean():.1f}")
col4.metric("Avg composite risk", f"{filtered['composite_risk_score'].mean():.2f}")

st.divider()

# ---- Charts ----
c1, c2 = st.columns(2)

with c1:
    by_region = filtered.groupby("region")["disruption_flag"].mean().sort_values(ascending=False).reset_index()
    fig = px.bar(by_region, x="region", y="disruption_flag", title="Disruption rate by region",
                 labels={"disruption_flag": "Disruption rate"})
    st.plotly_chart(fig, width='stretch')

with c2:
    by_mode = filtered.groupby("transport_mode")["disruption_flag"].mean().sort_values(ascending=False).reset_index()
    fig = px.bar(by_mode, x="transport_mode", y="disruption_flag", title="Disruption rate by transport mode",
                 labels={"disruption_flag": "Disruption rate"}, color_discrete_sequence=["#D85A30"])
    st.plotly_chart(fig, width='stretch')

c3, c4 = st.columns(2)

with c3:
    monthly = filtered.groupby(filtered["order_date"].dt.to_period("M").astype(str))["disruption_flag"].mean().reset_index()
    fig = px.line(monthly, x="order_date", y="disruption_flag", markers=True,
                  title="Monthly disruption rate trend", labels={"order_date": "Month", "disruption_flag": "Disruption rate"})
    st.plotly_chart(fig, width='stretch')

with c4:
    fig = px.scatter(filtered.sample(min(1000, len(filtered))), x="supplier_reliability_score", y="delay_days",
                      color="risk_tier", title="Supplier reliability vs delivery delay",
                      labels={"supplier_reliability_score": "Supplier reliability", "delay_days": "Delay (days)"})
    st.plotly_chart(fig, width='stretch')

st.divider()

# ---- High risk suppliers table ----
st.subheader("Highest-risk suppliers")
risk_table = (
    filtered.groupby("supplier_id")
    .agg(orders=("order_id", "count"), disruption_rate=("disruption_flag", "mean"), avg_delay=("delay_days", "mean"))
    .query("orders >= 5")
    .sort_values("disruption_rate", ascending=False)
    .head(15)
    .reset_index()
)
st.dataframe(risk_table, width='stretch')

st.divider()

# ---- Live prediction tool ----
st.subheader("Predict disruption risk for a new order")
p1, p2, p3 = st.columns(3)
with p1:
    reliability = st.slider("Supplier reliability score", 0.0, 1.0, 0.75)
    weather = st.slider("Weather risk index", 0.0, 1.0, 0.3)
with p2:
    geopolitical = st.slider("Geopolitical risk index", 0.0, 1.0, 0.2)
    congestion = st.slider("Port congestion index", 0.0, 1.0, 0.15)
with p3:
    mode_choice = st.selectbox("Transport mode", sorted(df["transport_mode"].unique()))
    region_choice = st.selectbox("Region", sorted(df["region"].unique()))
    category_choice = st.selectbox("Product category", sorted(df["product_category"].unique()))

inventory_ratio = st.slider("Inventory coverage ratio", 0.0, 3.0, 1.1)
demand_variance = st.slider("Demand variance (%)", -50.0, 50.0, 0.0)
order_value = st.number_input("Order value (USD)", min_value=1.0, value=5000.0)

composite_risk = round(0.4 * (1 - reliability) + 0.25 * weather + 0.20 * geopolitical + 0.15 * congestion, 3)

if st.button("Predict"):
    row = pd.DataFrame([{
        "supplier_reliability_score": reliability,
        "weather_risk_index": weather,
        "geopolitical_risk_index": geopolitical,
        "port_congestion_index": congestion,
        "inventory_coverage_ratio": inventory_ratio,
        "demand_variance_pct": demand_variance,
        "composite_risk_score": composite_risk,
        "order_value_usd": order_value,
        "transport_mode": encoders["transport_mode"].transform([mode_choice])[0],
        "region": encoders["region"].transform([region_choice])[0],
        "product_category": encoders["product_category"].transform([category_choice])[0],
    }])[features]

    proba = model.predict_proba(row)[0, 1]
    st.metric("Predicted disruption probability", f"{proba:.1%}")
    if proba > 0.6:
        st.error("High risk — consider alternate supplier/route.")
    elif proba > 0.3:
        st.warning("Moderate risk — monitor closely.")
    else:
        st.success("Low risk.")
