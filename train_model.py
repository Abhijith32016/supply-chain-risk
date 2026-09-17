"""
train_model.py
-----------------
Trains a machine learning model to classify/predict the likelihood of a
supply chain disruption based on historical order data.

Run:
    python train_model.py
Input:  data/supply_chain_clean.csv
Output: models/disruption_model.joblib, outputs/feature_importance.png,
        outputs/model_metrics.txt, outputs/confusion_matrix.png
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)

df = pd.read_csv("data/supply_chain_clean.csv")

FEATURES = [
    "supplier_reliability_score", "weather_risk_index", "geopolitical_risk_index",
    "port_congestion_index", "inventory_coverage_ratio", "demand_variance_pct",
    "composite_risk_score", "order_value_usd", "transport_mode", "region",
    "product_category",
]
TARGET = "disruption_flag"

data = df[FEATURES + [TARGET]].copy()

# Encode categorical columns
encoders = {}
for col in ["transport_mode", "region", "product_category"]:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    encoders[col] = le

X = data[FEATURES]
y = data[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

report = classification_report(y_test, y_pred, target_names=["No disruption", "Disruption"])
auc = roc_auc_score(y_test, y_proba)

print(report)
print(f"ROC-AUC: {auc:.3f}")

with open("outputs/model_metrics.txt", "w") as f:
    f.write("=== DISRUPTION PREDICTION MODEL — METRICS ===\n\n")
    f.write(report)
    f.write(f"\nROC-AUC: {auc:.3f}\n")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion matrix")
plt.colorbar()
for (i, j), val in np.ndenumerate(cm):
    plt.text(j, i, str(val), ha="center", va="center")
plt.xticks([0, 1], ["No disruption", "Disruption"])
plt.yticks([0, 1], ["No disruption", "Disruption"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png", dpi=150)
plt.close()

# Feature importance
importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
plt.figure(figsize=(8, 6))
importances.plot(kind="barh", color="#534AB7")
plt.title("Feature importance — disruption risk model")
plt.tight_layout()
plt.savefig("outputs/feature_importance.png", dpi=150)
plt.close()

# Save model + encoders together
joblib.dump({"model": model, "encoders": encoders, "features": FEATURES}, "models/disruption_model.joblib")
print("\nSaved model -> models/disruption_model.joblib")
print("Saved metrics/plots -> outputs/")
