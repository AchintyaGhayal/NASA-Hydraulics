# train_and_save.py
# Run locally once: python train_and_save.py
# Produces: hydraulics_eff_model.pkl and hydraulics_co2_model.pkl

import pandas as pd
import joblib
from pathlib import Path
from hydraulics import train_task  # your function in hydraulics.py

DATA_PATH = Path("hydraulics_ops_sustainability_dataset.csv")

# --- Load data ---
df = pd.read_csv(DATA_PATH)

# --- Define feature columns based on your CSV ---
# Categorical features (string-like columns)
categorical_cols = [
    "region",
    "client_type",
    "product_line",
    "material",
    "control_type",
    "sensor_pack",
    "oil_type",
]

# Numeric features (exclude target columns)
numeric_cols = [
    "filtration_rating_micron",
    "oil_change_interval_hours",
    "test_pressure_bar",
    "test_duration_min",
    "energy_used_kwh",
    "heat_loss_kwh",
    "leak_rate_ml_min",
    "spill_incident",
    "warranty_return",
    "parts_cost_usd",
    "labor_hours",
    "labor_cost_usd",
    "total_cost_usd",
    "turnaround_days",
]

# Optional: if your pipeline handles dates/IDs specially, keep them out of features
# job_id and date are intentionally excluded.

# --- Shared model params for HistGradientBoostingRegressor (safe defaults) ---
params = {
    # tune later if you want; these work well for first pass
    "learning_rate": 0.05,
    "max_depth": 6,
    "max_iter": 500,
    "l2_regularization": 0.0,
}

# --- Train Efficiency model ---
eff_target = "efficiency_pct"
eff_model = train_task(
    df=df,
    name="efficiency_model",
    numeric_cols=numeric_cols,
    categorical_cols=categorical_cols,
    target_col=eff_target,
    params=params,
)
joblib.dump(eff_model, "hydraulics_eff_model.pkl")
print("✅ Saved hydraulics_eff_model.pkl")

# --- Train CO2 model ---
co2_target = "co2_kg"
co2_model = train_task(
    df=df,
    name="co2_model",
    numeric_cols=numeric_cols,
    categorical_cols=categorical_cols,
    target_col=co2_target,
    params=params,
)
joblib.dump(co2_model, "hydraulics_co2_model.pkl")
print("✅ Saved hydraulics_co2_model.pkl")