# hydraulics.py
# Train two models (Efficiency, CO2), save metrics & artifacts.
# Run: python hydraulics.py

import json
import joblib
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

warnings.filterwarnings("ignore")

CSV_FILE = "hydraulics_ops_sustainability_dataset.csv"
SEED = 42
ARTIFACTS = Path("artifacts")
ARTIFACTS.mkdir(exist_ok=True)

REQ_COLS = [
    "filtration_rating_micron", "oil_change_interval_hours",
    "test_pressure_bar", "test_duration_min", "leak_rate_ml_min",
    "region", "client_type", "product_line", "material", "control_type", "sensor_pack", "oil_type",
    "efficiency_pct", "energy_used_kwh", "co2_kg", "heat_loss_kwh"
]

NUM_EFF = ["filtration_rating_micron","oil_change_interval_hours","test_pressure_bar","test_duration_min","leak_rate_ml_min"]
CAT_COMMON = ["region","client_type","product_line","material","control_type","sensor_pack","oil_type"]
NUM_CO2 = ["energy_used_kwh","filtration_rating_micron","test_pressure_bar","test_duration_min","efficiency_pct"]

def load_data() -> pd.DataFrame:
    df = pd.read_csv(CSV_FILE)
    missing = [c for c in REQ_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    # Make categoricals explicit
    for c in CAT_COMMON:
        df[c] = df[c].astype("category")

    return df

def build_preprocessor(numeric_cols, categorical_cols) -> ColumnTransformer:
    # Tree models don't need scaling; we impute only.
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    return ColumnTransformer(
        [("num", num_pipe, numeric_cols),
         ("cat", cat_pipe, categorical_cols)],
        remainder="drop"
    )

def kfold_report(model: Pipeline, X: pd.DataFrame, y: pd.Series, name: str):
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    r2 = cross_val_score(model, X, y, cv=cv, scoring="r2")
    mae = -cross_val_score(model, X, y, cv=cv, scoring="neg_mean_absolute_error")
    print(f"[{name}] CV 5-fold — R²: {r2.mean():.3f} ± {r2.std():.3f} | MAE: {mae.mean():.3f} ± {mae.std():.3f}")
    return {"r2_mean": float(r2.mean()), "r2_std": float(r2.std()),
            "mae_mean": float(mae.mean()), "mae_std": float(mae.std())}

def holdout_metrics(model: Pipeline, X: pd.DataFrame, y: pd.Series):
    yhat = model.predict(X)
    return {
        "r2": float(r2_score(y, yhat)),
        "mae": float(mean_absolute_error(y, yhat)),
        "rmse": float(np.sqrt(mean_squared_error(y, yhat))),
    }

def train_task(df: pd.DataFrame, name: str, numeric_cols, categorical_cols, target_col: str, params: dict):
    X = df[numeric_cols + categorical_cols].copy()
    y = df[target_col].copy()

    pre = build_preprocessor(numeric_cols, categorical_cols)
    model = HistGradientBoostingRegressor(random_state=SEED, **params)
    pipe = Pipeline([("pre", pre), ("hgb", model)])

    # CV report
    cv_summary = kfold_report(pipe, X, y, name)

    # Holdout split for a concrete test number
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)
    pipe.fit(Xtr, ytr)

    m_tr = holdout_metrics(pipe, Xtr, ytr)
    m_te = holdout_metrics(pipe, Xte, yte)

    # Persist
    out_path = ARTIFACTS / f"model_{name.lower()}.joblib"
    joblib.dump(pipe, out_path)
    with open(ARTIFACTS / f"feature_names_{name.lower()}.json", "w") as f:
        json.dump({"numeric": numeric_cols, "categorical": categorical_cols, "target": target_col}, f)

    print(f"✅ Saved {name} model → {out_path}")
    return {"cv": cv_summary, "train": m_tr, "test": m_te}

def main():
    print(f"Loading data from {CSV_FILE} ...")
    df = load_data()
    print(f"Loaded {len(df):,} rows.")

    metrics = {}

    metrics["efficiency"] = train_task(
        df, name="Efficiency", numeric_cols=NUM_EFF, categorical_cols=CAT_COMMON,
        target_col="efficiency_pct",
        params=dict(learning_rate=0.08, max_depth=6, max_iter=400, l2_regularization=0.02, early_stopping=True)
    )

    metrics["co2"] = train_task(
        df, name="CO2", numeric_cols=NUM_CO2, categorical_cols=CAT_COMMON,
        target_col="co2_kg",
        params=dict(learning_rate=0.07, max_depth=6, max_iter=450, l2_regularization=0.04, early_stopping=True)
    )

    (ARTIFACTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("📊 Metrics saved → artifacts/metrics.json")
    print("✅ Done.")

if __name__ == "__main__":
    main()