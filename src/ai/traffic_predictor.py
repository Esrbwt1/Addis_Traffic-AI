import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import sys

"""
Addis Ababa Traffic AI (Multi-Day Training V3)
----------------------------------------------
Trains on 30 days of synthetic data derived from the Digital Twin.
- Train Set: Days 1-25
- Test Set: Days 26-30 (Unseen Future)
- Technique: Chronological Splitting (No Shuffling)
"""

# --- CONFIG ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))

# Points to the new 30-day dataset
DATA_FILE = os.path.join(PROJECT_ROOT, "data", "raw", "synthetic_traffic_30days.csv")
MODEL_FILE = os.path.join(PROJECT_ROOT, "data", "models", "traffic_model.pkl")


def train_brain():
    print("🧠 Initializing AI Training Protocol (Multi-Day Mode)...")

    if not os.path.exists(DATA_FILE):
        print("❌ Error: Synthetic data not found.")
        print("   Run 'src/utils/generate_synthetic_data.py' first!")
        sys.exit(1)

    df = pd.read_csv(DATA_FILE)
    print(f"📊 Loaded {len(df)} data points ({df['day'].nunique()} Days).")

    # --- FEATURE ENGINEERING ---
    HORIZON = 300

    # Group by 'day' prevents shifting data from Day 1 into Day 2
    # This keeps the math clean.
    df["target"] = df.groupby("day")["vehicle_count"].shift(-HORIZON)
    df["lag_1min"] = df.groupby("day")["vehicle_count"].shift(60)
    df["lag_5min"] = df.groupby("day")["vehicle_count"].shift(300)

    df_clean = df.dropna()

    X = df_clean[["step", "vehicle_count", "avg_speed", "lag_1min", "lag_5min"]]
    y = df_clean["target"]

    # --- CHRONOLOGICAL SPLIT (The "Pro" Way) ---
    # We train on Days 1-25. We test on Days 26-30.
    # The AI has never seen the Test days before.
    split_day = 25

    train_mask = df_clean["day"] <= split_day
    test_mask = df_clean["day"] > split_day

    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]

    print(f"📉 Training Set: {len(X_train)} points (Days 1-{split_day})")
    print(f"📉 Testing Set:  {len(X_test)} points (Days {split_day + 1}-30)")

    print("🏋️  Training Random Forest...")
    # NOTE: shuffle=False is implied because we manually split the data
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # --- EVALUATION ---
    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)

    print("-" * 35)
    print(f"🏆 AI PERFORMANCE REPORT (Valid Time Series)")
    print(f"   Accuracy (R² Score): {r2:.4f}")
    print(f"   Mean Squared Error: {mse:.2f}")
    print("-" * 35)

    joblib.dump(model, MODEL_FILE)
    print(f"💾 Robust Model saved to: {MODEL_FILE}")


if __name__ == "__main__":
    train_brain()
