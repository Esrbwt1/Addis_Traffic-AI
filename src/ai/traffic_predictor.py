import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import sys

"""
Addis Ababa Traffic AI (Predictor Module V2)
--------------------------------------------
Trains a Machine Learning model to predict future traffic density.

Key Logic Updates:
1. Temporal Awareness: Added 'step' (Time) as a feature.
   - Traffic is time-dependent. 100 cars at 9 AM != 100 cars at 9 PM.
2. Lag Features: Uses past data (t-1, t-5) to detect trends (rising vs falling).
"""

# --- PATHS ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
DATA_FILE = os.path.join(PROJECT_ROOT, "data", "raw", "traffic_log.csv")
MODEL_FILE = os.path.join(PROJECT_ROOT, "data", "models", "traffic_model.pkl")


def train_brain():
    print("🧠 Initializing AI Training Protocol...")

    if not os.path.exists(DATA_FILE):
        print("❌ Error: No data found. Run the simulation first!")
        sys.exit(1)

    df = pd.read_csv(DATA_FILE)
    print(f"📊 Loaded {len(df)} data points from simulation logs.")

    # --- FEATURE ENGINEERING ---
    # Prediction Horizon: 5 minutes (300 seconds)
    HORIZON = 300

    # Target: The future vehicle count
    df["target"] = df["vehicle_count"].shift(-HORIZON)

    # Lag Features: The "Short Term Memory" of the AI
    df["lag_1min"] = df["vehicle_count"].shift(60)
    df["lag_5min"] = df["vehicle_count"].shift(300)

    # Clean up empty rows created by shifting
    df_clean = df.dropna()

    # INPUT FEATURES (X):
    # 1. step: The time of day (Crucial for learning cycles)
    # 2. vehicle_count: Current load
    # 3. avg_speed: Is traffic moving or stopped?
    # 4. lag_1min: Was it lower or higher a minute ago? (Trend detection)
    # 5. lag_5min: Contextual history
    X = df_clean[["step", "vehicle_count", "avg_speed", "lag_1min", "lag_5min"]]
    y = df_clean["target"]

    # --- TRAINING ---
    # We use random shuffle because we only have one simulation run.
    # In a production environment with multiple days of data, we would split chronologically.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    print("🏋️  Training Random Forest (Spatial + Temporal)...")
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # --- EVALUATION ---
    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)

    print("-" * 35)
    print(f"🏆 AI PERFORMANCE REPORT")
    print(f"   Model: Random Forest (v2)")
    print(f"   Prediction Horizon: {HORIZON} seconds")
    print(f"   Accuracy (R² Score): {r2:.4f}")
    print(f"   Mean Squared Error: {mse:.2f}")
    print("-" * 35)

    if r2 < 0.6:
        print("⚠️  WARNING: Model accuracy is low. Check simulation data quality.")

    # Save
    os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    print(f"💾 Trained Brain saved to: {MODEL_FILE}")


if __name__ == "__main__":
    train_brain()
