import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import sys

"""
Addis Ababa Traffic AI (Predictor Module)
-----------------------------------------
This script trains a Machine Learning model to predict future traffic density.

Methodology:
- Algorithm: Random Forest Regressor (Good for non-linear traffic patterns).
- Input (Features): Traffic density and speed from the past (Lag features).
- Output (Target): Traffic density 5 minutes into the future.
"""

# --- PATHS ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
DATA_FILE = os.path.join(PROJECT_ROOT, "data", "raw", "traffic_log.csv")
MODEL_FILE = os.path.join(PROJECT_ROOT, "data", "models", "traffic_model.pkl")


def train_brain():
    print("🧠 Initializing AI Training Protocol...")

    # 1. Load Data
    if not os.path.exists(DATA_FILE):
        print("❌ Error: No training data found.")
        sys.exit(1)

    df = pd.read_csv(DATA_FILE)
    print(f"📊 Loaded {len(df)} data points.")

    # 2. Feature Engineering (The "Memory")
    # We want the AI to look at the past to predict the future.
    # Prediction Horizon: 5 minutes (300 seconds -> 300 steps)
    HORIZON = 300

    # Create the 'Target': What will vehicle_count be in 300 steps?
    df["target_future_density"] = df["vehicle_count"].shift(-HORIZON)

    # Create 'Features': What was traffic like recently?
    # Lag 1: Traffic 1 minute ago
    df["lag_1min"] = df["vehicle_count"].shift(60)
    # Lag 5: Traffic 5 minutes ago
    df["lag_5min"] = df["vehicle_count"].shift(300)

    # Drop rows with NaN (empty data created by shifting)
    df_clean = df.dropna()

    # Define Input (X) and Output (y)
    # We use current count, speed, and past counts to predict future.
    X = df_clean[["vehicle_count", "avg_speed", "lag_1min", "lag_5min"]]
    y = df_clean["target_future_density"]

    print(f"📉 Training data shape after cleaning: {X.shape}")

    # 3. Split Data (Train vs Test)
    # 80% for training, 20% for testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    # 4. Train the Model (Random Forest)
    print("🏋️  Training Random Forest Model (Estimators=100)...")
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # 5. Evaluate
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("-" * 30)
    print(f"🏆 AI PERFORMANCE REPORT")
    print(f"   Target: Predict traffic {HORIZON} seconds ahead")
    print(f"   Accuracy (R² Score): {r2:.4f} (1.0 is perfect)")
    print(f"   Error Margin (MSE): {mse:.2f}")
    print("-" * 30)

    # 6. Save the Brain
    os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    print(f"💾 Model saved to: {MODEL_FILE}")


if __name__ == "__main__":
    train_brain()
