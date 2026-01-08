import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import os

# Paths
DATA_FILE = "data/raw/traffic_log.csv"


def verify_integrity():
    print("🕵️  AUDITING AI LOGIC...")
    df = pd.read_csv(DATA_FILE)

    # Feature Engineering
    HORIZON = 300
    df["target"] = df["vehicle_count"].shift(-HORIZON)
    df["lag_1min"] = df["vehicle_count"].shift(60)
    df["lag_5min"] = df["vehicle_count"].shift(300)
    df_clean = df.dropna()

    X = df_clean[["step", "vehicle_count", "avg_speed", "lag_1min", "lag_5min"]]
    y = df_clean["target"]

    # --- TEST 1: The "Hard" Split (No Shuffling) ---
    # We hide the last 20% of the simulation from the AI completely.
    # It has to guess the end of the day having only seen the morning/afternoon.
    cutoff = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:cutoff], X.iloc[cutoff:]
    y_train, y_test = y.iloc[:cutoff], y.iloc[cutoff:]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    score = r2_score(y_test, pred)

    print(f"\n🧪 TEST 1: Chronological Split Accuracy")
    print(f"   (If this is > 0.5, the AI is genuinely smart. If < 0, it's memorizing.)")
    print(f"   REAL ACCURACY: {score:.4f}")

    # --- TEST 2: Feature Importance ---
    # What is the AI actually looking at?
    print(f"\n🧠 TEST 2: How the AI thinks (Feature Importance)")
    importances = model.feature_importances_
    features = X.columns

    sorted_idx = np.argsort(importances)
    plt.figure(figsize=(10, 6))
    plt.title("What drives the prediction?")
    plt.barh(range(len(sorted_idx)), importances[sorted_idx], align="center")
    plt.yticks(range(len(sorted_idx)), [features[i] for i in sorted_idx])
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig("data/processed/ai_audit.png")
    print("   📊 Chart saved to data/processed/ai_audit.png")

    # --- TEST 3: Visual Proof ---
    # Plot Real vs Predicted for the test period
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.values, label="Reality", color="blue", alpha=0.6)
    plt.plot(pred, label="AI Prediction", color="red", linestyle="--")
    plt.title(f"Reality vs Prediction (Last 20% of Simulation) | R2: {score:.2f}")
    plt.legend()
    plt.savefig("data/processed/prediction_audit.png")
    print("   📉 Graph saved to data/processed/prediction_audit.png")


if __name__ == "__main__":
    verify_integrity()
