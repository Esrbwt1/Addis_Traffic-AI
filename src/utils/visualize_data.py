import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

"""
Traffic Data Analysis Module
----------------------------
This script is responsible for the "Reporting" phase of the project.
It reads the raw telemetry data harvested by the Simulation Manager
and converts it into human-readable performance metrics.

Libraries Used:
- pandas: For efficient CSV reading and data slicing.
- matplotlib: For rendering scientific-grade charts.
"""

# --- PATH CONFIGURATION ---
# We use dynamic paths so this script works on any computer (Linux/Windows/Mac)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))

# Input: The raw logs from the simulation
DATA_FILE = os.path.join(PROJECT_ROOT, "data", "raw", "traffic_log.csv")

# Output: Where we save the resulting graph
OUTPUT_IMG = os.path.join(PROJECT_ROOT, "data", "processed", "traffic_analysis.png")


def plot_traffic_flow():
    """
    Generates a dual-axis chart showing Traffic Volume vs. Flow Speed.
    """

    # 1. Validation: Check if data exists
    if not os.path.exists(DATA_FILE):
        print(f"❌ Error: Data file not found at {DATA_FILE}")
        print(
            "   -> Solution: Run 'src/simulation/sim_manager.py' to generate data first."
        )
        sys.exit(1)

    print(f"📊 Loading telemetry data from: {DATA_FILE}")

    # 2. Load Data using Pandas
    # We parse the CSV into a DataFrame (df), which is like a programmable Excel sheet.
    try:
        df = pd.read_csv(DATA_FILE)
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        sys.exit(1)

    # 3. Setup the Canvas
    # figsize=(14, 10) creates a large image (1400x1000 pixels approx)
    plt.figure(figsize=(14, 10))

    # --- CHART 1: CONGESTION (Vehicle Count) ---
    # subplot(2, 1, 1) means: 2 rows, 1 column, this is plot #1
    plt.subplot(2, 1, 1)

    # Plot X (Time) vs Y (Count)
    plt.plot(
        df["step"],
        df["vehicle_count"],
        label="Active Vehicles",
        color="#ff7f0e",  # Standard "Safety Orange" color
        linewidth=2,
    )

    plt.title("Addis Ababa (Bole Corridor) - Traffic Congestion Analysis", fontsize=16)
    plt.ylabel("Total Vehicles on Road", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)  # Add a faint grid for readability
    plt.legend(loc="upper left")

    # --- CHART 2: FLOW QUALITY (Average Speed) ---
    # subplot(2, 1, 2) means: 2 rows, 1 column, this is plot #2
    plt.subplot(2, 1, 2)

    # Plot X (Time) vs Y (Speed)
    plt.plot(
        df["step"],
        df["avg_speed"],
        label="Avg Network Speed (m/s)",
        color="#1f77b4",  # Standard "Matplotlib Blue"
        linewidth=2,
    )

    plt.xlabel("Simulation Duration (Seconds)", fontsize=14)
    plt.ylabel("Average Speed (m/s)", fontsize=12)

    # Add a "Critical Speed" line (Traffic Jam Threshold)
    # If speed drops below 5 m/s (~18 km/h), it's a jam.
    plt.axhline(y=5, color="red", linestyle=":", label="Congestion Threshold (5 m/s)")

    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper right")

    # 4. Save and Show
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(OUTPUT_IMG), exist_ok=True)

    plt.tight_layout()  # Adjusts margins so titles don't get cut off
    plt.savefig(OUTPUT_IMG)
    print(f"✅ Analysis Graph saved to: {OUTPUT_IMG}")

    # Show the interactive window
    print("🖥️  Opening visualization window...")
    plt.show()


if __name__ == "__main__":
    plot_traffic_flow()
