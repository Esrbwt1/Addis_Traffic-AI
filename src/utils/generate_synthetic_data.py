import pandas as pd
import numpy as np
import os

"""
Synthetic Data Generator (Multi-Day Augmentation)
-------------------------------------------------
This script mathematically generates 30 days of traffic data based on the 
Bell-Curve pattern we validated in the simulation.

Why? 
To train the AI properly (without shuffling), we need distinct days 
(e.g., Train on Days 1-25, Test on Days 26-30). Running the actual simulation 
30 times would take hours. This script mimics the simulation physics 
to generate the data instantly.
"""

# Config
DAYS_TO_SIMULATE = 30
STEPS_PER_DAY = 3600
OUTPUT_FILE = os.path.join("data", "raw", "synthetic_traffic_30days.csv")


def generate_day(day_id):
    # Time steps (0 to 3599)
    steps = np.arange(STEPS_PER_DAY)

    # --- 1. The Physics Pattern (Bell Curve) ---
    # We validated in 'fix/data-bias-logic' that traffic peaks ~1800s.
    # We add random variation so every day looks slightly different.

    # Peak Time: 1800s +/- 5 minutes
    peak_time = 1800 + np.random.randint(-300, 300)

    # Peak Duration (Width): Varies slightly
    width = 600 + np.random.randint(-50, 50)

    # Peak Height (Max Cars): 150 to 200 cars (Matches our Sim)
    max_cars = 180 + np.random.randint(-30, 30)

    # The Math: Gaussian Function
    counts = max_cars * np.exp(-((steps - peak_time) ** 2) / (2 * width**2))

    # --- 2. Add Realism (Noise) ---
    # Real sensors are noisy. We add +/- 5 cars jitter.
    noise = np.random.normal(0, 5, STEPS_PER_DAY)
    counts = counts + noise

    # Physics Constraint: Cars cannot be negative
    counts = np.maximum(counts, 0).astype(int)

    # --- 3. Calculate Speed ---
    # Standard Traffic Flow Theory: More Cars = Lower Speed.
    # Free flow = 15 m/s. Jam = 2 m/s.
    speeds = 15 - (counts / max_cars * 13)

    # Add random speed variations (some drivers are slow/fast)
    speeds = speeds + np.random.normal(0, 1, STEPS_PER_DAY)
    speeds = np.clip(speeds, 1, 20)  # Speed limits

    # Build the Table
    df = pd.DataFrame(
        {
            "day": day_id,
            "step": steps,
            "vehicle_count": counts,
            "avg_speed": np.round(speeds, 2),
        }
    )

    return df


if __name__ == "__main__":
    print(f"🎲 Generating {DAYS_TO_SIMULATE} days of synthetic traffic data...")

    # Generate all days
    all_days = []
    for i in range(1, DAYS_TO_SIMULATE + 1):
        all_days.append(generate_day(i))

    # Combine into one huge dataset
    final_df = pd.concat(all_days)

    # Ensure directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # Save
    final_df.to_csv(OUTPUT_FILE, index=False)

    print(f"✅ Generated {len(final_df)} data points.")
    print(f"💾 Saved to: {OUTPUT_FILE}")
