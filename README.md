# Addis Ababa Digital Twin: Traffic Prediction & Optimization AI

## 🌍 Project Overview
This project implements a **Digital Twin** of the Bole Road / Meskel Square corridor in Addis Ababa using the SUMO (Simulation of Urban MObility) engine. It uses real-world OpenStreetMap data to simulate traffic flow and employs a **Machine Learning (Random Forest)** pipeline to predict congestion 5 minutes in advance.

## 🏗️ System Architecture
The project is modularized into three core components:

1.  **The Environment (`src/simulation`)**:
    *   **Map Generation**: Automatically pulls real-time coordinates `(8.995, 38.760)` from OSM to build the Bole road network.
    *   **Traffic Logic**: Implements an **Adaptive Signal Control** system that dynamically extends green lights based on sensor queue length.
2.  **The Data Pipeline (`src/utils`)**:
    *   Harvests telemetry data (vehicle count, speed, timestamps) every second.
    *   Visualizes network performance using Matplotlib (Congestion vs. Flow analysis).
3.  **The AI Core (`src/ai`)**:
    *   **Model**: Random Forest Regressor (`n_estimators=100`).
    *   **Feature Engineering**:
        *   **Temporal**: Uses Simulation Step (Time of Day) to distinguish between traffic buildup and clearing.
        *   **Spatial**: Uses Lag Features (t-1min, t-5min) to track flow trends.
    *   **Performance**: Achieved **>0.90 R² Score** on a 5-minute prediction horizon (Interpolation).

## 📂 Repository Structure
```text
Addis_Traffic-AI/
├── config/             # Global simulation settings
├── data/
│   ├── raw/            # Telemetry logs (traffic_log.csv)
│   ├── processed/      # Generated graphs and reports
│   └── models/         # Trained AI models (.pkl)
├── src/
│   ├── ai/             # Prediction Logic
│   ├── simulation/     # SUMO Manager & Map Configs
│   └── utils/          # Map Generators & Data Visualizers
└── README.md
```

## 🚀 Installation & Usage

### 1. Prerequisites
*   Python 3.10+
*   SUMO Simulation Engine (`sudo pacman -S sumo` or via pip)

### 2. Setup
```bash
# Create environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install traci pandas numpy scikit-learn matplotlib pyyaml requests
```

### 3. Running the Digital Twin
**Step A: Generate the Map (Optional)**
Repulls the latest road data from OpenStreetMap.
```bash
python src/utils/generate_map.py
```

**Step B: Run the Simulation**
Launches the GUI and starts the Adaptive Control System.
```bash
python src/simulation/sim_manager.py
```

**Step C: Train the AI**
Trains the predictor on the newly harvested data.
```bash
python src/ai/traffic_predictor.py
```

## 📊 Results & Validation
*   **Congestion Management**: The Adaptive Control system successfully maintained network flow above critical thresholds (5 m/s) during peak load.
*   **Bell-Curve Simulation**: Traffic generation was tuned to simulate a full Rush Hour cycle (Rise $\to$ Peak $\to$ Clear) to ensure the AI learns how traffic dissipates.
*   **Model Note**: The high accuracy (0.99 R²) is achieved via randomized cross-validation on a single-day simulation. For production deployment, multi-day data collection would be required to prevent overfitting to a specific daily schedule.
