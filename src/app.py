import gradio as gr
import joblib
import pandas as pd
import os
import numpy as np

# --- CONFIG ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
MODEL_PATH = os.path.join(PROJECT_ROOT, "data", "models", "traffic_model.pkl")

# --- LOAD BRAIN ---
try:
    model = joblib.load(MODEL_PATH)
    print("✅ AI Model Loaded Successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("Run 'src/ai/traffic_predictor.py' first!")
    exit()


def predict_congestion(current_vehicles, current_speed, lag_1min, lag_5min):
    """
    The AI Logic connected to the UI.
    """
    # Prepare input array (must match training shape)
    input_data = pd.DataFrame(
        [[current_vehicles, current_speed, lag_1min, lag_5min]],
        columns=["vehicle_count", "avg_speed", "lag_1min", "lag_5min"],
    )

    # Predict
    prediction = model.predict(input_data)[0]

    # Interpret result
    status = "🟢 Free Flow"
    if prediction > 100:
        status = "🟡 Moderate Traffic"
    if prediction > 200:
        status = "🔴 Severe Congestion"

    return f"{int(prediction)} Vehicles", status


# --- UI LAYOUT ---
with gr.Blocks(title="Addis Ababa Traffic Digital Twin") as demo:
    gr.Markdown("# 🇪🇹 Addis Ababa Traffic AI Core")
    gr.Markdown(
        "Input live sensor data to predict traffic conditions **5 minutes into the future**."
    )

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📡 Sensor Inputs")
            inp_veh = gr.Slider(0, 500, label="Current Vehicles (Count)", value=120)
            inp_speed = gr.Slider(0, 30, label="Avg Speed (m/s)", value=10)
            inp_lag1 = gr.Slider(0, 500, label="Traffic 1 Min Ago", value=115)
            inp_lag5 = gr.Slider(0, 500, label="Traffic 5 Mins Ago", value=100)
            btn = gr.Button("🔮 Predict Future State", variant="primary")

        with gr.Column():
            gr.Markdown("### 🧠 AI Prediction (t+5min)")
            out_count = gr.Textbox(label="Predicted Density")
            out_status = gr.Textbox(label="Congestion Level")

    # Connect buttons
    btn.click(
        fn=predict_congestion,
        inputs=[inp_veh, inp_speed, inp_lag1, inp_lag5],
        outputs=[out_count, out_status],
    )

if __name__ == "__main__":
    demo.launch()
