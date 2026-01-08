import gradio as gr
import pandas as pd
import joblib
import os

"""
Addis Ababa Traffic Dashboard
-----------------------------
Web UI for interacting with the predictive Digital Twin.
"""

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(
    os.path.dirname(PROJECT_ROOT), "data", "models", "traffic_model.pkl"
)


def predict_congestion(step_time, current_count, current_speed, lag1, lag5):
    """AI Inference Engine."""
    if not os.path.exists(MODEL_PATH):
        return "⚠️ Error: Model not found. Train AI first!", "Error"

    try:
        model = joblib.load(MODEL_PATH)

        # Input DataFrame must match training features exactly
        input_data = pd.DataFrame(
            [[step_time, current_count, current_speed, lag1, lag5]],
            columns=["step", "vehicle_count", "avg_speed", "lag_1min", "lag_5min"],
        )

        pred_float = model.predict(input_data)[0]
        prediction = int(pred_float)

        status = "🟢 Free Flow"
        if prediction > 100:
            status = "🟡 Moderate Traffic"
        if prediction > 180:
            status = "🔴 Severe Congestion"

        return f"{prediction} Vehicles", status

    except Exception as e:
        return f"Error: {str(e)}", "System Failure"


# UI Layout
with gr.Blocks(title="Addis Traffic Digital Twin") as demo:
    gr.Markdown("# 🇪🇹 Addis Ababa Traffic AI Core")
    gr.Markdown(
        "Input live sensor data to predict traffic conditions **5 minutes into the future**."
    )

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📡 Sensor Inputs")
            step = gr.Slider(
                0,
                3600,
                value=1800,
                label="Time of Day (Step)",
                info="0=Start, 3600=End",
            )
            count = gr.Slider(0, 500, value=120, label="Current Vehicles")
            speed = gr.Slider(0, 30, value=10, label="Avg Speed (m/s)")

            gr.Markdown("### 📉 Historical Context")
            lag1 = gr.Slider(0, 500, value=115, label="Traffic 1 Min Ago")
            lag5 = gr.Slider(0, 500, value=100, label="Traffic 5 Mins Ago")

            btn = gr.Button("🔮 Predict Future State", variant="primary")

        with gr.Column():
            gr.Markdown("### 🧠 AI Prediction (t+5min)")
            out_val = gr.Textbox(label="Predicted Density", text_align="center")
            out_stat = gr.Textbox(label="Congestion Level", text_align="center")

    btn.click(
        fn=predict_congestion,
        inputs=[step, count, speed, lag1, lag5],
        outputs=[out_val, out_stat],
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
