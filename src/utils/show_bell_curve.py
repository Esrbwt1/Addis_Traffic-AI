import numpy as np
import matplotlib.pyplot as plt


def plot_theoretical_curve():
    # 1. Setup Time (0 to 3600 seconds)
    steps = np.arange(3600)

    # 2. Define the Gaussian (Bell Curve) Parameters
    # These match what we put in generate_synthetic_data.py
    peak_time = 1800  # Peak traffic at 30 minutes
    width = 600  # How wide the traffic jam is
    max_cars = 180  # Maximum number of cars

    # 3. The Math Formula
    # y = Height * exp( - (x - peak)^2 / (2 * width^2) )
    counts = max_cars * np.exp(-((steps - peak_time) ** 2) / (2 * width**2))

    # 4. Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(
        steps, counts, color="orange", linewidth=3, label="Theoretical Traffic Flow"
    )

    # Add labels to explain it to the teacher
    plt.axvline(x=1800, color="red", linestyle="--", alpha=0.5, label="Peak Rush Hour")
    plt.text(1850, 170, "Peak Congestion", color="red")

    plt.text(200, 50, "Traffic Buildup \n(Morning)", color="green")
    plt.text(2800, 50, "Traffic Clearing \n(Afternoon)", color="green")

    plt.title("The Gaussian Bell Curve Model", fontsize=16)
    plt.xlabel("Simulation Step (Time)", fontsize=12)
    plt.ylabel("Vehicle Count", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Save
    plt.savefig("data/processed/bell_curve_model.png")
    print("✅ Graph saved to data/processed/bell_curve_model.png")
    plt.show()


if __name__ == "__main__":
    plot_theoretical_curve()
