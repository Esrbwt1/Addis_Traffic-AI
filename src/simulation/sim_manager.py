import os
import sys
import traci  # Traffic Control Interface: The bridge between Python and SUMO
import pandas as pd
from datetime import datetime

"""
Addis Ababa Traffic Control System (Manager)
--------------------------------------------
This script acts as the 'Brain' of the simulation.
It connects to the running SUMO instance via TraCI.

Responsibilities:
1. Actuation: Reads sensor data and changes traffic lights.
2. Data Harvesting: Logs vehicle counts and speeds for AI training.
3. Simulation Control: Manages the step-by-step execution loop.
"""

# --- CONFIGURATION ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
SUMO_CFG = os.path.join(CURRENT_DIR, "osm.sumocfg")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")

# Simulation Parameters
GUI_MODE = True  # Set False for headless training (faster)
MAX_STEPS = 3600  # 1 Hour of simulated traffic


class AddisTrafficBrain:
    def __init__(self):
        self.step = 0
        self.data_buffer = []  # In-memory storage for logs

        # Ensure output directory exists
        os.makedirs(DATA_DIR, exist_ok=True)

        # Construct the command to launch SUMO
        sumo_binary = "sumo-gui" if GUI_MODE else "sumo"
        cmd = [sumo_binary, "-c", SUMO_CFG, "--start"]

        try:
            # Start the connection
            traci.start(cmd)
            print(f"🚀 Simulation Initialized. Target: {SUMO_CFG}")
        except Exception as e:
            print(f"❌ Fatal Error: Could not start SUMO.\nReason: {e}")
            sys.exit(1)

    def optimize_traffic_lights(self):
        """
        Rule-Based Optimization Logic.
        ----------------------------
        This function queries the induction loops (sensors) at intersections.
        If a specific lane has a long queue, it dynamically extends the Green phase.

        Future Goal: Replace this 'If/Else' logic with a Neural Network.
        """
        # Get list of all traffic lights in the map
        tls_ids = traci.trafficlight.getIDList()

        for tls_id in tls_ids:
            # Get the lanes controlled by this specific traffic light
            controlled_lanes = traci.trafficlight.getControlledLanes(tls_id)

            max_queue_length = 0

            # Check every lane for stopped cars
            for lane in controlled_lanes:
                # 'getLastStepHaltingNumber' returns count of cars with speed < 0.1 m/s
                queue = traci.lane.getLastStepHaltingNumber(lane)
                if queue > max_queue_length:
                    max_queue_length = queue

            # --- CONTROL LOGIC ---
            # Threshold: If more than 10 cars are waiting, traffic is building up.
            if max_queue_length > 10:
                # Get current timer details
                current_phase_duration = traci.trafficlight.getPhaseDuration(tls_id)

                # Action: Add 10 seconds to the current green light
                # This helps flush out the queue.
                traci.trafficlight.setPhaseDuration(tls_id, current_phase_duration + 10)

                # Log the action periodically so we don't spam the terminal
                if self.step % 100 == 0:
                    print(
                        f"🚦 Intervention at {tls_id}: Queue={max_queue_length}, Extended Green."
                    )

    def collect_data(self):
        """
        Data Harvesting Module.
        -----------------------
        Captures the global state of the network at every second.
        This data will be used as the 'Ground Truth' to train the AI predictor.
        """
        # Count total vehicles currently active in the network
        veh_count = traci.vehicle.getIDCount()

        # Calculate Network Average Speed
        # We iterate through all vehicles to get their current speed.
        # If the network is empty, average speed is 0.
        all_speeds = [traci.vehicle.getSpeed(veh) for veh in traci.vehicle.getIDList()]
        avg_speed = sum(all_speeds) / len(all_speeds) if all_speeds else 0.0

        self.data_buffer.append(
            {
                "step": self.step,
                "vehicle_count": veh_count,
                "avg_speed": round(avg_speed, 2),  # Round to 2 decimals
                "timestamp": datetime.now().strftime("%H:%M:%S"),
            }
        )

    def run(self):
        """
        Main Execution Loop.
        Runs until MAX_STEPS is reached or simulation is empty.
        """
        print("🤖 System Online. Monitoring Traffic...")

        try:
            # Main Loop
            while traci.simulation.getMinExpectedNumber() > 0 and self.step < MAX_STEPS:
                # 1. Advance Physics (Move cars)
                traci.simulationStep()

                # 2. Apply Control Logic
                self.optimize_traffic_lights()

                # 3. Record Data
                self.collect_data()

                self.step += 1

        except KeyboardInterrupt:
            print("\n🛑 Simulation aborted by user.")
        finally:
            # Cleanup
            self.save_data()
            traci.close()
            print("✅ Simulation Session Ended.")

    def save_data(self):
        """Dumps the in-memory buffer to a CSV file."""
        filename = os.path.join(DATA_DIR, "traffic_log.csv")
        df = pd.DataFrame(self.data_buffer)
        df.to_csv(filename, index=False)
        print(f"💾 Training Data saved to: {filename}")


if __name__ == "__main__":
    # Entry point
    brain = AddisTrafficBrain()
    brain.run()
