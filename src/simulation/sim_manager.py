import os
import sys
import traci
import pandas as pd
from datetime import datetime

"""
Addis Ababa Traffic Control System (Manager)
--------------------------------------------
Acts as the central controller for the Digital Twin.
- Connects to SUMO via TraCI.
- Implements Adaptive Signal Control logic.
- Harvests telemetry data for AI training.
"""

# --- CONFIGURATION ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
SUMO_CFG = os.path.join(CURRENT_DIR, "osm.sumocfg")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")

GUI_MODE = True
MAX_STEPS = 3600


class AddisTrafficBrain:
    def __init__(self):
        self.step = 0
        self.data_buffer = []
        os.makedirs(DATA_DIR, exist_ok=True)

        sumo_binary = "sumo-gui" if GUI_MODE else "sumo"
        cmd = [sumo_binary, "-c", SUMO_CFG, "--start"]

        try:
            traci.start(cmd)
            print(f"🚀 Simulation Initialized. Target: {SUMO_CFG}")
        except Exception as e:
            print(f"❌ Fatal Error: Could not start SUMO.\nReason: {e}")
            sys.exit(1)

    def optimize_traffic_lights(self):
        """
        Adaptive Control Logic:
        scans intersections for long queues. If a queue > 10 cars is found,
        AND the light is Green, it extends the phase to flush traffic.
        """
        tls_ids = traci.trafficlight.getIDList()

        for tls_id in tls_ids:
            controlled_lanes = traci.trafficlight.getControlledLanes(tls_id)
            max_queue = 0

            # Find longest queue at this intersection
            for lane in controlled_lanes:
                queue = traci.lane.getLastStepHaltingNumber(lane)
                if queue > max_queue:
                    max_queue = queue

            # --- LOGIC FIX: Green Light Check ---
            if max_queue > 10:
                # 1. Get State (e.g. "GrGr")
                current_state = traci.trafficlight.getRedYellowGreenState(tls_id)

                # 2. Check: Is it actually Green?
                if "G" in current_state or "g" in current_state:
                    # 3. Safe Access: Get duration only if we are going to use it
                    current_duration = traci.trafficlight.getPhaseDuration(tls_id)
                    traci.trafficlight.setPhaseDuration(tls_id, current_duration + 10)

                    if self.step % 100 == 0:
                        print(
                            f"🚦 Intervention at {tls_id}: Queue={max_queue}, Extended Green."
                        )

    def collect_data(self):
        """Harvests global network state."""
        veh_count = traci.vehicle.getIDCount()
        all_speeds = [traci.vehicle.getSpeed(veh) for veh in traci.vehicle.getIDList()]
        avg_speed = sum(all_speeds) / len(all_speeds) if all_speeds else 0.0

        self.data_buffer.append(
            {
                "step": self.step,
                "vehicle_count": veh_count,
                "avg_speed": round(avg_speed, 2),
                "timestamp": datetime.now().strftime("%H:%M:%S"),
            }
        )

    def run(self):
        print("🤖 System Online. Monitoring Traffic...")
        try:
            while traci.simulation.getMinExpectedNumber() > 0 and self.step < MAX_STEPS:
                traci.simulationStep()
                self.optimize_traffic_lights()
                self.collect_data()
                self.step += 1
        except KeyboardInterrupt:
            print("\n🛑 Simulation aborted by user.")
        finally:
            self.save_data()
            traci.close()
            print("✅ Simulation Session Ended.")

    def save_data(self):
        filename = os.path.join(DATA_DIR, "traffic_log.csv")
        df = pd.DataFrame(self.data_buffer)
        df.to_csv(filename, index=False)
        print(f"💾 Training Data saved to: {filename}")


if __name__ == "__main__":
    brain = AddisTrafficBrain()
    brain.run()
