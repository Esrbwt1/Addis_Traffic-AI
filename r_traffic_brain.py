import traci
import csv
import time
import numpy as np

# R PROTOCOL: CONSTANTS
SUMO_BINARY = "sumo-gui"
 # Changed to "sumo" (fast mode, no GUI) for rapid data harvesting. Change back to "sumo-gui" to watch.
CONFIG_FILE = "osm.sumocfg"

def get_network_stats():
    """Harms raw data from the simulation."""
    # Count total vehicles
    veh_count = traci.vehicle.getIDCount()
    
    # Calculate average speed of all cars in the system
    # (If no cars, speed is 0)
    speeds = [traci.vehicle.getSpeed(veh_id) for veh_id in traci.vehicle.getIDList()]
    avg_speed = sum(speeds) / len(speeds) if speeds else 0.0
    
    return veh_count, avg_speed

def run_simulation():
    # CONNECT TO SIMULATION
    cmd = [SUMO_BINARY, "-c", CONFIG_FILE, "--start"]
    traci.start(cmd)
    
    print("I am R. Data Harvest Protocol Initiated.")
    
    # PREPARE THE FILE
    with open('addis_data.csv', 'w', newline='') as csvfile:
        fieldnames = ['step', 'vehicle_count', 'avg_speed']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        step = 0
        try:
            # Run for 3600 steps (1 virtual hour) or until empty
            while step < 3600 and traci.simulation.getMinExpectedNumber() > 0:
                traci.simulationStep()
                
                # OPTIMIZE LIGHTS (Rule-Based)
                if step % 100 == 0:
                    tls_list = traci.trafficlight.getIDList()
                    for tls in tls_list:
                         # Simple logic: If busy, extend phase
                        pass # (Kept simple for speed, focus is now on Logging)

                # LOG DATA
                count, speed = get_network_stats()
                writer.writerow({'step': step, 'vehicle_count': count, 'avg_speed': speed})
                
                if step % 500 == 0:
                    print(f"Recording Step {step}... Vehicles: {count}")
                
                step += 1
                
        except Exception as e:
            print(f"R ERROR: {e}")
        finally:
            traci.close()
            print("R REPORT: Data harvest complete. 'addis_data.csv' created.")

if __name__ == "__main__":
    run_simulation()