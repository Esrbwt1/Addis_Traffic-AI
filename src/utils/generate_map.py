import os
import subprocess
import requests
import sys

"""
Addis Ababa Traffic Map Generator
---------------------------------
This script automates the creation of a digital twin for the simulation.
It performs three main steps:
1. Downloads raw geographic data (OSM) for the Bole/Meskel corridor.
2. Converts raw data into a graph network (Nodes/Edges) for vehicles.
3. Generates 3D polygons for building visualization.
"""

# --- CONFIGURATION ---
# Coordinates for Addis Ababa (Bole Road Corridor)
# Format: [min_longitude, min_latitude, max_longitude, max_latitude]
# Verified via OpenStreetMap Export Tool.
BBOX = "38.760,8.995,38.780,9.015"

# File Paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
SIM_DIR = os.path.join(PROJECT_ROOT, "src", "simulation")

# Output Files
OSM_FILE = os.path.join(SIM_DIR, "addis_raw.osm")  # Raw download
NET_FILE = os.path.join(SIM_DIR, "osm.net.xml")  # Drivable road network
POLY_FILE = os.path.join(SIM_DIR, "osm.poly.xml")  # 3D Buildings


def download_osm_data():
    """
    Connects to the OpenStreetMap API to fetch the latest map data.
    """
    print(f"🌍 Connecting to OpenStreetMap API...")
    print(f"📍 Target Zone: Bole/Meskel Corridor ({BBOX})")

    url = f"https://api.openstreetmap.org/api/0.6/map?bbox={BBOX}"

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()  # Check for HTTP errors (e.g., 404, 500)

        # Write binary data to disk
        with open(OSM_FILE, "wb") as f:
            f.write(response.content)
        print("✅ Download successful. Raw data saved.")

    except requests.exceptions.RequestException as e:
        print(f"❌ Network Error: Failed to download map data.\nDetails: {e}")
        sys.exit(1)


def build_network():
    """
    Uses SUMO binaries (netconvert, polyconvert) to compile the raw data.
    """
    print("🚧 Initializing SUMO Network Compiler...")

    # Locate SUMO binaries from the environment variable
    try:
        import sumo

        sumo_home = sumo.SUMO_HOME
        netconvert_bin = os.path.join(sumo_home, "bin", "netconvert")
        polyconvert_bin = os.path.join(sumo_home, "bin", "polyconvert")
        typemap_file = os.path.join(
            sumo_home, "data", "typemap", "osmPolyconvert.typ.xml"
        )
    except ImportError:
        print("❌ Error: SUMO python module not found. Is your venv active?")
        sys.exit(1)

    # --- Step 1: Netconvert (Roads) ---
    # We remove 'geometry' to simplify curved roads into straight segments for performance.
    # We enable 'tls.guess' to automatically place traffic lights at major intersections.
    cmd_net = [
        netconvert_bin,
        "--osm-files",
        OSM_FILE,
        "--output-file",
        NET_FILE,
        "--geometry.remove",
        "true",
        "--roundabouts.guess",
        "true",
        "--ramps.guess",
        "true",
        "--junctions.join",
        "true",
        "--tls.guess",
        "true",
        "--tls.join",
        "true",
        "--verbose",
        "false",
    ]

    try:
        subprocess.run(
            cmd_net, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
        )
        print("✅ Road Network (.net.xml) generated successfully.")
    except subprocess.CalledProcessError:
        print("❌ Critical Error: netconvert failed to build the network.")
        sys.exit(1)

    # --- Step 2: Polyconvert (Buildings) ---
    # This extracts building footprints to make the simulation look realistic.
    cmd_poly = [
        polyconvert_bin,
        "--osm-files",
        OSM_FILE,
        "--net-file",
        NET_FILE,
        "--output-file",
        POLY_FILE,
        "--type-file",
        typemap_file,
    ]

    try:
        subprocess.run(
            cmd_poly, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
        )
        print("✅ 3D Scenery (.poly.xml) generated successfully.")
    except subprocess.CalledProcessError:
        print("⚠️ Warning: Failed to generate buildings (Simulation will still work).")


def generate_traffic():
    """
    Calls SUMO's randomTrips.py to generate traffic demand automatically.
    """
    print("🚗 Generating Random Traffic Demand...")

    try:
        import sumo

        sumo_home = sumo.SUMO_HOME
        random_trips = os.path.join(sumo_home, "tools", "randomTrips.py")
    except:
        random_trips = os.path.join(
            os.environ.get("SUMO_HOME"), "tools", "randomTrips.py"
        )

    # 1. Passenger Cars
    cmd_cars = [
        "python",
        random_trips,
        "-n",
        NET_FILE,
        "-o",
        os.path.join(SIM_DIR, "osm.passenger.trips.xml"),
        "-e",
        "3600",
        "-p",
        "1.0",
        "--validate",
    ]

    # 2. Buses
    cmd_buses = [
        "python",
        random_trips,
        "-n",
        NET_FILE,
        "-o",
        os.path.join(SIM_DIR, "osm.bus.trips.xml"),
        "-e",
        "3600",
        "-p",
        "60.0",
        "--vehicle-class",
        "bus",
        "--prefix",
        "bus",
        "--validate",
    ]

    subprocess.run(cmd_cars, check=True)
    subprocess.run(cmd_buses, check=True)
    print("✅ Traffic demand generated.")


if __name__ == "__main__":
    # Ensure the simulation directory exists
    os.makedirs(SIM_DIR, exist_ok=True)

    download_osm_data()
    build_network()
    generate_traffic()
    print("\n🎉 Digital Twin Construction Complete.")
