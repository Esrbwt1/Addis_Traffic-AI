import os
import subprocess
import requests
import sys

"""
Addis Ababa Traffic Map Generator (V2.1 - Automated)
----------------------------------------------------
This script builds the Digital Twin environment.
It handles:
1. Downloading raw OSM data for Bole/Meskel Square.
2. Compiling the SUMO Network topology (.net.xml).
3. Generating 3D Building geometry (.poly.xml).
4. Generating dynamic Traffic Demand (Cars/Buses) automatically.
"""

# --- CONFIGURATION ---
# Coordinates: Bole Road Corridor
BBOX = "38.760,8.995,38.780,9.015"

# Paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
SIM_DIR = os.path.join(PROJECT_ROOT, "src", "simulation")

# Output Files
OSM_FILE = os.path.join(SIM_DIR, "addis_raw.osm")
NET_FILE = os.path.join(SIM_DIR, "osm.net.xml")
POLY_FILE = os.path.join(SIM_DIR, "osm.poly.xml")


def download_osm_data():
    """Fetches raw geographic data from OpenStreetMap API."""
    print(f"🌍 Connecting to OpenStreetMap API...")
    url = f"https://api.openstreetmap.org/api/0.6/map?bbox={BBOX}"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        with open(OSM_FILE, "wb") as f:
            f.write(response.content)
        print("✅ Map Data Downloaded.")
    except Exception as e:
        print(f"❌ Network Error: {e}")
        sys.exit(1)


def build_network():
    """Compiles the raw map into a drivable SUMO network."""
    print("🚧 Compiling SUMO Network...")
    try:
        import sumo

        sumo_home = sumo.SUMO_HOME
        netconvert = os.path.join(sumo_home, "bin", "netconvert")
        polyconvert = os.path.join(sumo_home, "bin", "polyconvert")
        typemap = os.path.join(sumo_home, "data", "typemap", "osmPolyconvert.typ.xml")
    except ImportError:
        sys.exit("❌ SUMO python module not found.")

    # Netconvert: Build Roads
    cmd_net = [
        netconvert,
        "--osm-files",
        OSM_FILE,
        "--output-file",
        NET_FILE,
        "--geometry.remove",
        "true",
        "--tls.guess",
        "true",
        "--verbose",
        "false",
    ]

    # Polyconvert: Build Buildings
    cmd_poly = [
        polyconvert,
        "--osm-files",
        OSM_FILE,
        "--net-file",
        NET_FILE,
        "--output-file",
        POLY_FILE,
        "--type-file",
        typemap,
    ]

    subprocess.run(cmd_net, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(cmd_poly, check=False, stdout=subprocess.DEVNULL)
    print("✅ Network & Scenery Built.")


def generate_traffic():
    """
    Generates random traffic trips.
    - Cars: Stop at 1800s to allow traffic to clear (Bell Curve data).
    - Buses: Run continuously.
    """
    print("🚗 Generating Traffic Demand...")
    try:
        import sumo

        random_trips = os.path.join(sumo.SUMO_HOME, "tools", "randomTrips.py")
    except:
        random_trips = os.path.join(
            os.environ.get("SUMO_HOME"), "tools", "randomTrips.py"
        )

    # FIX: Removed "true" after --validate to prevent argparse error
    cmd_cars = [
        "python",
        random_trips,
        "-n",
        NET_FILE,
        "-o",
        os.path.join(SIM_DIR, "osm.passenger.trips.xml"),
        "-e",
        "1800",  # Stop halfway to let traffic clear
        "-p",
        "1.5",  # High density
        "--validate",
    ]

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
    print("✅ Traffic Demand Generated.")


if __name__ == "__main__":
    os.makedirs(SIM_DIR, exist_ok=True)
    download_osm_data()
    build_network()
    generate_traffic()
    print("\n🎉 Digital Twin Environment Ready.")
