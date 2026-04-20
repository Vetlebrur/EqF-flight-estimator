"""Compare NIMBUS24 estimator trajectory vs GPS in 3D (NED frame)."""

import sys

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# =============================================================================
# Constants
# =============================================================================

R_EARTH = 6_378_137.0

# Column indices in the FC CSV (0-based)
COL_TIME = 0
COL_LON = 1
COL_LAT = 2
COL_ALT_MM = 3
COL_PN = 36
COL_PE = 37
COL_PD = 38
COL_ROLL = 29
COL_PITCH = 30
COL_YAW = 31


# =============================================================================
# Data Loading
# =============================================================================


def load_flight_csv(path):
    """Load flight data from CSV."""
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    return {
        "t": data[:, COL_TIME],
        "lat": data[:, COL_LAT],
        "lon": data[:, COL_LON],
        "alt": data[:, COL_ALT_MM] / 1000.0,
        "pn": data[:, COL_PN],
        "pe": data[:, COL_PE],
        "pd": data[:, COL_PD],
        "roll": data[:, COL_ROLL],
        "pitch": data[:, COL_PITCH],
        "yaw": data[:, COL_YAW],
    }


def gps_to_ned(lat_deg, lon_deg, alt_m, lat0, lon0, alt0):
    """GPS (lat/lon/alt) → NED (north, east, down) relative to reference."""
    dlat = np.radians(lat_deg - lat0)
    dlon = np.radians(lon_deg - lon0)
    lat0_rad = np.radians(lat0)

    north = dlat * R_EARTH
    east = dlon * np.cos(lat0_rad) * R_EARTH
    down = -(alt_m - alt0)
    return north, east, down


# =============================================================================
# Plotting
# =============================================================================


def plot_3d(data):
    """Plot 3D trajectory comparison."""
    lat = data["lat"]
    lon = data["lon"]
    alt = data["alt"]

    assert np.all(np.abs(lat) <= 90), "lat out of range – check column order"
    assert np.all(np.abs(lon) <= 180), "lon out of range – check column order"

    valid = (lat != 0) & (lon != 0)
    first = np.argmax(valid)
    lat0, lon0, alt0 = lat[first], lon[first], alt[first]

    gps_n, gps_e, gps_d = gps_to_ned(lat, lon, alt, lat0, lon0, alt0)

    gps_n = np.where(valid, gps_n, np.nan)
    gps_e = np.where(valid, gps_e, np.nan)
    gps_d = np.where(valid, gps_d, np.nan)

    n0, e0, d0 = data["pn"][0], data["pe"][0], data["pd"][0]
    est_n = data["pn"] - n0
    est_e = data["pe"] - e0
    est_d = data["pd"] - d0

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(est_e, est_n, -est_d, linewidth=1.2, color="tab:blue", label="Estimator (NED)")
    ax.plot(gps_e, gps_n, -gps_d, linewidth=1.2, color="tab:orange", linestyle="--", label="GPS → NED")

    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.set_zlabel("Up [m]")
    ax.set_title("NIMBUS24 3D Trajectory: Estimator vs GPS (NED frame)")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    csv_path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path("data/20241011_NIMBUS24_Flight_FC_Data.csv")
    )

    print(f"Loading {csv_path} ...")
    data = load_flight_csv(csv_path)
    print(f"Loaded {len(data['t'])} samples")

    plot_3d(data)
