"""Compare filter output with GNSS ground truth and compute error metrics."""

import numpy as np
import csv
from pathlib import Path

# Constants
R_EARTH = 6_378_137.0
_C = {
    "t": 0,
    "lon": 1,
    "lat": 2,
    "alt": 3,
    "gps_vn": 4,
    "gps_ve": 5,
    "gps_vd": 6,
    "ax": 9,
    "ay": 10,
    "az": 11,
    "gx": 15,
    "gy": 16,
    "gz": 17,
}

def gps_to_ned(lat, lon, alt, lat0, lon0, alt0):
    """Convert GPS (lat/lon/alt) to NED relative to reference."""
    dlat = np.radians(lat - lat0)
    dlon = np.radians(lon - lon0)
    lat0_rad = np.radians(lat0)
    north = dlat * R_EARTH
    east = dlon * np.cos(lat0_rad) * R_EARTH
    down = -(alt - alt0)  # Positive down (negative altitude difference)
    return np.array([north, east, down])

def load_gnss_data(csv_path):
    """Load GNSS data from raw CSV."""
    raw = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)

    # Find first valid GPS fix for reference
    valid = (raw[:, _C["lat"]] != 0) & (raw[:, _C["lon"]] != 0)
    first = np.argmax(valid)
    lat0 = raw[first, _C["lat"]]
    lon0 = raw[first, _C["lon"]]
    alt0 = raw[first, _C["alt"]] / 1000.0

    # Extract GNSS data
    gnss_t = []
    gnss_pos = []
    gnss_vel = []

    for row in raw:
        t = row[_C["t"]]
        if not np.isfinite(t):
            continue

        lat = row[_C["lat"]]
        lon = row[_C["lon"]]

        if lat != 0 and lon != 0:
            alt = row[_C["alt"]] / 1000.0
            pos_ned = gps_to_ned(lat, lon, alt, lat0, lon0, alt0)
            vel_ned = np.array([row[_C["gps_vn"]], row[_C["gps_ve"]], row[_C["gps_vd"]]]) / 1000.0
            gnss_t.append(t)
            gnss_pos.append(pos_ned)
            gnss_vel.append(vel_ned)

    return np.array(gnss_t), np.array(gnss_pos), np.array(gnss_vel)

def load_filter_output(csv_path):
    """Load filter output CSV."""
    filt_t = []
    filt_pos = []
    filt_vel = []

    with open(csv_path) as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            t = float(row[0])
            p = np.array([float(row[1]), float(row[2]), float(row[3])])
            v = np.array([float(row[4]), float(row[5]), float(row[6])])
            filt_t.append(t)
            filt_pos.append(p)
            filt_vel.append(v)

    return np.array(filt_t), np.array(filt_pos), np.array(filt_vel)

def interpolate_gnss(gnss_t, gnss_pos, gnss_vel, filt_t):
    """Interpolate GNSS data to filter timestamps."""
    gnss_pos_interp = np.zeros((len(filt_t), 3))
    gnss_vel_interp = np.zeros((len(filt_t), 3))

    for i, t in enumerate(filt_t):
        # Find surrounding GNSS points
        idx = np.searchsorted(gnss_t, t)
        if idx == 0 or idx == len(gnss_t):
            continue  # Outside GNSS coverage

        # Linear interpolation
        t0, t1 = gnss_t[idx - 1], gnss_t[idx]
        w = (t - t0) / (t1 - t0)

        gnss_pos_interp[i] = (1 - w) * gnss_pos[idx - 1] + w * gnss_pos[idx]
        gnss_vel_interp[i] = (1 - w) * gnss_vel[idx - 1] + w * gnss_vel[idx]

    return gnss_pos_interp, gnss_vel_interp

def compute_metrics(filt_t, filt_pos, filt_vel, gnss_t, gnss_pos, gnss_vel):
    """Compute error metrics comparing filter to GNSS."""

    # Interpolate GNSS to filter timestamps
    gnss_pos_interp, gnss_vel_interp = interpolate_gnss(gnss_t, gnss_pos, gnss_vel, filt_t)

    # Position errors
    pos_error = filt_pos - gnss_pos_interp
    pos_error_mag = np.linalg.norm(pos_error, axis=1)

    # Velocity errors
    vel_error = filt_vel - gnss_vel_interp
    vel_error_mag = np.linalg.norm(vel_error, axis=1)

    # Filter out periods outside GNSS coverage (where gnss_pos_interp is zero)
    valid = np.any(gnss_pos_interp != 0, axis=1)

    if not np.any(valid):
        print("ERROR: No valid GNSS data for comparison")
        return None

    pos_error_valid = pos_error_mag[valid]
    vel_error_valid = vel_error_mag[valid]
    filt_t_valid = filt_t[valid]

    # Compute metrics
    metrics = {
        "pos_rms": np.sqrt(np.mean(pos_error_valid**2)),
        "pos_mean": np.mean(pos_error_valid),
        "pos_max": np.max(pos_error_valid),
        "pos_std": np.std(pos_error_valid),
        "vel_rms": np.sqrt(np.mean(vel_error_valid**2)),
        "vel_mean": np.mean(vel_error_valid),
        "vel_max": np.max(vel_error_valid),
        "vel_std": np.std(vel_error_valid),
    }

    # Drift rate (error growth per unit time)
    if len(filt_t_valid) > 1:
        drift_rate = (pos_error_valid[-1] - pos_error_valid[0]) / (filt_t_valid[-1] - filt_t_valid[0])
        metrics["drift_rate"] = drift_rate

    # Error components (North, East, Down)
    metrics["pos_error_north_rms"] = np.sqrt(np.mean(pos_error[valid, 0]**2))
    metrics["pos_error_east_rms"] = np.sqrt(np.mean(pos_error[valid, 1]**2))
    metrics["pos_error_down_rms"] = np.sqrt(np.mean(pos_error[valid, 2]**2))

    return metrics, pos_error_valid, filt_t_valid

def main():
    # Load data
    print("Loading data...")
    gnss_t, gnss_pos, gnss_vel = load_gnss_data("data/20241011_NIMBUS24_Flight_FC_Data.csv")
    filt_t, filt_pos, filt_vel = load_filter_output("outputs/tg_eqf_output.csv")

    print(f"GNSS measurements: {len(gnss_t)} points")
    print(f"Filter output: {len(filt_t)} points")

    # Compute metrics
    print("\nComputing error metrics...")
    result = compute_metrics(filt_t, filt_pos, filt_vel, gnss_t, gnss_pos, gnss_vel)

    if result is None:
        return

    metrics, pos_errors, times = result

    # Print metrics
    print("\n" + "="*60)
    print("FILTER PERFORMANCE METRICS")
    print("="*60)
    print(f"\nPosition Error:")
    print(f"  RMS:     {metrics['pos_rms']:8.2f} m")
    print(f"  Mean:    {metrics['pos_mean']:8.2f} m")
    print(f"  Max:     {metrics['pos_max']:8.2f} m")
    print(f"  Std:     {metrics['pos_std']:8.2f} m")

    print(f"\nPosition Error by Component:")
    print(f"  North RMS: {metrics['pos_error_north_rms']:8.2f} m")
    print(f"  East RMS:  {metrics['pos_error_east_rms']:8.2f} m")
    print(f"  Down RMS:  {metrics['pos_error_down_rms']:8.2f} m")

    print(f"\nVelocity Error:")
    print(f"  RMS:     {metrics['vel_rms']:8.2f} m/s")
    print(f"  Mean:    {metrics['vel_mean']:8.2f} m/s")
    print(f"  Max:     {metrics['vel_max']:8.2f} m/s")
    print(f"  Std:     {metrics['vel_std']:8.2f} m/s")

    if "drift_rate" in metrics:
        print(f"\nDrift Rate: {metrics['drift_rate']:8.4f} m/s")

    print("\n" + "="*60)

    # Overall quality score (lower is better)
    quality_score = metrics['pos_rms'] + 0.1 * metrics['vel_rms']
    print(f"\nQuality Score (RMS pos + 0.1*RMS vel): {quality_score:.2f}")
    print("(Lower is better)")

if __name__ == "__main__":
    main()
