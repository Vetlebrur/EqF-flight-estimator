"""Replay and animate TG-EqF filter output – EqF attitude vs FC attitude."""

import csv
import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


# =============================================================================
# Configuration
# =============================================================================

TARGET_FPS = 25
FC_CSV = Path("data/20241011_NIMBUS24_Flight_FC_Data.csv")

# Default to combined data (static + flight)
DEFAULT_EQF_CSV = Path("outputs/tg_eqf_output.csv")

# FC CSV column indices
_FC_T = 0
_FC_ROLL = 29
_FC_PITCH = 30
_FC_YAW = 31
_FC_BARO = 22
_FC_GPS_ALT = 3
_FC_GPS_LAT = 2
_FC_GPS_LON = 1
_FC_FC_PN = 36
_FC_FC_PE = 37
_FC_FC_PD = 38

R_EARTH = 6_378_137.0


# =============================================================================
# Geometry and Math
# =============================================================================

def _rz_ned(a):
    """Clockwise rotation around Z (NED convention)."""
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])


def _rx(a):
    """Rotation around X axis."""
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def attitude_matrix(yaw, pitch, roll):
    """Body-to-world rotation. Nose (+Z body) points up when pitch = π/2."""
    return _rz_ned(yaw) @ _rx(pitch - math.pi / 2) @ _rz_ned(roll)


def R_to_euler(R):
    """Extract Euler angles from rotation matrix."""
    pitch = math.asin(max(-1.0, min(1.0, -R[2, 0])))
    yaw = math.atan2(R[1, 0], R[0, 0])
    roll = math.atan2(R[2, 1], R[2, 2])
    return yaw, pitch, roll


def build_wire():
    """Wire-frame rocket segments."""
    th = np.linspace(0, 2 * np.pi, 17)
    r = 0.3
    segs = []

    for z in np.linspace(-2.0, 1.5, 5):
        segs.append(np.array([r * np.cos(th), r * np.sin(th), np.full(17, z)]))
    for i in range(0, 16, 4):
        segs.append(
            np.array(
                [
                    [r * np.cos(th[i])] * 2,
                    [r * np.sin(th[i])] * 2,
                    [-2.0, 1.5],
                ]
            )
        )

    segs.append(np.array([r * np.cos(th), r * np.sin(th), np.full(17, 1.5)]))
    for i in range(0, 16, 4):
        segs.append(
            np.array(
                [
                    [r * np.cos(th[i]), 0],
                    [r * np.sin(th[i]), 0],
                    [1.5, 2.5],
                ]
            )
        )

    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        segs.append(
            np.array(
                [
                    [dx * 0.3, dx * 0.8, dx * 0.3, dx * 0.3],
                    [dy * 0.3, dy * 0.8, dy * 0.3, dy * 0.3],
                    [-2.0, -2.6, -1.7, -2.0],
                ]
            )
        )
    return segs


def _add_wire_lines(ax, body_color, nose_color, fin_color):
    """Create Line3D objects for one rocket."""
    lines = []
    for _ in range(9):
        (l,) = ax.plot([], [], [], color=body_color, linewidth=1)
        lines.append(l)
    for _ in range(9, 14):
        (l,) = ax.plot([], [], [], color=nose_color, linewidth=1)
        lines.append(l)
    for _ in range(14, 18):
        (l,) = ax.plot([], [], [], color=fin_color, linewidth=1)
        lines.append(l)
    return lines


def _update_wire(lines, segs, R):
    """Update wire frame positions for given rotation."""
    for i, seg in enumerate(segs):
        rotated = R @ seg
        lines[i].set_data_3d(rotated[0], rotated[1], rotated[2])


def _setup_3d_ax(ax, title):
    """Configure 3D axis limits and labels (ENU frame)."""
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_zlim(-3, 3)
    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.set_zlabel("Up [m]")
    ax.set_title(title)


# =============================================================================
# Data Loading
# =============================================================================

def load_eqf_csv(path: Path) -> list[dict]:
    """Load EqF output CSV with position, velocity, and Euler angles."""
    rows = []
    with path.open(newline="") as f:
        for raw in csv.DictReader(f):
            row = {k: float(v) for k, v in raw.items()}

            # Map position columns
            if "px" in row:
                row["pn(m)"] = row["px"]
                row["pe(m)"] = row["py"]
                row["pd(m)"] = row["pz"]

            # Map velocity columns
            if "vx" in row:
                row["vn(m/s)"] = row["vx"]
                row["ve(m/s)"] = row["vy"]
                row["vd(m/s)"] = row["vz"]

            # Map time column
            if "t" in row and "time(s)" not in row:
                row["time(s)"] = row["t"]

            # Euler angles already in CSV (roll, pitch, yaw in radians, unwrapped)
            # Rename columns to match expected format
            if "roll" in row:
                row["roll(rad)"] = row["roll"]
            else:
                row["roll(rad)"] = 0.0
            if "pitch" in row:
                row["pitch(rad)"] = row["pitch"]
            else:
                row["pitch(rad)"] = 0.0
            if "yaw" in row:
                row["yaw(rad)"] = row["yaw"]
            else:
                row["yaw(rad)"] = 0.0

            row.setdefault("std_pn(m)", 0.0)

            rows.append(row)
    return rows


def load_fc_attitude(path: Path):
    """Load FC attitude data (time, roll, pitch, yaw, baro_alt, gps_alt)."""
    raw = np.genfromtxt(
        path,
        delimiter=",",
        skip_header=1,
        usecols=(_FC_T, _FC_ROLL, _FC_PITCH, _FC_YAW, _FC_BARO, _FC_GPS_ALT, _FC_GPS_LAT),
    )
    valid = np.isfinite(raw).all(axis=1)
    raw = raw[valid]
    gps_alt = np.where(raw[:, 6] != 0, raw[:, 5] / 1000.0, np.nan)
    return raw[:, 0], raw[:, 1], raw[:, 2], raw[:, 3], raw[:, 4], gps_alt


def _gps_to_ned(lat, lon, alt, lat0, lon0, alt0):
    """Convert GPS (lat/lon/alt) to NED relative to reference."""
    dlat = np.radians(lat - lat0)
    dlon = np.radians(lon - lon0)
    lat0_rad = np.radians(lat0)

    north = dlat * R_EARTH
    east = dlon * np.cos(lat0_rad) * R_EARTH
    down = -(alt - alt0)
    return north, east, down


def load_fc_trajectory(path: Path):
    """Load FC trajectory (pn, pe, pd) from FC CSV."""
    raw = np.genfromtxt(
        path,
        delimiter=",",
        skip_header=1,
        usecols=(_FC_FC_PN, _FC_FC_PE, _FC_FC_PD),
    )
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    return raw[:, 0], raw[:, 1], raw[:, 2]


def load_gps_trajectory(path: Path):
    """Load and convert GPS trajectory to NED coordinates."""
    raw = np.genfromtxt(
        path,
        delimiter=",",
        skip_header=1,
        usecols=(_FC_GPS_LAT, _FC_GPS_LON, _FC_GPS_ALT),
    )
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)

    lat = raw[:, 0]
    lon = raw[:, 1]
    alt = raw[:, 2] / 1000.0

    valid = (lat != 0) & (lon != 0)
    first_valid = np.argmax(valid)
    lat0, lon0, alt0 = lat[first_valid], lon[first_valid], alt[first_valid]

    gps_n, gps_e, gps_d = _gps_to_ned(lat, lon, alt, lat0, lon0, alt0)

    gps_n = np.where(valid, gps_n, np.nan)
    gps_e = np.where(valid, gps_e, np.nan)
    gps_d = np.where(valid, gps_d, np.nan)

    return gps_n, gps_e, gps_d


def select_render_indices(rows, target_fps):
    """Select frame indices for rendering at target FPS."""
    if not rows:
        return []
    times = [r["time(s)"] for r in rows]
    dt_sim = (times[-1] - times[0]) / len(rows)
    step = max(1, round((1.0 / target_fps) / dt_sim))
    indices = list(range(0, len(rows), step))
    if indices[-1] != len(rows) - 1:
        indices.append(len(rows) - 1)
    return indices


# =============================================================================
# Animation
# =============================================================================

def animate(rows: list[dict], fc_path: Path = FC_CSV, realtime: bool = True) -> None:
    """Animate EqF filter output with FC attitude comparison and 3D trajectory (ENU frame)."""
    # Load NED data
    pn_eqf = [r["pn(m)"] for r in rows]
    pe_eqf = [r["pe(m)"] for r in rows]
    pd_eqf = [r["pd(m)"] for r in rows]

    pn_fc, pe_fc, pd_fc = load_fc_trajectory(fc_path)

    gps_n, gps_e, gps_d = load_gps_trajectory(fc_path)

    # Convert from NED to ENU: negate down (z) to get up
    pu_eqf = np.array([-d for d in pd_eqf])  # Up = -Down
    pu_fc = np.array([-d for d in pd_fc])
    gps_u = np.array([-d for d in gps_d])

    render_indices = select_render_indices(rows, TARGET_FPS)
    render_rows = [rows[i] for i in render_indices]

    fc_t, fc_roll, fc_pitch, fc_yaw, _, _ = load_fc_attitude(fc_path)

    segs = build_wire()

    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[1, 1])

    ax_eqf = fig.add_subplot(gs[0, 0], projection="3d")
    ax_fc = fig.add_subplot(gs[1, 0], projection="3d")
    ax_traj = fig.add_subplot(gs[:, 1], projection="3d")

    lines_eqf = _add_wire_lines(ax_eqf, "steelblue", "tomato", "dimgray")
    lines_fc = _add_wire_lines(ax_fc, "darkorange", "firebrick", "gray")

    _setup_3d_ax(ax_eqf, "TG-EqF Attitude")
    _setup_3d_ax(ax_fc, "FC Attitude (NIMBUS24)")

    ax_traj.set_xlabel("East [m]")
    ax_traj.set_ylabel("North [m]")
    ax_traj.set_zlabel("Up [m]")
    ax_traj.set_title("3D Trajectory Comparison (ENU)")

    (traj_eqf,) = ax_traj.plot([], [], [], color="steelblue", linewidth=1.5, label="EqF Estimate")
    (traj_fc,) = ax_traj.plot([], [], [], color="darkorange", linewidth=1.5, label="FC Estimate")
    (traj_gps,) = ax_traj.plot([], [], [], color="green", linewidth=1.0, linestyle="--", label="GPS", marker=".", markersize=2)
    (pos_eqf,) = ax_traj.plot([], [], [], "o", color="steelblue", markersize=8)
    (pos_fc,) = ax_traj.plot([], [], [], "s", color="darkorange", markersize=8)
    ax_traj.legend(fontsize=8)

    telem_eqf = ax_eqf.text2D(0.02, 0.97, "", transform=ax_eqf.transAxes,
                              fontsize=7, verticalalignment="top", family="monospace")
    telem_fc = ax_fc.text2D(0.02, 0.97, "", transform=ax_fc.transAxes,
                            fontsize=7, verticalalignment="top", family="monospace")

    # Auto-scale trajectory axes based on all trajectory bounds (ENU)
    all_n = list(pn_eqf) + list(pn_fc) + list(gps_n[~np.isnan(gps_n)])
    all_e = list(pe_eqf) + list(pe_fc) + list(gps_e[~np.isnan(gps_e)])
    all_u = list(pu_eqf) + list(pu_fc) + list(gps_u[~np.isnan(gps_u)])

    pn_min, pn_max = min(all_n), max(all_n)
    pe_min, pe_max = min(all_e), max(all_e)
    pu_min, pu_max = min(all_u), max(all_u)
    margin = 100
    ax_traj.set_xlim(pe_min - margin, pe_max + margin)
    ax_traj.set_ylim(pn_min - margin, pn_max + margin)
    ax_traj.set_zlim(pu_min - margin, pu_max + margin)

    fig.tight_layout()
    plt.ion()
    plt.show(block=False)

    prev_wall = time.perf_counter()
    prev_t = render_rows[0]["time(s)"]

    for frame_idx, row in enumerate(render_rows):
        t = row["time(s)"]

        if realtime:
            sim_dt = t - prev_t
            wall_dt = time.perf_counter() - prev_wall
            gap = sim_dt - wall_dt
            if gap > 0.001:
                time.sleep(gap)
            prev_wall = time.perf_counter()
            prev_t = t

        yaw, pitch, roll = row["yaw(rad)"], row["pitch(rad)"], row["roll(rad)"]
        _update_wire(lines_eqf, segs, attitude_matrix(yaw, pitch, roll))
        # ENU altitude: negate down to get up
        altitude_up = -row['pd(m)']
        telem_eqf.set_text(
            f"t     = {t:.2f} s\n"
            f"pitch = {math.degrees(pitch):+.1f}°\n"
            f"yaw   = {math.degrees(yaw):+.1f}°\n"
            f"roll  = {math.degrees(roll):+.1f}°\n"
            f"alt   = {altitude_up:+.1f} m"
        )

        fc_idx = int(np.searchsorted(fc_t, t, side="left"))
        fc_idx = min(fc_idx, len(fc_t) - 1)
        fc_p, fc_y, fc_r = fc_pitch[fc_idx], fc_yaw[fc_idx], fc_roll[fc_idx]
        _update_wire(lines_fc, segs, attitude_matrix(fc_y, fc_p, fc_r))
        telem_fc.set_text(
            f"pitch = {math.degrees(fc_p):+.1f}°\n"
            f"yaw   = {math.degrees(fc_y):+.1f}°\n"
            f"roll  = {math.degrees(fc_r):+.1f}°"
        )

        # Update trajectory plot (ENU coordinates)
        row_idx = render_indices[frame_idx]
        traj_eqf.set_data(pe_eqf[:row_idx + 1], pn_eqf[:row_idx + 1])
        traj_eqf.set_3d_properties(pu_eqf[:row_idx + 1])
        traj_fc.set_data(pe_fc[:row_idx + 1], pn_fc[:row_idx + 1])
        traj_fc.set_3d_properties(pu_fc[:row_idx + 1])
        traj_gps.set_data(gps_e[:row_idx + 1], gps_n[:row_idx + 1])
        traj_gps.set_3d_properties(gps_u[:row_idx + 1])
        pos_eqf.set_data([pe_eqf[row_idx]], [pn_eqf[row_idx]])
        pos_eqf.set_3d_properties([pu_eqf[row_idx]])
        pos_fc.set_data([pe_fc[row_idx]], [pn_fc[row_idx]])
        pos_fc.set_3d_properties([pu_fc[row_idx]])

        plt.pause(0.001)

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    # Determine which CSV to use
    if len(sys.argv) > 1:
        csv_path = Path(sys.argv[1])
        data_type = "custom"
    else:
        # Default to combined static+flight data
        csv_path = DEFAULT_EQF_CSV
        if csv_path.exists():
            data_type = "COMBINED (30s static + flight)"
        else:
            csv_path = Path("outputs/tg_eqf_output.csv")
            data_type = "FLIGHT ONLY"

    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found")
        print()
        print("Options:")
        print("  python eqf_renderer.py")
        print(f"    Loads: {DEFAULT_EQF_CSV}")
        print()
        print("  python eqf_renderer.py outputs/tg_eqf_output.csv")
        print("    Loads: Flight-only data")
        print()
        print("Required files:")
        print("  - EqF output CSV (run: python eqf_filter.py)")
        print("  - FC data (data/20241011_NIMBUS24_Flight_FC_Data.csv)")
        sys.exit(1)

    print("=" * 70)
    print("ROCKET ATTITUDE & TRAJECTORY RENDERER (ENU FRAME)")
    print("=" * 70)
    print(f"EqF Data: {data_type}")
    print(f"File:     {csv_path}")
    print()

    rows = load_eqf_csv(csv_path)
    print(f"Loaded {len(rows)} EqF rows")
    print()
    print("Comparison:")
    print("  Left-top:    TG-EqF Filter Estimate (blue rocket)")
    print("  Left-bottom: Flight Computer Estimate (orange rocket)")
    print("  Right:       3D Trajectory (EqF blue, FC orange, GPS green)")
    print()
    print("=" * 70)
    print()

    animate(rows, realtime=True)
