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

# FC CSV column indices
_FC_T = 0
_FC_ROLL = 29
_FC_PITCH = 30
_FC_YAW = 31
_FC_BARO = 22
_FC_GPS_ALT = 3
_FC_GPS_LAT = 2


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
    """Configure 3D axis limits and labels."""
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_zlim(-3, 3)
    ax.set_xlabel("East")
    ax.set_ylabel("North")
    ax.set_zlabel("Up")
    ax.set_title(title)


# =============================================================================
# Data Loading
# =============================================================================

def load_eqf_csv(path: Path) -> list[dict]:
    """Load EqF output CSV and convert rotation matrix to Euler angles."""
    rows = []
    with path.open(newline="") as f:
        for raw in csv.DictReader(f):
            row = {k: float(v) for k, v in raw.items()}

            if "r00" in row:
                R = np.array(
                    [
                        [row["r00"], row["r01"], row["r02"]],
                        [row["r10"], row["r11"], row["r12"]],
                        [row["r20"], row["r21"], row["r22"]],
                    ]
                )
                row["yaw(rad)"], row["pitch(rad)"], row["roll(rad)"] = R_to_euler(R)

            if "px" in row:
                row["pn(m)"] = row["px"]
                row["pe(m)"] = row["py"]
                row["pd(m)"] = row["pz"]
            if "vx" in row:
                row["vn(m/s)"] = row["vx"]
                row["ve(m/s)"] = row["vy"]
                row["vd(m/s)"] = row["vz"]
            if "t" in row and "time(s)" not in row:
                row["time(s)"] = row["t"]
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
    """Animate EqF filter output with FC attitude comparison."""
    times = [r["time(s)"] for r in rows]
    alts = [-r["pd(m)"] for r in rows]

    render_indices = select_render_indices(rows, TARGET_FPS)
    render_rows = [rows[i] for i in render_indices]

    fc_t, fc_roll, fc_pitch, fc_yaw, fc_baro, fc_gps = load_fc_attitude(fc_path)

    segs = build_wire()

    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[1, 1])

    ax_eqf = fig.add_subplot(gs[0, 0], projection="3d")
    ax_fc = fig.add_subplot(gs[1, 0], projection="3d")
    ax2d = fig.add_subplot(gs[:, 1])

    lines_eqf = _add_wire_lines(ax_eqf, "steelblue", "tomato", "dimgray")
    lines_fc = _add_wire_lines(ax_fc, "darkorange", "firebrick", "gray")

    _setup_3d_ax(ax_eqf, "TG-EqF Attitude")
    _setup_3d_ax(ax_fc, "FC Attitude (NIMBUS24)")

    (trail_eqf,) = ax2d.plot([], [], color="steelblue", linewidth=1.2, label="EqF alt")
    (trail_fc,) = ax2d.plot([], [], color="darkorange", linewidth=1.0, linestyle="--", label="FC baro alt")
    (trail_gps,) = ax2d.plot([], [], color="green", linewidth=0.8, linestyle=":", label="GPS alt", marker=".", markersize=3)
    cursor = ax2d.axvline(times[0], color="tab:red", linewidth=0.8, linestyle="--")

    ax2d.set_xlabel("Time [s]")
    ax2d.set_ylabel("Altitude [m]")
    ax2d.set_title("Altitude Profile")
    ax2d.set_xlim(times[0], times[-1])
    all_alts = list(alts) + list(fc_baro) + list(fc_gps[np.isfinite(fc_gps)])
    ax2d.set_ylim(min(all_alts) - 50, 3000)
    ax2d.legend(fontsize=8)
    ax2d.grid(True)

    telem_eqf = ax_eqf.text2D(0.02, 0.97, "", transform=ax_eqf.transAxes,
                              fontsize=7, verticalalignment="top", family="monospace")
    telem_fc = ax_fc.text2D(0.02, 0.97, "", transform=ax_fc.transAxes,
                            fontsize=7, verticalalignment="top", family="monospace")

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
        telem_eqf.set_text(
            f"t     = {t:.2f} s\n"
            f"pitch = {math.degrees(pitch):+.1f}°\n"
            f"yaw   = {math.degrees(yaw):+.1f}°\n"
            f"roll  = {math.degrees(roll):+.1f}°\n"
            f"alt   = {-row['pd(m)']:+.1f} m"
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

        row_idx = render_indices[frame_idx]
        trail_eqf.set_data(times[:row_idx + 1], alts[:row_idx + 1])
        fc_mask = fc_t <= t
        trail_fc.set_data(fc_t[fc_mask], fc_baro[fc_mask])
        trail_gps.set_data(fc_t[fc_mask], fc_gps[fc_mask])
        cursor.set_xdata([t, t])

        plt.pause(0.001)

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    csv_path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path("outputs/tg_eqf_output.csv")
    )

    if not csv_path.exists():
        print(f"{csv_path} not found — run python eqf_filter.py first")
        sys.exit(1)

    rows = load_eqf_csv(csv_path)
    print(f"Loaded {len(rows)} EqF rows")
    animate(rows, realtime=True)
