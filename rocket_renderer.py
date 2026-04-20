"""Live 3D rocket attitude visualiser – replays NIMBUS24 FC CSV."""

import csv
import math
import threading
import time

import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# Configuration
# =============================================================================

CSV_PATH = "data/20241011_NIMBUS24_Flight_FC_Data.csv"

TELEMETRY_FIELDS = [
    "time(s)",
    "roll(rad)",
    "pitch(rad)",
    "yaw(rad)",
    "baro_alt(m)",
    "az(g)",
]


# =============================================================================
# Thread-Safe State
# =============================================================================


class State:
    """Thread-safe telemetry state container."""

    def __init__(self):
        """Initialize empty state with lock."""
        self._data: dict = {}
        self._lock = threading.Lock()

    def set(self, values: dict) -> None:
        """Update state with new values."""
        with self._lock:
            self._data.update(values)

    def get(self) -> dict:
        """Get current state snapshot."""
        with self._lock:
            return dict(self._data)


state = State()


# =============================================================================
# Field Parsing
# =============================================================================


def parse_field(name: str, raw: str) -> float | None:
    """Parse field value with unit conversion (mm→m, g→m/s²)."""
    try:
        v = float(raw)
    except (ValueError, TypeError):
        return None
    if not math.isfinite(v):
        return None
    if "(mm)" in name:
        return v / 1000.0
    if "(g)" in name:
        return v * 9.81
    return v


# =============================================================================
# Data Stream
# =============================================================================


def data_stream(filepath: str) -> None:
    """Read CSV file in real-time, updating shared state."""
    prev_t: float | None = None
    last: dict = {}

    with open(filepath, newline="") as f:
        for row in csv.DictReader(f):
            if not row.get("time(s)"):
                continue
            current: dict = {}
            for field in TELEMETRY_FIELDS:
                v = parse_field(field, row.get(field, ""))
                current[field] = v if v is not None else last.get(field, 0.0)
                last[field] = current[field]

            state.set(current)

            t = current.get("time(s)", 0.0)
            if prev_t is not None and t > prev_t:
                time.sleep(t - prev_t)
            prev_t = t


# =============================================================================
# Rocket Geometry
# =============================================================================


def _build_rocket():
    """Build rocket surface meshes (nose at +Z, tail at −Z)."""
    th = np.linspace(0, 2 * np.pi, 30)

    thg, zb = np.meshgrid(th, np.linspace(-2.0, 1.5, 30))
    body = (0.3 * np.cos(thg), 0.3 * np.sin(thg), zb)

    thg, zn = np.meshgrid(th, np.linspace(1.5, 2.5, 20))
    r_nose = (2.5 - zn) * 0.3
    nose = (r_nose * np.cos(thg), r_nose * np.sin(thg), zn)

    fins = []
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        fins.append(
            np.array(
                [
                    [dx * 0.3, dy * 0.3, -2.0],
                    [dx * 0.8, dy * 0.8, -2.6],
                    [dx * 0.3, dy * 0.3, -1.7],
                ]
            )
        )

    return body, nose, fins


BODY, NOSE, FINS = _build_rocket()


# =============================================================================
# Attitude Rotation
# =============================================================================
# Convention: pitch = elevation, yaw = compass heading (NED), roll = rocket axis
# Sequence: Rz_ned(yaw) @ Rx(pitch − π/2) @ Rz_ned(roll)


def _rz_ned(a: float) -> np.ndarray:
    """Clockwise rotation around Z (NED compass convention)."""
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])


def _rx(a: float) -> np.ndarray:
    """Rotation around X axis."""
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def attitude_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Body-to-world rotation. Nose (+Z body) is up when pitch=π/2."""
    return _rz_ned(yaw) @ _rx(pitch - np.pi / 2) @ _rz_ned(roll)


# =============================================================================
# Geometry Helpers
# =============================================================================


def _rotate_mesh(R: np.ndarray, x, y, z):
    """Rotate mesh coordinates by rotation matrix."""
    pts = R @ np.stack([x.ravel(), y.ravel(), z.ravel()])
    return pts[0].reshape(x.shape), pts[1].reshape(y.shape), pts[2].reshape(z.shape)


def _rotate_fin(R: np.ndarray, fin: np.ndarray) -> np.ndarray:
    """Rotate fin vertices by rotation matrix."""
    return (R @ fin.T).T


# =============================================================================
# Render Loop
# =============================================================================

threading.Thread(target=data_stream, args=(CSV_PATH,), daemon=True).start()
plt.ion()
fig = plt.figure(figsize=(6, 8))
ax = fig.add_subplot(111, projection="3d")

while True:
    elev, azim = ax.elev, ax.azim
    ax.clear()
    data = state.get()

    yaw_r = data.get("yaw(rad)", 0.0)
    pitch_r = data.get("pitch(rad)", math.radians(84.7))
    roll_r = data.get("roll(rad)", 0.0)

    R = attitude_matrix(yaw_r, pitch_r, roll_r)

    bx, by, bz = _rotate_mesh(R, *BODY)
    ax.plot_surface(bx, by, bz, color="steelblue", alpha=0.85, linewidth=0)

    nx, ny, nz = _rotate_mesh(R, *NOSE)
    ax.plot_surface(nx, ny, nz, color="tomato", alpha=0.85, linewidth=0)

    for fin in FINS:
        rf = _rotate_fin(R, fin)
        ax.plot_trisurf(rf[:, 0], rf[:, 1], rf[:, 2], color="dimgray", alpha=0.9)

    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_zlim(-3, 3)
    ax.set_xlabel("East")
    ax.set_ylabel("North")
    ax.set_zlabel("Up")
    ax.set_title("NIMBUS24 Attitude")
    ax.view_init(elev=elev, azim=azim)

    lines = []
    for field in TELEMETRY_FIELDS:
        val = data.get(field, 0.0)
        if "(rad)" in field:
            label = field.replace("(rad)", "")
            lines.append(f"{label}: {math.degrees(val):+.1f}°")
        elif "(g)" in field:
            label = field.replace("(g)", "")
            lines.append(f"{label}: {val:.2f} m/s²")
        elif "(mm)" in field:
            label = field.replace("(mm)", "")
            lines.append(f"{label}: {val:.1f} m")
        else:
            lines.append(f"{field}: {val:.2f}")

    ax.text2D(
        0.02,
        0.95,
        "\n".join(lines),
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        family="monospace",
    )

    plt.pause(0.02)
