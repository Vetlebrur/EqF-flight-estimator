"""Visualize TG-EqF filter estimates – trajectory, attitude, and velocity."""

import csv
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


# =============================================================================
# Configuration
# =============================================================================

ESTIMATE_CSV = Path("eqf_estimates.csv")
FC_CSV = Path("data/20241011_NIMBUS24_Flight_FC_Data.csv")


# =============================================================================
# Geometry and Math
# =============================================================================

def R_to_euler(R):
    """Extract Euler angles from rotation matrix (NED convention)."""
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
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_zlim(-50, 50)
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_zlabel("Up (m)")
    ax.set_title(title)


# =============================================================================
# Data Loading
# =============================================================================

def load_eqf_estimates(path: Path) -> tuple:
    """Load EqF estimate CSV and extract pose, velocity, and bias."""
    times, positions, velocities, rotations, biases = [], [], [], [], []

    with path.open(newline='') as f:
        for row in csv.DictReader(f):
            times.append(float(row['t']))
            positions.append([float(row['px']), float(row['py']), float(row['pz'])])
            velocities.append([float(row['vx']), float(row['vy']), float(row['vz'])])

            R = np.array([
                [float(row['r00']), float(row['r01']), float(row['r02'])],
                [float(row['r10']), float(row['r11']), float(row['r12'])],
                [float(row['r20']), float(row['r21']), float(row['r22'])],
            ])
            rotations.append(R)

            biases.append([
                float(row['b_gx']), float(row['b_gy']), float(row['b_gz']),
                float(row['b_ax']), float(row['b_ay']), float(row['b_az']),
                float(row['b_mu_x']), float(row['b_mu_y']), float(row['b_mu_z']),
            ])

    return np.array(times), np.array(positions), np.array(velocities), rotations, np.array(biases)


# =============================================================================
# Visualization
# =============================================================================

def plot_trajectory_3d(ax, positions):
    """Plot 3D trajectory."""
    if len(positions) > 0:
        ax.plot(positions[:, 1], positions[:, 0], -positions[:, 2],
                'b-', alpha=0.6, linewidth=1, label='EqF Trajectory')
        ax.scatter(positions[0, 1], positions[0, 0], -positions[0, 2],
                   color='green', s=100, marker='o', label='Start')
        ax.scatter(positions[-1, 1], positions[-1, 0], -positions[-1, 2],
                   color='red', s=100, marker='s', label='End')


def plot_attitude(ax, times, rotations):
    """Plot Euler angles over time."""
    yaws, pitches, rolls = [], [], []
    for R in rotations:
        yaw, pitch, roll = R_to_euler(R)
        yaws.append(np.degrees(yaw))
        pitches.append(np.degrees(pitch))
        rolls.append(np.degrees(roll))

    ax.plot(times, rolls, 'r-', alpha=0.7, label='Roll')
    ax.plot(times, pitches, 'g-', alpha=0.7, label='Pitch')
    ax.plot(times, yaws, 'b-', alpha=0.7, label='Yaw')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angle (deg)')
    ax.set_title('Estimated Attitude')
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_velocity(ax, times, velocities):
    """Plot velocity components over time."""
    ax.plot(times, velocities[:, 0], 'r-', alpha=0.7, label='Vn (North)')
    ax.plot(times, velocities[:, 1], 'g-', alpha=0.7, label='Ve (East)')
    ax.plot(times, velocities[:, 2], 'b-', alpha=0.7, label='Vd (Down)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title('Estimated Velocity')
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_position(ax, times, positions):
    """Plot position components over time."""
    ax.plot(times, positions[:, 0], 'r-', alpha=0.7, label='Pn (North)')
    ax.plot(times, positions[:, 1], 'g-', alpha=0.7, label='Pe (East)')
    ax.plot(times, -positions[:, 2], 'b-', alpha=0.7, label='Altitude')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position (m)')
    ax.set_title('Estimated Position')
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_bias(ax, times, biases):
    """Plot bias estimates over time."""
    ax.plot(times, biases[:, 0], 'r-', alpha=0.7, label='Gyro X')
    ax.plot(times, biases[:, 1], 'g-', alpha=0.7, label='Gyro Y')
    ax.plot(times, biases[:, 2], 'b-', alpha=0.7, label='Gyro Z')
    ax.plot(times, biases[:, 3], 'r--', alpha=0.7, label='Accel X')
    ax.plot(times, biases[:, 4], 'g--', alpha=0.7, label='Accel Y')
    ax.plot(times, biases[:, 5], 'b--', alpha=0.7, label='Accel Z')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Bias (units)')
    ax.set_title('Estimated IMU Bias (Gyroscope + Accelerometer)')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)


def main():
    """Load and visualize EqF estimates."""
    if not ESTIMATE_CSV.exists():
        print(f"Error: {ESTIMATE_CSV} not found. Run eqf_loop.py first.")
        return

    print(f"Loading estimates from {ESTIMATE_CSV}...")
    times, positions, velocities, rotations, biases = load_eqf_estimates(ESTIMATE_CSV)

    print(f"Loaded {len(times)} estimates")
    print(f"Time range: {times[0]:.2f}s to {times[-1]:.2f}s")

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(3, 2, figure=fig)

    # 3D trajectory
    ax_3d = fig.add_subplot(gs[0, :], projection='3d')
    _setup_3d_ax(ax_3d, 'EqF Estimated Trajectory')
    plot_trajectory_3d(ax_3d, positions)
    ax_3d.legend()

    # Attitude over time
    ax_att = fig.add_subplot(gs[1, 0])
    plot_attitude(ax_att, times, rotations)

    # Velocity over time
    ax_vel = fig.add_subplot(gs[1, 1])
    plot_velocity(ax_vel, times, velocities)

    # Position over time
    ax_pos = fig.add_subplot(gs[2, 0])
    plot_position(ax_pos, times, positions)

    # Bias over time
    ax_bias = fig.add_subplot(gs[2, 1])
    plot_bias(ax_bias, times, biases)

    plt.tight_layout()
    plt.savefig('eqf_estimates_visualization.png', dpi=150, bbox_inches='tight')
    print("Visualization saved to eqf_estimates_visualization.png")
    plt.show()


if __name__ == '__main__':
    main()
