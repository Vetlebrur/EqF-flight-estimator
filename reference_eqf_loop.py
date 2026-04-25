"""
Run the reference SE23_se23_EqF implementation on flight data for comparison.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'eqf-reference'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'eqf-reference', 'Simulation'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'eqf-reference', 'Symmetries', 'Calibrated', 'SE23_se23'))

import numpy as np
import csv
from pathlib import Path
from Filters.Calibrated.SE23_se23_EqF import SE23_se23_EqF
from Symmetries.Calibrated.SE23_se23.Symmetry import *
from update_processor import UpdateProcessor


def main():
    """
    Run reference implementation on flight data.
    """
    print("=" * 70)
    print("Reference SE23_se23_EqF - Flight Data Processing")
    print("=" * 70)

    # Initialize reference filter
    filter_ref = SE23_se23_EqF(
        initial_att_noise=1.0,
        initial_vel_noise=1.0,
        initial_pos_noise=1.0,
        initial_bias_noise=0.01,
        propagationonly=True,  # IMU-only, no measurement updates
        measure_b_mu=False
    )
    print(f"\nReference filter initialized")

    # Initialize data processor
    csv_path = r"data\20241011_NIMBUS24_Flight_FC_Data.csv"
    processor = UpdateProcessor(csv_path)
    print(f"Loading flight data from: {csv_path}")

    # Initialize CSV writer for reference estimates
    output_csv = Path("eqf_estimates_reference.csv")
    csv_file = output_csv.open('w', newline='')
    fieldnames = ['t', 'px', 'py', 'pz', 'vx', 'vy', 'vz',
                  'r00', 'r01', 'r02', 'r10', 'r11', 'r12', 'r20', 'r21', 'r22',
                  'b_gx', 'b_gy', 'b_gz', 'b_ax', 'b_ay', 'b_az', 'b_mu_x', 'b_mu_y', 'b_mu_z']
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    csv_writer.writeheader()

    # Noise parameters for reference implementation
    gyro_noise = 0.05    # rad/s
    accel_noise = 0.1    # m/s²
    bias_noise = 1e-4    # per sqrt(second)
    virtual_noise = 0.1  # virtual input noise

    stats = {
        'total_rows': 0,
        'imu_predicts': 0,
    }

    print("\nProcessing flight data...")
    print("-" * 70)

    last_timestamp = None

    try:
        while processor.has_data():
            row_idx, row = processor.get_next_row()
            timestamp = row['time(s)']
            stats['total_rows'] += 1

            # Calculate time step
            if last_timestamp is not None:
                dt = timestamp - last_timestamp
            else:
                dt = 0.0

            # Extract IMU data
            accel = np.array([
                row['ax(g)'] * 9.81,
                row['ay(g)'] * 9.81,
                row['az(g)'] * 9.81
            ]).reshape(6, 1)  # Reference expects (6,1) for velocity vector

            gyro = np.array([
                row['gx(rad/s)'],
                row['gy(rad/s)'],
                row['gz(rad/s)']
            ]).reshape(6, 1)

            # Run IMU prediction if we have a valid time step
            if dt > 0:
                vel = np.vstack((gyro, accel))  # [omega; a] = (6,1) format
                filter_ref.propagate(
                    timestamp, vel,
                    gyro_noise, accel_noise, bias_noise, virtual_noise
                )
                stats['imu_predicts'] += 1

            last_timestamp = timestamp

            # Get state estimate
            R, p, v, bw, ba, bmu, _ = filter_ref.getEstimate()

            csv_writer.writerow({
                't': timestamp,
                'px': p[0, 0], 'py': p[1, 0], 'pz': p[2, 0],
                'vx': v[0, 0], 'vy': v[1, 0], 'vz': v[2, 0],
                'r00': R[0, 0], 'r01': R[0, 1], 'r02': R[0, 2],
                'r10': R[1, 0], 'r11': R[1, 1], 'r12': R[1, 2],
                'r20': R[2, 0], 'r21': R[2, 1], 'r22': R[2, 2],
                'b_gx': bw[0, 0], 'b_gy': bw[1, 0], 'b_gz': bw[2, 0],
                'b_ax': ba[0, 0], 'b_ay': ba[1, 0], 'b_az': ba[2, 0],
                'b_mu_x': bmu[0, 0], 'b_mu_y': bmu[1, 0], 'b_mu_z': bmu[2, 0],
            })

    except StopIteration:
        pass
    finally:
        csv_file.close()

    print("-" * 70)
    print("\nProcessing Complete!")
    print(f"\nStatistics:")
    print(f"  Total data rows processed: {stats['total_rows']}")
    print(f"  IMU samples: {stats['imu_predicts']}")
    print(f"\nOutput written to: {output_csv}")
    print("=" * 70)


if __name__ == "__main__":
    main()
