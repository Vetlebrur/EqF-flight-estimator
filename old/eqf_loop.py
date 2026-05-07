import numpy as np
import csv
from pathlib import Path
from old.tg_eqf import TGEqF
from old.update_processor import UpdateProcessor
import old.eqf_predict as eqf_predict  # Attaches predict methods to TGEqF
import old.eqf_update as eqf_update  # Attaches update methods to TGEqF
import old.eqf_reset as eqf_reset  # Attaches reset methods to TGEqF


def main():
    """
    Main filter loop: Initialize filter and process flight data.
    """
    print("=" * 70)
    print("Tangent-Group Equivariant Filter (TGEqF) - Flight Data Processing")
    print("=" * 70)

    # Initialize data processor
    csv_path = r"data\20241011_NIMBUS24_Flight_FC_Data.csv"
    processor = UpdateProcessor(csv_path)
    print(f"\nLoading flight data from: {csv_path}")

    # Skip first row to sync with processor
    _ = processor.get_next_row()

    # Initialize filter with identity (rocket pointing up)
    # FC's initial state is applied via group action in renderer
    filter = TGEqF()
    print(f"\nFilter initialized at t=0 with identity state")
    print(f"  Rotation: identity (rocket pointing up)")
    print(f"  Position: origin, Velocity: zero")
    print(f"  Covariance: P = {np.linalg.norm(filter.P):.4f} (Frobenius norm)")

    # Initialize CSV writer for filter estimates
    output_csv = Path("eqf_estimates.csv")
    csv_file = output_csv.open('w', newline='')
    fieldnames = ['t', 'px', 'py', 'pz', 'vx', 'vy', 'vz',
                  'r00', 'r01', 'r02', 'r10', 'r11', 'r12', 'r20', 'r21', 'r22',
                  'b_gx', 'b_gy', 'b_gz', 'b_ax', 'b_ay', 'b_az', 'b_mu_x', 'b_mu_y', 'b_mu_z']
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    csv_writer.writeheader()

    # Statistics
    stats = {
        'total_rows': 0,
        'gnss_updates': 0,
        'mag_updates': 0,
        'baro_updates': 0,
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

            # ===== MEASUREMENT UPDATES =====

            # GNSS Update
            gnss_data = processor.check_gnss_update(row)
            if gnss_data is not None:
                # Disabled - implementation needs InnovationLift proper mapping
                # pos, vel = gnss_data
                # filter.gnss_update(pos, vel)
                pass
                # stats['gnss_updates'] += 1

            # Magnetometer Update
            mag_data = processor.check_magnetometer_update(row)
            if mag_data is not None:
                # Disabled - test propagation-only stability
                # filter.magnetometer_update(mag_data)
                pass
                # stats['mag_updates'] += 1

            # Barometer Update
            baro_data = processor.check_barometer_update(row)
            if baro_data is not None:
                # Disabled for now - barometer not properly integrated
                # filter.barometer_update(baro_data)
                pass
                # stats['baro_updates'] += 1

            # ===== PREDICTION =====
            # Calculate time step
            if last_timestamp is not None:
                dt = timestamp - last_timestamp
            else:
                dt = 0.0  # First iteration, no prediction

            # Extract IMU data and predict state
            # Use h_ax/y/z (high-pass filtered accel with gravity removed)
            accel = np.array([
                row['h_ax(g)'] * 9.81,
                row['h_ay(g)'] * 9.81,
                row['h_az(g)'] * 9.81
            ])
            gyro = np.array([
                row['gx(rad/s)'],
                row['gy(rad/s)'],
                row['gz(rad/s)']
            ])

            # Run IMU prediction if we have a valid time step
            if dt > 0:
                # Structural input mu: force to b_mu when not measuring separately
                mu = filter.b[6:9]
                filter.imu_predict(dt, accel, gyro, mu=mu)
                stats['imu_predicts'] += 1

            filter.time = timestamp
            last_timestamp = timestamp

            # Save filter estimate to CSV
            R = filter.T.R().as_matrix()
            vel = filter.T.x().as_vector()  # x() returns velocity (column 3)
            pos = filter.T.w().as_vector()  # w() returns position (column 4)
            csv_writer.writerow({
                't': timestamp,
                'px': pos[0], 'py': pos[1], 'pz': pos[2],
                'vx': vel[0], 'vy': vel[1], 'vz': vel[2],
                'r00': R[0, 0], 'r01': R[0, 1], 'r02': R[0, 2],
                'r10': R[1, 0], 'r11': R[1, 1], 'r12': R[1, 2],
                'r20': R[2, 0], 'r21': R[2, 1], 'r22': R[2, 2],
                'b_gx': filter.b[0], 'b_gy': filter.b[1], 'b_gz': filter.b[2],
                'b_ax': filter.b[3], 'b_ay': filter.b[4], 'b_az': filter.b[5],
                'b_mu_x': filter.b[6], 'b_mu_y': filter.b[7], 'b_mu_z': filter.b[8],
            })

    except StopIteration:
        pass
    finally:
        csv_file.close()

    print("-" * 70)
    print("\nProcessing Complete!")
    print(f"\nStatistics:")
    print(f"  Total data rows processed: {stats['total_rows']}")
    print(f"  GNSS updates: {stats['gnss_updates']}")
    print(f"  Magnetometer updates: {stats['mag_updates']}")
    print(f"  Barometer updates: {stats['baro_updates']}")
    print(f"  IMU samples: {stats['imu_predicts']}")
    print(f"\nFinal filter state at t={filter.time:.3f}s:")
    print(f"  Covariance norm: {np.linalg.norm(filter.P):.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
