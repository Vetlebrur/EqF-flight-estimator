"""Plot raw and smoothed gyro/accel data from NIMBUS24 flight."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

# Load flight data
df = pd.read_csv('data/20241011_NIMBUS24_Flight_FC_Data.csv')

# Extract sensors
t = df['time(s)'].values
gx = df['gx(rad/s)'].values
gy = df['gy(rad/s)'].values
gz = df['gz(rad/s)'].values
ax = df['ax(g)'].values * 9.81  # Convert to m/s²
ay = df['ay(g)'].values * 9.81
az = df['az(g)'].values * 9.81

# Apply moving average (window=10)
window = 10
gx_ma = uniform_filter1d(gx, size=window, mode='nearest')
gy_ma = uniform_filter1d(gy, size=window, mode='nearest')
gz_ma = uniform_filter1d(gz, size=window, mode='nearest')
ax_ma = uniform_filter1d(ax, size=window, mode='nearest')
ay_ma = uniform_filter1d(ay, size=window, mode='nearest')
az_ma = uniform_filter1d(az, size=window, mode='nearest')

# Create figure with 2 subplots
fig, (ax_gyro, ax_accel) = plt.subplots(2, 1, figsize=(14, 10))

# =========================================================================
# Plot Gyroscope Data
# =========================================================================
ax_gyro.plot(t, gx, label='Gx (raw)', linewidth=0.8, alpha=0.6, color='red')
ax_gyro.plot(t, gx_ma, label='Gx (MA, window=10)', linewidth=1.5, color='darkred')

ax_gyro.plot(t, gy, label='Gy (raw)', linewidth=0.8, alpha=0.6, color='green')
ax_gyro.plot(t, gy_ma, label='Gy (MA, window=10)', linewidth=1.5, color='darkgreen')

ax_gyro.plot(t, gz, label='Gz (raw)', linewidth=0.8, alpha=0.6, color='blue')
ax_gyro.plot(t, gz_ma, label='Gz (MA, window=10)', linewidth=1.5, color='darkblue')

ax_gyro.set_xlabel('Time [s]', fontsize=11)
ax_gyro.set_ylabel('Angular Velocity [rad/s]', fontsize=11)
ax_gyro.set_title('Gyroscope: Raw vs Moving Average (window=10)', fontsize=12, fontweight='bold')
ax_gyro.legend(loc='upper right', fontsize=9)
ax_gyro.grid(True, alpha=0.3)

# =========================================================================
# Plot Accelerometer Data
# =========================================================================
ax_accel.plot(t, ax, label='Ax (raw)', linewidth=0.8, alpha=0.6, color='red')
ax_accel.plot(t, ax_ma, label='Ax (MA, window=10)', linewidth=1.5, color='darkred')

ax_accel.plot(t, ay, label='Ay (raw)', linewidth=0.8, alpha=0.6, color='green')
ax_accel.plot(t, ay_ma, label='Ay (MA, window=10)', linewidth=1.5, color='darkgreen')

ax_accel.plot(t, az, label='Az (raw)', linewidth=0.8, alpha=0.6, color='blue')
ax_accel.plot(t, az_ma, label='Az (MA, window=10)', linewidth=1.5, color='darkblue')

ax_accel.set_xlabel('Time [s]', fontsize=11)
ax_accel.set_ylabel('Acceleration [m/s²]', fontsize=11)
ax_accel.set_title('Accelerometer: Raw vs Moving Average (window=10)', fontsize=12, fontweight='bold')
ax_accel.legend(loc='upper right', fontsize=9)
ax_accel.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/sensor_smoothing_comparison.png', dpi=150, bbox_inches='tight')
print("Saved plot to outputs/sensor_smoothing_comparison.png")
plt.close()

# Print statistics
print("\n" + "="*70)
print("SENSOR STATISTICS")
print("="*70)
print("\nGyroscope (rad/s):")
print(f"  Gx: mean={gx.mean():7.4f}, std={gx.std():7.4f}, max={gx.max():7.4f}")
print(f"  Gy: mean={gy.mean():7.4f}, std={gy.std():7.4f}, max={gy.max():7.4f}")
print(f"  Gz: mean={gz.mean():7.4f}, std={gz.std():7.4f}, max={gz.max():7.4f}")

print("\nGyroscope MA (rad/s):")
print(f"  Gx: mean={gx_ma.mean():7.4f}, std={gx_ma.std():7.4f}, max={gx_ma.max():7.4f}")
print(f"  Gy: mean={gy_ma.mean():7.4f}, std={gy_ma.std():7.4f}, max={gy_ma.max():7.4f}")
print(f"  Gz: mean={gz_ma.mean():7.4f}, std={gz_ma.std():7.4f}, max={gz_ma.max():7.4f}")

print("\nAccelerometer (m/s²):")
print(f"  Ax: mean={ax.mean():7.2f}, std={ax.std():7.2f}, max={ax.max():7.2f}")
print(f"  Ay: mean={ay.mean():7.2f}, std={ay.std():7.2f}, max={ay.max():7.2f}")
print(f"  Az: mean={az.mean():7.2f}, std={az.std():7.2f}, max={az.max():7.2f}")

print("\nAccelerometer MA (m/s²):")
print(f"  Ax: mean={ax_ma.mean():7.2f}, std={ax_ma.std():7.2f}, max={ax_ma.max():7.2f}")
print(f"  Ay: mean={ay_ma.mean():7.2f}, std={ay_ma.std():7.2f}, max={ay_ma.max():7.2f}")
print(f"  Az: mean={az_ma.mean():7.2f}, std={az_ma.std():7.2f}, max={az_ma.max():7.2f}")

# Show noise reduction
print("\n" + "="*70)
print("NOISE REDUCTION (MA vs Raw)")
print("="*70)
print(f"Gx std reduction: {(1 - gx_ma.std()/gx.std())*100:.1f}%")
print(f"Gy std reduction: {(1 - gy_ma.std()/gy.std())*100:.1f}%")
print(f"Gz std reduction: {(1 - gz_ma.std()/gz.std())*100:.1f}%")
print(f"Ax std reduction: {(1 - ax_ma.std()/ax.std())*100:.1f}%")
print(f"Ay std reduction: {(1 - ay_ma.std()/ay.std())*100:.1f}%")
print(f"Az std reduction: {(1 - az_ma.std()/az.std())*100:.1f}%")
