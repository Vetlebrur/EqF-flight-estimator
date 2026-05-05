"""Compare attitude estimates with raw ADC vs converted gyro data."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load raw flight data
df = pd.read_csv('data/20241011_NIMBUS24_Flight_FC_Data.csv')
t = df['time(s)'].values
gx_raw = df['gx(rad/s)'].values
gy_raw = df['gy(rad/s)'].values
gz_raw = df['gz(rad/s)'].values

# Apply conversion
gyro_scale = (2000.0 / 32768.0) * (np.pi / 180.0)  # raw ADC -> rad/s
gx_converted = gx_raw * gyro_scale
gy_converted = gy_raw * gyro_scale
gz_converted = gz_raw * gyro_scale

# Load filter output
output = pd.read_csv('outputs/tg_eqf_output.csv')
t_out = output['t'].values
roll_out = output['roll'].values
pitch_out = output['pitch'].values
yaw_out = output['yaw'].values

# Create comparison plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Row 1: Gyro data comparison
ax = axes[0, 0]
ax.plot(t, gx_raw, label='Raw ADC (mislabeled as rad/s)', alpha=0.5, linewidth=0.5)
ax.plot(t, gx_converted, label='Converted ADC to rad/s', linewidth=0.8, color='darkred')
ax.set_xlabel('Time [s]')
ax.set_ylabel('Gx [rad/s]')
ax.set_title('Gyro X: Raw vs Converted')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.plot(t, np.abs(gx_raw), label='|Raw ADC|', alpha=0.5, linewidth=0.5)
ax.plot(t, np.abs(gx_converted), label='|Converted|', linewidth=0.8, color='darkred')
ax.set_xlabel('Time [s]')
ax.set_ylabel('|Gyro X| [rad/s]')
ax.set_title('Gyro X Magnitude: Before vs After Conversion')
ax.set_yscale('log')
ax.legend()
ax.grid(True, alpha=0.3)

# Row 2: Attitude estimate
ax = axes[1, 0]
ax.plot(t_out, np.degrees(roll_out), label='Roll (estimated)', linewidth=0.8, color='blue')
ax.plot(t_out, np.degrees(pitch_out), label='Pitch (estimated)', linewidth=0.8, color='green')
ax.plot(t_out, np.degrees(yaw_out), label='Yaw (estimated)', linewidth=0.8, color='red')
ax.set_xlabel('Time [s]')
ax.set_ylabel('Angle [degrees]')
ax.set_title('Attitude Estimates with Corrected Gyro Data')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim([-200, 200])

# Peak analysis
ax = axes[1, 1]
peak_idx = np.argmax(np.abs(gx_raw))
peak_t = t[peak_idx]
window = (t >= peak_t - 2) & (t <= peak_t + 2)
window_out = (t_out >= peak_t - 2) & (t_out <= peak_t + 2)

ax.plot(t[window], np.degrees(gx_converted[window]), label='Gx (converted)', linewidth=1, color='red')
ax.plot(t_out[window_out], np.degrees(roll_out[window_out]), label='Roll estimate', linewidth=1, color='blue')
ax.set_xlabel('Time [s]')
ax.set_ylabel('Value [deg or deg/s]')
ax.set_title(f'Detail at peak time t={peak_t:.2f}s')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/gyro_conversion_comparison.png', dpi=150, bbox_inches='tight')
print("Saved plot to outputs/gyro_conversion_comparison.png")

# Print statistics
print("\n" + "=" * 70)
print("GYRO CONVERSION STATISTICS")
print("=" * 70)
print(f"\nRaw ADC Data (mislabeled as rad/s):")
print(f"  Gx: mean={gx_raw.mean():.2f}, std={gx_raw.std():.2f}, max={np.abs(gx_raw).max():.2f}")
print(f"  Gy: mean={gy_raw.mean():.2f}, std={gy_raw.std():.2f}, max={np.abs(gy_raw).max():.2f}")
print(f"  Gz: mean={gz_raw.mean():.2f}, std={gz_raw.std():.2f}, max={np.abs(gz_raw).max():.2f}")

print(f"\nConverted to rad/s (2000/32768 factor):")
print(f"  Gx: mean={gx_converted.mean():.6f}, std={gx_converted.std():.6f}, max={np.abs(gx_converted).max():.6f} rad/s")
print(f"  Gy: mean={gy_converted.mean():.6f}, std={gy_converted.std():.6f}, max={np.abs(gy_converted).max():.6f} rad/s")
print(f"  Gz: mean={gz_converted.mean():.6f}, std={gz_converted.std():.6f}, max={np.abs(gz_converted).max():.6f} rad/s")

print(f"\nIn degrees/sec (multiply by 180/pi):")
gx_deg = gx_converted * 180.0 / np.pi
gy_deg = gy_converted * 180.0 / np.pi
gz_deg = gz_converted * 180.0 / np.pi
print(f"  Gx: max={np.abs(gx_deg).max():.2f} deg/s")
print(f"  Gy: max={np.abs(gy_deg).max():.2f} deg/s")
print(f"  Gz: max={np.abs(gz_deg).max():.2f} deg/s")
print(f"  Sensor limit: ±2000 deg/s")
print(f"  Gx exceeds limit by: {np.abs(gx_deg).max() / 2000.0:.2f}x")

print(f"\nFilter Output:")
print(f"  Roll: mean={np.degrees(roll_out).mean():.2f}°, std={np.degrees(roll_out).std():.2f}°, max={np.abs(np.degrees(roll_out)).max():.2f}°")
print(f"  Pitch: mean={np.degrees(pitch_out).mean():.2f}°, std={np.degrees(pitch_out).std():.2f}°, max={np.abs(np.degrees(pitch_out)).max():.2f}°")
print(f"  Yaw: mean={np.degrees(yaw_out).mean():.2f}°, std={np.degrees(yaw_out).std():.2f}°, max={np.abs(np.degrees(yaw_out)).max():.2f}°")

print(f"\nPeak event analysis (t ~ {peak_t:.2f}s):")
peak_idx_out = np.argmin(np.abs(t_out - peak_t))
print(f"  Raw ADC Gx: {gx_raw[peak_idx]:.2f} (raw counts)")
print(f"  Converted Gx: {gx_converted[peak_idx]:.6f} rad/s = {gx_deg[peak_idx]:.2f} deg/s")
print(f"  Estimated roll: {np.degrees(roll_out[peak_idx_out]):.2f}°")
print(f"  Estimated pitch: {np.degrees(pitch_out[peak_idx_out]):.2f}°")
