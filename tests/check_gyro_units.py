"""Diagnose gyro data units - check if rad/s values are actually degrees or raw ADC."""

import numpy as np
import pandas as pd

# Load flight data
df = pd.read_csv('data/20241011_NIMBUS24_Flight_FC_Data.csv')

t = df['time(s)'].values
gx = df['gx(rad/s)'].values
gy = df['gy(rad/s)'].values
gz = df['gz(rad/s)'].values

print("=" * 70)
print("GYRO DATA UNIT ANALYSIS")
print("=" * 70)

# Find the extreme values
gx_max_idx = np.argmax(np.abs(gx))
gy_max_idx = np.argmax(np.abs(gy))
gz_max_idx = np.argmax(np.abs(gz))

print(f"\nAssuming CSV is in rad/s (current interpretation):")
print(f"  Gx max: {gx[gx_max_idx]:.2f} rad/s at t={t[gx_max_idx]:.2f}s")
print(f"  Gy max: {gy[gy_max_idx]:.2f} rad/s at t={t[gy_max_idx]:.2f}s")
print(f"  Gz max: {gz[gz_max_idx]:.2f} rad/s at t={t[gz_max_idx]:.2f}s")

print(f"\nIf these are ACTUALLY in degrees (deg/s):")
print(f"  Gx max: {np.degrees(gx[gx_max_idx]):.2f} deg/s")
print(f"  Gy max: {np.degrees(gy[gy_max_idx]):.2f} deg/s")
print(f"  Gz max: {np.degrees(gz[gz_max_idx]):.2f} deg/s")

print(f"\nIf these are raw ADC counts (need conversion by 2000/32768):")
gx_adc_converted = gx * (2000.0 / 32768.0)
gy_adc_converted = gy * (2000.0 / 32768.0)
gz_adc_converted = gz * (2000.0 / 32768.0)
print(f"  Gx max: {gx_adc_converted[gx_max_idx]:.2f} deg/s (from ADC)")
print(f"  Gy max: {gy_adc_converted[gy_max_idx]:.2f} deg/s (from ADC)")
print(f"  Gz max: {gz_adc_converted[gz_max_idx]:.2f} deg/s (from ADC)")

# Check what makes sense physically
print("\n" + "=" * 70)
print("PHYSICAL FEASIBILITY CHECK")
print("=" * 70)

# Sensor specs: ±2000 °/s max
sensor_max_degs = 2000.0
sensor_max_rads = np.radians(sensor_max_degs)

print(f"\nICM_20608 sensor max: ±2000 °/s = ±{sensor_max_rads:.2f} rad/s")

# Check current CSV as rad/s
current_max_rads = np.max([np.abs(gx[gx_max_idx]), np.abs(gy[gy_max_idx]), np.abs(gz[gz_max_idx])])
print(f"Current max in CSV (as rad/s): {current_max_rads:.2f} rad/s")
print(f"  >{np.degrees(current_max_rads):.2f} °/s")
print(f"  >{np.degrees(current_max_rads) / sensor_max_degs:.1f}x sensor max (IMPOSSIBLE)")

# Check if ADC conversion gives reasonable value
current_max_adc_degs = np.degrees(gx_adc_converted[gx_max_idx])
print(f"\nIf converted from raw ADC (2000/32768 factor):")
print(f"  Max: {current_max_adc_degs:.2f} °/s")
print(f"  >{current_max_adc_degs / sensor_max_degs:.2f}x sensor max (REASONABLE)")

# Show a sample around the peak
peak_idx = gx_max_idx
print(f"\n" + "=" * 70)
print("SAMPLE AROUND PEAK (t = {:.2f}s):".format(t[peak_idx]))
print("=" * 70)
print(f"{'Time (s)':<10} {'Gx (CSV)':<15} {'Gx (°/s if ADC)':<20} {'Gy (CSV)':<15} {'Gz (CSV)':<15}")
print("-" * 75)
for i in range(max(0, peak_idx - 2), min(len(t), peak_idx + 3)):
    gx_as_adc = gx[i] * (2000.0 / 32768.0)
    print(f"{t[i]:<10.3f} {gx[i]:<15.2f} {np.degrees(gx_as_adc):<20.2f} {gy[i]:<15.2f} {gz[i]:<15.2f}")

# Statistical check
print(f"\n" + "=" * 70)
print("STATISTICAL CHECK")
print("=" * 70)
print(f"Mean gyro magnitude (current): {np.sqrt(np.mean(gx**2 + gy**2 + gz**2)):.2f}")
print(f"95th percentile of max(|gx|,|gy|,|gz|): {np.percentile(np.max([np.abs(gx), np.abs(gy), np.abs(gz)], axis=0), 95):.2f}")

# If ADC
gx_gy_gz_adc = np.array([gx_adc_converted, gy_adc_converted, gz_adc_converted])
print(f"Mean gyro magnitude (if ADC): {np.sqrt(np.mean(gx_gy_gz_adc**2)):.2f}")
print(f"95th percentile of max(|gx|,|gy|,|gz|) (if ADC): {np.percentile(np.max(np.abs(gx_gy_gz_adc), axis=0), 95):.2f}")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
if current_max_rads > sensor_max_rads:
    print("[NO] Current values EXCEED sensor max - NOT valid rad/s")
    print("[YES] ADC conversion gives reasonable values - Data is likely raw ADC counts")
else:
    print("[YES] Current values within sensor max - Could be rad/s")
    print("[NO] ADC conversion would be too small - Data is likely already in rad/s")
