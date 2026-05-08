"""Compare GNSS, FC, and accel-integrated velocity (NED) from raw flight data."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

# =============================================================================
# Config
# =============================================================================

DATASET = "30s"

_FILES = {
    "full":    "data/20241011_NIMBUS24_Flight_FC_Data.csv",
    "30s":     "data/20241011_NIMBUS24_Flight_FC_Data_30s.csv",
    "1s_loop": "data/20241011_NIMBUS24_Flight_FC_Data_1s_loop.csv",
}

g = 9.81

# Column indices (0-based, confirmed from CSV header)
_C = {
    "t":      0,
    "lat":    2,
    "lon":    1,
    "gps_vn": 4,   # GNSS velocity North (mm/s)
    "gps_ve": 5,   # GNSS velocity East  (mm/s)
    "gps_vd": 6,   # GNSS velocity Down  (mm/s)
    "fc_vn":  39,  # FC onboard velocity North (m/s)
    "fc_ve":  40,  # FC onboard velocity East  (m/s)
    "fc_vd":  41,  # FC onboard velocity Down  (m/s)
    "ax":     9,   # IMU accel X (g)
    "ay":     10,  # IMU accel Y (g)  — mounted inverted, sign flipped below
    "az":     11,  # IMU accel Z (g)
}

# =============================================================================
# Load and preprocess
# =============================================================================

raw = np.genfromtxt(_FILES[DATASET], delimiter=",", skip_header=1)
if raw.ndim == 1:
    raw = raw.reshape(1, -1)

t_all = raw[:, _C["t"]]
finite_t = np.isfinite(t_all)
raw   = raw[finite_t]
t_all = t_all[finite_t]

# --- GNSS ---
gps_valid = (
    (raw[:, _C["lat"]] != 0) &
    np.isfinite(raw[:, _C["gps_vn"]]) &
    np.isfinite(raw[:, _C["gps_ve"]]) &
    np.isfinite(raw[:, _C["gps_vd"]])
)
t_gnss  = t_all[gps_valid]
vn_gnss = raw[gps_valid, _C["gps_vn"]] / 1000.0
ve_gnss = raw[gps_valid, _C["gps_ve"]] / 1000.0
vd_gnss = raw[gps_valid, _C["gps_vd"]] / 1000.0

# --- FC onboard velocity ---
fc_valid = (
    np.isfinite(raw[:, _C["fc_vn"]]) &
    np.isfinite(raw[:, _C["fc_ve"]]) &
    np.isfinite(raw[:, _C["fc_vd"]])
)
t_fc  = t_all[fc_valid]
vn_fc = raw[fc_valid, _C["fc_vn"]]
ve_fc = raw[fc_valid, _C["fc_ve"]]
vd_fc = raw[fc_valid, _C["fc_vd"]]

# --- Accelerometer: same sign convention as the filters (ay negated) ---
accel_valid = (
    np.isfinite(raw[:, _C["ax"]]) &
    np.isfinite(raw[:, _C["ay"]]) &
    np.isfinite(raw[:, _C["az"]])
)
t_acc = t_all[accel_valid]
ax_ms2 = raw[accel_valid, _C["ax"]] * g                # body X  [m/s²]
ay_ms2 = raw[accel_valid, _C["ay"]] * g * -1.0         # body Y  [m/s²], inverted mount
az_ms2 = raw[accel_valid, _C["az"]] * g                # body Z  [m/s²]

# Estimate pre-launch window: first 2 s where |a| is close to g (rocket stationary)
accel_norm = np.sqrt(ax_ms2**2 + ay_ms2**2 + az_ms2**2)
pre_launch = (t_acc < 2.0) & (np.abs(accel_norm - g) < 3.0)
if pre_launch.any():
    ax_bias = np.mean(ax_ms2[pre_launch])
    ay_bias = np.mean(ay_ms2[pre_launch])
    az_bias = np.mean(az_ms2[pre_launch])
else:
    ax_bias = ay_bias = az_bias = 0.0

print(f"Pre-launch accel bias (m/s²):  ax={ax_bias:.3f}  ay={ay_bias:.3f}  az={az_bias:.3f}")

# Subtract static bias so that gravity + sensor offset are removed
ax_dyn = ax_ms2 - ax_bias
ay_dyn = ay_ms2 - ay_bias
az_dyn = az_ms2 - az_bias

# Integrate: velocity relative to start (cumulative trapezoid)
vax = cumulative_trapezoid(ax_dyn, t_acc, initial=0.0)
vay = cumulative_trapezoid(ay_dyn, t_acc, initial=0.0)
vaz = cumulative_trapezoid(az_dyn, t_acc, initial=0.0)

# Shift each integrated velocity so it starts at the first GNSS velocity sample
def shift_to_gnss_start(v_int: np.ndarray, t_int: np.ndarray,
                         v_gnss: np.ndarray, t_gnss: np.ndarray) -> np.ndarray:
    """Shift integral so v_int[0] matches the GNSS velocity at the same time."""
    v_gnss_at_t0 = float(np.interp(t_int[0], t_gnss, v_gnss))
    return v_int - v_int[0] + v_gnss_at_t0

vax_n = shift_to_gnss_start(vax, t_acc, vn_gnss, t_gnss)
vax_e = shift_to_gnss_start(vax, t_acc, ve_gnss, t_gnss)
vax_d = shift_to_gnss_start(vax, t_acc, vd_gnss, t_gnss)

vay_n = shift_to_gnss_start(vay, t_acc, vn_gnss, t_gnss)
vay_e = shift_to_gnss_start(vay, t_acc, ve_gnss, t_gnss)
vay_d = shift_to_gnss_start(vay, t_acc, vd_gnss, t_gnss)

vaz_n = shift_to_gnss_start(vaz, t_acc, vn_gnss, t_gnss)
vaz_e = shift_to_gnss_start(vaz, t_acc, ve_gnss, t_gnss)
vaz_d = shift_to_gnss_start(vaz, t_acc, vd_gnss, t_gnss)

speed_gnss = np.sqrt(vn_gnss**2 + ve_gnss**2 + vd_gnss**2)
speed_fc   = np.sqrt(vn_fc**2   + ve_fc**2   + vd_fc**2)
speed_acc  = np.sqrt(vax**2 + vay**2 + vaz**2)

print(f"GNSS max speed: {np.max(speed_gnss):.1f} m/s")
print(f"FC   max speed: {np.max(speed_fc):.1f} m/s")
print(f"Accel int max speed (magnitude): {np.max(speed_acc):.1f} m/s")

# =============================================================================
# Plot
# =============================================================================

fig, axes = plt.subplots(4, 1, figsize=(14, 13), sharex=True)
fig.suptitle(
    f"Velocity comparison: GNSS vs FC vs ∫accel — {DATASET}\n"
    "Body→NED mapping: az→N, ay→E, −ax→D  (bias removed, shifted to GNSS t₀)",
    fontsize=12, fontweight='bold'
)

accel_style = dict(linewidth=1.0, alpha=0.7)

titles = [
    "Velocity North (vN)",
    "Velocity East  (vE)",
    "Velocity Down  (vD)",
    "Total Speed",
]
ylabels = ["North [m/s]", "East [m/s]", "Down [m/s]", "Speed [m/s]"]
gnss_v  = [vn_gnss, ve_gnss, vd_gnss, speed_gnss]
fc_v    = [vn_fc,   ve_fc,   vd_fc,   speed_fc]

# User-confirmed body→NED mapping:  az→N,  ay→E,  -ax→D
vN_acc = shift_to_gnss_start(vaz, t_acc, vn_gnss, t_gnss)   # ∫az  → North
vE_acc = shift_to_gnss_start(vay, t_acc, ve_gnss, t_gnss)   # ∫ay  → East
vD_acc = shift_to_gnss_start(-vax, t_acc, vd_gnss, t_gnss)  # ∫-ax → Down
speed_acc_ned = np.sqrt(vN_acc**2 + vE_acc**2 + vD_acc**2)

acc_ned = [vN_acc, vE_acc, vD_acc, speed_acc_ned]
acc_labels = ['∫az → North', '∫ay → East', '∫(−ax) → Down', '∫a speed']

for i, (ax, ylabel, title, gv, fv, av, al) in enumerate(
        zip(axes, ylabels, titles, gnss_v, fc_v, acc_ned, acc_labels)):
    ax.plot(t_gnss, gv, 'g-',  linewidth=2.0, label='GNSS',       alpha=0.9, zorder=3)
    ax.plot(t_fc,   fv, 'b--', linewidth=1.5, label='FC onboard', alpha=0.9, zorder=3)
    ax.plot(t_acc, av, color='#e67e22', linestyle='-.', label=al, **accel_style)

    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, loc='upper left', ncol=2)
    ax.grid(True, alpha=0.4)

axes[-1].set_xlabel("Time [s]", fontsize=10)
plt.tight_layout()

out = f"outputs/velocity_comparison_{DATASET}.png"
plt.savefig(out, dpi=150)
print(f"\nSaved to {out}")
plt.show()
