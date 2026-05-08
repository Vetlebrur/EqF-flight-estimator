"""Gyro + magnetometer attitude estimation — automatic axis-mapping search.

Searches all 48 combinations of magnetometer axis permutations and sign flips,
picks the one with the lowest angular RMSE vs FC, and plots the result alongside
pure gyro integration, FC attitude, and EqF filter output.
No hard-iron bias correction is applied (raw magnetometer only).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations, product as iproduct
from scipy.spatial.transform import Rotation

# =============================================================================
# Config
# =============================================================================

DATASET = "30s"

_DATASETS = {
    "full":    ("data/20241011_NIMBUS24_Flight_FC_Data.csv",         "outputs/tg_eqf_output_full.csv"),
    "30s":     ("data/20241011_NIMBUS24_Flight_FC_Data_30s.csv",     "outputs/tg_eqf_output_30s.csv"),
    "1s_loop": ("data/20241011_NIMBUS24_Flight_FC_Data_1s_loop.csv", "outputs/tg_eqf_output_1s_loop.csv"),
}

csv_in, eqf_out = _DATASETS[DATASET]

g          = 9.81
gyro_scale = np.pi / 180.0  # deg/s → rad/s (CSV already in deg/s)

# WMM2025 — NIMBUS24 launch site (Ribeira Grande, Azores)
WMM_DECLINATION = -13.4
WMM_INCLINATION = 56.8
WMM_MAGNITUDE   = 47000.0
_dec = np.radians(WMM_DECLINATION)
_inc = np.radians(WMM_INCLINATION)
_h   = WMM_MAGNITUDE * np.cos(_inc)
_wmm_ned = np.array([_h * np.cos(_dec), _h * np.sin(_dec), WMM_MAGNITUDE * np.sin(_inc)])
WMM_NED_UNIT = _wmm_ned / np.linalg.norm(_wmm_ned)

MAG_YAW_TAU    = 2.0   # soft correction time constant [s]
STATIC_TOL     = 3.0   # m/s² — max deviation from |a|=g to count as static
MIN_INIT_SAMPLES = 5

_C = {
    "t":  0, "lat": 2,
    "ax": 9, "ay": 10, "az": 11,
    "gx": 15, "gy": 16, "gz": 17,
    "mx": 18, "my": 19, "mz": 20,
    "roll": 29, "pitch": 30, "yaw": 31,
}
_E = {"t": 0, "q": slice(16, 20), "mag_q": slice(29, 33)}

# =============================================================================
# Helpers
# =============================================================================

def triad_rotation(b1: np.ndarray, b2: np.ndarray,
                   r1: np.ndarray, r2: np.ndarray) -> np.ndarray | None:
    t1 = r1 / np.linalg.norm(r1)
    t12 = np.cross(r1, r2); n12 = np.linalg.norm(t12)
    if n12 < 1e-6: return None
    t2 = t12 / n12; t3 = np.cross(t1, t2)
    s1 = b1 / np.linalg.norm(b1)
    s12 = np.cross(b1, b2); sn12 = np.linalg.norm(s12)
    if sn12 < 1e-6: return None
    s2 = s12 / sn12; s3 = np.cross(s1, s2)
    return np.column_stack([t1, t2, t3]) @ np.column_stack([s1, s2, s3]).T



def to_euler_deg(q_wxyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """q_wxyz [N,4] → (yaw, pitch, roll) in degrees, unwrapped."""
    rot = Rotation.from_quat(q_wxyz[:, [1, 2, 3, 0]])
    e = rot.as_euler('ZYX', degrees=True)
    return (np.unwrap(e[:, 0], discont=180, period=360),
            np.unwrap(e[:, 1], discont=180, period=360),
            np.unwrap(e[:, 2], discont=180, period=360))


def align_start(t_est: np.ndarray, angle_est: np.ndarray,
                t_ref: np.ndarray, angle_ref: np.ndarray) -> np.ndarray:
    """Shift angle_est so it starts at the same value as angle_ref at t_est[0]."""
    ref_at_t0 = float(np.interp(t_est[0], t_ref, angle_ref))
    return angle_est - angle_est[0] + ref_at_t0


def angular_rmse(t_est: np.ndarray, q_est: np.ndarray,
                 t_ref: np.ndarray, q_ref: np.ndarray) -> float:
    q_i = np.column_stack([np.interp(t_est, t_ref, q_ref[:, i]) for i in range(4)])
    dot = np.clip(np.abs(np.einsum('ij,ij->i', q_est, q_i)), 0.0, 1.0)
    return float(np.sqrt(np.mean(np.degrees(2 * np.arccos(dot)) ** 2)))


def angular_err_series(t_est: np.ndarray, q_est: np.ndarray,
                       t_ref: np.ndarray, q_ref: np.ndarray) -> np.ndarray:
    q_i = np.column_stack([np.interp(t_est, t_ref, q_ref[:, i]) for i in range(4)])
    dot = np.clip(np.abs(np.einsum('ij,ij->i', q_est, q_i)), 0.0, 1.0)
    return np.degrees(2 * np.arccos(dot))


# =============================================================================
# Load raw data
# =============================================================================

raw = np.genfromtxt(csv_in, delimiter=",", skip_header=1)
if raw.ndim == 1:
    raw = raw.reshape(1, -1)
raw   = raw[np.isfinite(raw[:, _C["t"]])]
t_all = raw[:, _C["t"]]

# FC attitude (radians in CSV)
att_ok  = (np.isfinite(raw[:, _C["roll"]]) &
           np.isfinite(raw[:, _C["pitch"]]) &
           np.isfinite(raw[:, _C["yaw"]]))
t_fc    = t_all[att_ok]
roll_fc  = np.unwrap(raw[att_ok, _C["roll"]],  discont=np.pi)
pitch_fc = np.unwrap(raw[att_ok, _C["pitch"]], discont=np.pi)
yaw_fc   = np.unwrap(raw[att_ok, _C["yaw"]],   discont=np.pi)
fc_q_wxyz = Rotation.from_euler(
    'ZYX', np.column_stack([yaw_fc, pitch_fc, roll_fc])
).as_quat()[:, [3, 0, 1, 2]]  # [w,x,y,z]

# Gyro mask (shared for display and integration)
gyro_ok = (np.isfinite(raw[:, _C["gx"]]) &
           np.isfinite(raw[:, _C["gy"]]) &
           np.isfinite(raw[:, _C["gz"]]))
t_gyro  = t_all[gyro_ok]

# Raw gyro display (CSV already in deg/s)
gyro_N = raw[gyro_ok, _C["gx"]]
gyro_E = raw[gyro_ok, _C["gy"]]
gyro_D = raw[gyro_ok, _C["gz"]]

# Integration arrays — CSV is in deg/s, convert to rad/s; pitch and yaw signs corrected
t_intg  = t_gyro.copy()
omega_x =  raw[gyro_ok, _C["gx"]] * gyro_scale
omega_y = -raw[gyro_ok, _C["gy"]] * gyro_scale  # pitch flipped
omega_z = -raw[gyro_ok, _C["gz"]] * gyro_scale  # yaw flipped
mx_intg = raw[gyro_ok, _C["mx"]]
my_intg = raw[gyro_ok, _C["my"]]
mz_intg = raw[gyro_ok, _C["mz"]]

grav_ned = np.array([0.0, 0.0, 1.0])

# =============================================================================
# TRIAD initialization for a given magnetometer axis mapping
# =============================================================================

def triad_init(mag_order: np.ndarray, mag_signs: np.ndarray) -> np.ndarray | None:
    """Compute body→NED rotation via TRIAD using pre-launch static phase."""
    acc = np.zeros(3); n = 0; init_m = None
    for _r in raw:
        if not np.isfinite(_r[_C["t"]]): continue
        _a = np.array([_r[_C["ax"]], _r[_C["ay"]], _r[_C["az"]]]) * g
        _m = np.array([_r[_C["mx"]], _r[_C["my"]], _r[_C["mz"]]])
        if not (np.all(np.isfinite(_a)) and np.all(np.isfinite(_m))): continue
        if abs(np.linalg.norm(_a) - g) > STATIC_TOL: continue
        acc += _a; n += 1
        if init_m is None:
            mb = _m[mag_order] * mag_signs
            nm = np.linalg.norm(mb)
            if nm > 1e-6:
                init_m = mb / nm
        if n >= MIN_INIT_SAMPLES and init_m is not None:
            break
    if n < MIN_INIT_SAMPLES or init_m is None:
        return None
    b1 = -(acc / n) / np.linalg.norm(acc / n)
    return triad_rotation(b1, init_m, grav_ned, WMM_NED_UNIT)


# =============================================================================
# Gyro integration with optional soft yaw-only magnetometer correction
# =============================================================================

def integrate(r0: np.ndarray, mag_order: np.ndarray, mag_signs: np.ndarray,
              apply_mag: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Integrate gyro from r0 with optional soft magnetometer correction.

    Returns (t [N], q_wxyz [N,4]).
    """
    rot = Rotation.from_matrix(r0)
    t_prev = None; prev_mag = None; t_last_mag = None
    _t: list[float] = []; _q: list[list[float]] = []

    for i in range(len(t_intg)):
        ti = t_intg[i]

        if t_prev is not None:
            dt = ti - t_prev
            if 0 < dt <= 1.0:
                rot = rot * Rotation.from_rotvec(
                    np.array([omega_x[i], omega_y[i], omega_z[i]]) * dt
                )
        t_prev = ti

        if apply_mag:
            ms = np.array([mx_intg[i], my_intg[i], mz_intg[i]])
            if np.all(np.isfinite(ms)) and (prev_mag is None or not np.allclose(ms, prev_mag)):
                dt_m = 0.1 if t_last_mag is None else (ti - t_last_mag)
                prev_mag = ms.copy(); t_last_mag = ti

                mb = ms[mag_order] * mag_signs
                mn = np.linalg.norm(mb)
                if mn > 1e-6:
                    alpha = float(np.clip(dt_m / MAG_YAW_TAU, 0.0, 1.0))
                    mb_ned = rot.apply(mb / mn)
                    axis  = np.cross(mb_ned, WMM_NED_UNIT)
                    sin_a = np.linalg.norm(axis)
                    cos_a = float(np.clip(np.dot(mb_ned, WMM_NED_UNIT), -1.0, 1.0))
                    ang   = float(np.arctan2(sin_a, cos_a))
                    if sin_a > 1e-6:
                        rot = Rotation.from_rotvec((axis / sin_a) * ang * alpha) * rot

        q = rot.as_quat()  # scipy: [x,y,z,w]
        _t.append(ti)
        _q.append([q[3], q[0], q[1], q[2]])  # store as [w,x,y,z]

    return np.array(_t), np.array(_q)


# =============================================================================
# Search all 48 axis-mapping combinations
# =============================================================================

print("Searching 48 magnetometer axis-mapping combinations...")
all_results: list[tuple[float, np.ndarray, np.ndarray]] = []

for order in permutations([0, 1, 2]):
    for signs in iproduct([1.0, -1.0], repeat=3):
        ord_a = np.array(order, dtype=int)
        sgn_a = np.array(signs)
        r0 = triad_init(ord_a, sgn_a)
        if r0 is None:
            continue
        t_arr, q_arr = integrate(r0, ord_a, sgn_a, apply_mag=True)
        rmse = angular_rmse(t_arr, q_arr, t_fc, fc_q_wxyz)
        all_results.append((rmse, ord_a, sgn_a, r0, t_arr, q_arr))

all_results.sort(key=lambda x: x[0])

print(f"\nTop 5 combinations (RMSE vs FC):")
for rank, (rmse, ord_a, sgn_a, *_) in enumerate(all_results[:5]):
    print(f"  #{rank+1}  order={ord_a}  signs={sgn_a.astype(int)}  RMSE={rmse:.2f}°")

best_rmse, best_order, best_signs, best_r0, best_t, best_q = all_results[0]
print(f"\nSelected: order={best_order}  signs={best_signs.astype(int)}  RMSE={best_rmse:.2f}°")

# Pure gyro track — same init as best combination, no mag correction
t_pure, q_pure = integrate(best_r0, best_order, best_signs, apply_mag=False)

# =============================================================================
# Euler angles and angular error
# =============================================================================

yaw_best,  pitch_best,  roll_best  = to_euler_deg(best_q)
yaw_pure,  pitch_pure,  roll_pure  = to_euler_deg(q_pure)

fc_yaw_deg   = np.degrees(yaw_fc)
fc_pitch_deg = np.degrees(pitch_fc)
fc_roll_deg  = np.degrees(roll_fc)

yaw_best  = align_start(best_t, yaw_best,  t_fc, fc_yaw_deg)
pitch_best = align_start(best_t, pitch_best, t_fc, fc_pitch_deg)
roll_best  = align_start(best_t, roll_best,  t_fc, fc_roll_deg)
yaw_pure  = align_start(t_pure, yaw_pure,  t_fc, fc_yaw_deg)
pitch_pure = align_start(t_pure, pitch_pure, t_fc, fc_pitch_deg)
roll_pure  = align_start(t_pure, roll_pure,  t_fc, fc_roll_deg)
err_best = angular_err_series(best_t, best_q, t_fc, fc_q_wxyz)
err_pure = angular_err_series(t_pure, q_pure, t_fc, fc_q_wxyz)

pure_rmse = angular_rmse(t_pure, q_pure, t_fc, fc_q_wxyz)
print(f"Pure gyro RMSE vs FC: {pure_rmse:.2f}°")

# =============================================================================
# Load EqF output (optional)
# =============================================================================

have_eqf = False
t_eqf = filt_roll = filt_pitch = filt_yaw = ang_err_filt = np.zeros(0)

try:
    eqf = np.genfromtxt(eqf_out, delimiter=",", skip_header=1)
    if eqf.ndim == 1:
        eqf = eqf.reshape(1, -1)
    have_eqf = eqf.shape[0] > 0 and eqf.shape[1] >= 33
except Exception:
    pass

if have_eqf:
    valid_eqf = np.isfinite(eqf[:, :29]).all(axis=1)
    eqf       = eqf[valid_eqf]
    t_eqf     = eqf[:, _E["t"]]
    q_filt    = eqf[:, _E["q"]]
    fe = Rotation.from_quat(q_filt[:, [1, 2, 3, 0]]).as_euler('ZYX', degrees=True)
    filt_yaw   = align_start(t_eqf, np.unwrap(fe[:, 0], discont=180, period=360), t_fc, fc_yaw_deg)
    filt_pitch = align_start(t_eqf, np.unwrap(fe[:, 1], discont=180, period=360), t_fc, fc_pitch_deg)
    filt_roll  = align_start(t_eqf, np.unwrap(fe[:, 2], discont=180, period=360), t_fc, fc_roll_deg)
    ang_err_filt = angular_err_series(t_eqf, q_filt, t_fc, fc_q_wxyz)
    print(f"EqF filter RMSE vs FC: {angular_rmse(t_eqf, q_filt, t_fc, fc_q_wxyz):.2f}°")

# =============================================================================
# Plot
# =============================================================================

fig, axes = plt.subplots(5, 1, figsize=(14, 18), sharex=True)
fig.suptitle(
    f"Attitude: gyro+mag (best axis mapping) vs FC  [{DATASET}]\n"
    f"Best: order={best_order}  signs={best_signs.astype(int)}  "
    f"RMSE={best_rmse:.1f}°  |  Pure gyro RMSE={pure_rmse:.1f}°",
    fontsize=11, fontweight='bold'
)

fc_sty   = dict(color='red',     lw=1.5, ls='--', alpha=0.85, zorder=3, label='FC')
filt_sty = dict(color='blue',    lw=1.2, ls='-',  alpha=0.85, zorder=3, label='EqF filter')
best_sty = dict(color='#e67e22', lw=1.0, ls='-',  alpha=0.85, zorder=2,
                label=f'Gyro + mag (best mapping, τ={MAG_YAW_TAU}s)')
pure_sty = dict(color='#7f8c8d', lw=0.8, ls='--', alpha=0.65, zorder=1, label='Pure gyro')

for idx, (title, ylabel, fc_v, filt_v, best_v, pure_v) in enumerate([
    ('Roll',  'Roll [deg]',  fc_roll_deg,  filt_roll  if have_eqf else None, roll_best,  roll_pure),
    ('Pitch', 'Pitch [deg]', fc_pitch_deg, filt_pitch if have_eqf else None, pitch_best, pitch_pure),
    ('Yaw',   'Yaw [deg]',   fc_yaw_deg,   filt_yaw   if have_eqf else None, yaw_best,   yaw_pure),
]):
    ax = axes[idx]
    ax.plot(t_fc,     fc_v,   **fc_sty)
    if filt_v is not None:
        ax.plot(t_eqf,    filt_v, **filt_sty)
    ax.plot(best_t,   best_v, **best_sty)
    ax.plot(t_pure,   pure_v, **pure_sty)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=4)
    ax.grid(True, alpha=0.4)

# Angular error
ax = axes[3]
ax.plot(best_t, err_best, color='#e67e22', lw=1.0,
        label=f'Gyro+mag vs FC  (RMSE={best_rmse:.1f}°)')
ax.plot(t_pure, err_pure, color='#7f8c8d', lw=0.8, alpha=0.7,
        label=f'Pure gyro vs FC  (RMSE={pure_rmse:.1f}°)')
if have_eqf:
    ax.plot(t_eqf, ang_err_filt, color='blue', lw=1.2,
            label=f'EqF filter vs FC  (RMSE={angular_rmse(t_eqf, q_filt, t_fc, fc_q_wxyz):.1f}°)')
ax.set_ylabel('Error [deg]')
ax.set_title('Angular error vs FC')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.4)

# Raw gyro
ax = axes[4]
ax.plot(t_gyro, gyro_N, color='#e74c3c', lw=0.8, alpha=0.8, label='Gyro X/North (gx) [deg/s]')
ax.plot(t_gyro, gyro_E, color='#27ae60', lw=0.8, alpha=0.8, label='Gyro Y/East  (gy) [deg/s]')
ax.plot(t_gyro, gyro_D, color='#2980b9', lw=0.8, alpha=0.8, label='Gyro Z/Down  (gz) [deg/s]')
ax.set_ylabel('Rate [deg/s]')
ax.set_ylim(-180, 180)
ax.set_title('Raw gyro (body frame, CSV units)')
ax.set_xlabel('Time [s]')
ax.legend(fontsize=8, ncol=3)
ax.grid(True, alpha=0.4)

plt.tight_layout()
os.makedirs("outputs", exist_ok=True)
out_path = f"outputs/mag_attitude_{DATASET}.png"
plt.savefig(out_path, dpi=150)
print(f"Saved to {out_path}")
plt.show()
