"""Standalone CNS TG-EqF inspired inertial/GNSS filter using pylie."""

import os

import numpy as np
from dataclasses import dataclass, field
from scipy.linalg import expm
from pylie import SO3, SE23


# =============================================================================
# Constants
# =============================================================================

g = 9.81


# =============================================================================
# Helpers
# =============================================================================


def col(x):
    x = np.asarray(x, dtype=float)
    return x.reshape(-1, 1)


def skew(v):
    return SO3.wedge(col(v))


def sym(A):
    return 0.5 * (A + A.T)


def safe_inv(A, eps=1e-9):
    A = sym(A)
    A = A + eps * np.eye(A.shape[0])
    return np.linalg.inv(A)


def blockdiag(*arrs):
    """Create block diagonal matrix from array list."""
    n = sum(a.shape[0] for a in arrs)
    m = sum(a.shape[1] for a in arrs)
    out = np.zeros((n, m))
    r = 0
    c = 0
    for a in arrs:
        rr, cc = a.shape
        out[r : r + rr, c : c + cc] = a
        r += rr
        c += cc
    return out


# =============================================================================
# Input Space
# =============================================================================


@dataclass
class InputSpace:
    w: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    a: np.ndarray = field(default_factory=lambda: np.zeros((3, 1)))
    mu: np.ndarray = field(default_factory=lambda: np.zeros((3, 1)))
    tau: np.ndarray = field(default_factory=lambda: np.zeros((9, 1)))

    def as_W_mat(self):
        """Convert input to W matrix form."""
        M = np.zeros((5, 5))
        M[0:3, 0:3] = self.w
        M[0:3, 3:4] = self.a
        M[0:3, 4:5] = self.mu
        return M

    def as_W_vec(self):
        """Convert input to W vector form."""
        out = np.zeros((9, 1))
        out[0:3, 0:1] = col(SO3.vee(self.w))
        out[3:6, 0:1] = self.a
        out[6:9, 0:1] = self.mu
        return out


# =============================================================================
# Filter State
# =============================================================================


@dataclass
class FilterState:
    """State: (R, v, p) on SE23, bias vector, covariance."""

    X: SE23 = field(default_factory=SE23.identity)
    b: np.ndarray = field(default_factory=lambda: np.zeros((9, 1)))
    P: np.ndarray = field(default_factory=lambda: np.eye(18) * 0.1)


# =============================================================================
# Gravity Matrix
# =============================================================================

G = np.zeros((5, 5))
G[2, 3] = -g


# =============================================================================
# Symmetry Lift
# =============================================================================


def f_10(T):
    """Symmetry lift correction term."""
    F = np.zeros((5, 5))
    F[0:3, 4:5] = T[0:3, 3:4]
    return F


def continuous_lift(state: FilterState, U: InputSpace):
    """Compute continuous lift (tangent-space dynamics)."""
    X = state.X
    b = state.b
    L = np.zeros((18, 1))

    term = (
        U.as_W_vec()
        - b
        + col(SE23.vee(X.inv().as_matrix() @ (G + f_10(X.as_matrix()))))
    )

    L[0:9] = term
    L[9:18] = -U.tau
    return L


# =============================================================================
# Propagation Jacobian
# =============================================================================

def A_matrix():
    """Continuous-time propagation matrix."""
    A = np.zeros((18, 18))
    A[0:9, 9:18] = -np.eye(9)
    return A


def B_matrix():
    """Process noise input matrix."""
    return np.eye(18)


# =============================================================================
# Filter
# =============================================================================


class TGEqF:
    """TG-EqF Inertial/GNSS Filter."""

    def __init__(self):
        """Initialize filter state and process noise."""
        self.state = FilterState()
        self.t_prev = None

        q_nav = 1e-2
        q_bias = 1e-5
        self.Q = blockdiag(np.eye(9) * q_nav, np.eye(9) * q_bias)

    # =========================================================================
    # Propagation
    # =========================================================================

    def propagate(self, t, gyro, accel):
        """Propagate state using gyro and accel measurements."""
        gyro = col(gyro)
        accel = col(accel)

        if self.t_prev is None:
            self.t_prev = t
            return

        dt = t - self.t_prev
        if dt <= 0 or dt > 1.0:
            self.t_prev = t
            return

        U = InputSpace()
        U.w = skew(gyro)
        U.a = accel
        U.mu = self.state.b[6:9]

        lift = continuous_lift(self.state, U)

        self.state.X = self.state.X * SE23.exp(lift[0:9] * dt)
        self.state.b = self.state.b + lift[9:18] * dt

        A = A_matrix()
        Phi = expm(A * dt)
        self.state.P = Phi @ self.state.P @ Phi.T + self.Q * dt
        self.state.P = sym(self.state.P)

        self.t_prev = t

    # =========================================================================
    # Position Correction
    # =========================================================================

    def correct_position(self, pos_NED, R_pos):
        """Correct position estimate using GNSS measurement."""
        z = col(pos_NED)
        p_hat = col(self.state.X.w().as_vector())

        innov = z - p_hat

        H = np.zeros((3, 18))
        H[:, 6:9] = np.eye(3)

        P = self.state.P
        S = H @ P @ H.T + R_pos
        S = sym(S) + 1e-9 * np.eye(3)

        K = P @ H.T @ safe_inv(S)
        dx = K @ innov

        self.state.X = self.state.X * SE23.exp(dx[0:9])
        self.state.b = self.state.b + dx[9:18]

        I = np.eye(18)
        self.state.P = (I - K @ H) @ P @ (I - K @ H).T + K @ R_pos @ K.T
        self.state.P = sym(self.state.P)

    # =========================================================================

    def output_row(self, t):
        """Extract state as output row [t, p, v, R, b_gyro, b_accel]."""
        R = self.state.X.R().as_matrix()
        v = col(self.state.X.x().as_vector())
        p = col(self.state.X.w().as_vector())
        b = self.state.b

        return [
            t,
            p[0, 0],
            p[1, 0],
            p[2, 0],
            v[0, 0],
            v[1, 0],
            v[2, 0],
            R[0, 0],
            R[0, 1],
            R[0, 2],
            R[1, 0],
            R[1, 1],
            R[1, 2],
            R[2, 0],
            R[2, 1],
            R[2, 2],
            b[0, 0],
            b[1, 0],
            b[2, 0],
            b[3, 0],
            b[4, 0],
            b[5, 0],
        ]


# =============================================================================
# CSV I/O
# =============================================================================

# NIMBUS24 FC CSV column indices
_C = {
    "t": 0,
    "lon": 1,
    "lat": 2,
    "alt": 3,
    "vn": 4,
    "ve": 5,
    "vd": 6,
    "ax": 9,
    "ay": 10,
    "az": 11,
    "gx": 15,
    "gy": 16,
    "gz": 17,
}
R_EARTH = 6_378_137.0


def _gps_to_ned(lat, lon, alt, lat0, lon0, alt0):
    """Convert GPS (lat/lon/alt) to NED relative to reference."""
    dlat = np.radians(lat - lat0)
    dlon = np.radians(lon - lon0)
    lat0_rad = np.radians(lat0)

    north = dlat * R_EARTH
    east = dlon * np.cos(lat0_rad) * R_EARTH
    down = -(alt - alt0)
    return np.array([north, east, down])


def run(csv_in="data/20241011_NIMBUS24_Flight_FC_Data.csv",
        csv_out="outputs/tg_eqf_output.csv"):
    """Run filter on NIMBUS24 FC CSV data."""
    raw = np.genfromtxt(csv_in, delimiter=",", skip_header=1)

    if raw.ndim == 1:
        raw = raw.reshape(1, -1)

    print(f"Loaded {len(raw)} rows")
    os.makedirs(os.path.dirname(csv_out), exist_ok=True)

    valid = (raw[:, _C["lat"]] != 0) & (raw[:, _C["lon"]] != 0)
    first = np.argmax(valid)
    lat0 = raw[first, _C["lat"]]
    lon0 = raw[first, _C["lon"]]
    alt0 = raw[first, _C["alt"]] / 1000.0

    filt = TGEqF()
    out = []
    prev_lat, prev_lon = None, None
    R_pos = np.eye(3) * 5.0**2

    for row in raw:
        t = row[_C["t"]]
        if not np.isfinite(t):
            continue

        gyro = row[[_C["gx"], _C["gy"], _C["gz"]]]
        accel = row[[_C["ax"], _C["ay"], _C["az"]]] * 9.80665

        if not np.all(np.isfinite(np.concatenate([gyro, accel]))):
            continue

        filt.propagate(t, gyro, accel)

        lat = row[_C["lat"]]
        lon = row[_C["lon"]]
        if lat != 0 and lon != 0 and (lat != prev_lat or lon != prev_lon):
            alt = row[_C["alt"]] / 1000.0
            pos_NED = _gps_to_ned(lat, lon, alt, lat0, lon0, alt0)
            filt.correct_position(pos_NED, R_pos)
            prev_lat, prev_lon = lat, lon

        out.append(filt.output_row(t))

    out = np.asarray(out)
    header = (
        "t,px,py,pz,vx,vy,vz,"
        "r00,r01,r02,r10,r11,r12,r20,r21,r22,"
        "bgx,bgy,bgz,bax,bay,baz"
    )

    np.savetxt(csv_out, out, delimiter=",", header=header, comments="")
    print(f"Wrote {len(out)} rows to {csv_out}")


if __name__ == "__main__":
    cin = "data/20241011_NIMBUS24_Flight_FC_Data.csv"
    cout = "outputs/tg_eqf_output.csv"
    run(cin, cout)