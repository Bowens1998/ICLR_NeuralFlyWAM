"""Neural-Fly CSV parser and per-flight feature construction.

Produces, for one logged flight, the arrays the rest of the pipeline consumes:

    state        (T, 12)  [v (world), rot6d, omega (body)]
    v_body       (T, 3)   R^T v -- a deterministic function of the state,
                          supplied as an input feature because aerodynamic
                          forces depend on body-frame airspeed. Not a target.
    action       (T, 5)   [T_sp, q_sp(4)]
    delta_state  (T-1, 9) [dv, so3_log(R_t^T R_{t+1}), domega]

Wind condition is metadata only: it is used for splits, evaluation and the
diagnostic oracle, and never reaches M0-M3.
"""

from __future__ import annotations

import hashlib
import re
from ast import literal_eval
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..utils.rotation import (
    enforce_quat_hemisphere,
    matrix_to_rot6d_np,
    project_to_so3,
    quat_to_matrix_np,
    so3_log_np,
)

STATE_DIM = 12
ACTION_DIM = 5
DELTA_DIM = 9
VBODY_DIM = 3

STATE_SLICES = {"v": slice(0, 3), "rot6d": slice(3, 9), "omega": slice(9, 12)}
DELTA_SLICES = {"dv": slice(0, 3), "dtheta": slice(3, 6), "domega": slice(6, 9)}

#: fan-array duty cycle -> wind speed [m/s], from the Neural-Fly README.
WIND_MPS = {
    "nowind": 0.0,
    "10wind": 1.3,
    "20wind": 2.5,
    "30wind": 3.7,
    "35wind": 4.2,
    "40wind": 4.9,
    "50wind": 6.1,
    "70wind": 8.5,
    "70p20sint": 8.5,  # 8.5 + 2.4 sin(t); nominal value, time-varying
    "100wind": 12.1,
}
#: conditions whose wind speed varies within the flight
TIME_VARYING = {"70p20sint"}

_FILENAME_FIELDS = ("vehicle", "trajectory", "method", "condition")
_ARRAY_COLS = ("p", "p_d", "v", "v_d", "q", "R", "w", "T_sp", "q_sp", "hover_throttle", "fa", "pwm")
#: the transfer drone (Intel-Aero) logs a reduced schema: no gyro column, no
#: attitude/thrust setpoints; motor PWMs are the only actuation record.
_ARRAY_COLS_INTEL = ("p", "v", "q", "R", "fa", "pwm")


@dataclass
class Flight:
    """One parsed flight log."""

    name: str
    path: Path
    vehicle: str
    trajectory: str
    method: str
    condition: str
    dt: float
    t: np.ndarray  # (T,)
    state: np.ndarray  # (T, 12)
    v_body: np.ndarray  # (T, 3)
    action: np.ndarray  # (T, 5)
    delta_state: np.ndarray  # (T-1, 9)
    R: np.ndarray  # (T, 3, 3) body->world, projected onto SO(3)
    p: np.ndarray  # (T, 3) position, metadata/diagnostics only

    @property
    def n_steps(self) -> int:
        return self.t.shape[0]

    @property
    def wind_mps(self) -> float:
        return WIND_MPS[self.condition]

    @property
    def wind_is_time_varying(self) -> bool:
        return self.condition in TIME_VARYING


class DataError(RuntimeError):
    """Raised when a flight log violates an assumption the pipeline relies on."""


def parse_filename(path: Path) -> dict[str, str]:
    stem = path.stem
    parts = stem.split("_")
    if len(parts) != len(_FILENAME_FIELDS):
        raise DataError(f"{path.name}: expected <VEHICLE>_<TRAJECTORY>_<METHOD>_<CONDITION>.csv")
    return dict(zip(_FILENAME_FIELDS, parts))


def _read_raw(path: Path) -> dict[str, np.ndarray]:
    df = pd.read_csv(path)
    out: dict[str, np.ndarray] = {}
    for col in df.columns[1:]:  # column 0 is the unnamed row index
        series = df[col]
        if isinstance(series.iloc[0], str):
            out[col] = np.array(series.apply(literal_eval).tolist(), dtype=np.float64)
        else:
            out[col] = np.asarray(series.tolist(), dtype=np.float64)
    return out


def _check(raw: dict[str, np.ndarray], path: Path, schema: str) -> float:
    """Run the checks that would silently corrupt training if violated."""
    required = _ARRAY_COLS if schema == "main" else _ARRAY_COLS_INTEL
    missing = [c for c in required if c not in raw]
    if missing:
        raise DataError(f"{path.name}: missing columns {missing}")

    for col, arr in raw.items():
        n_bad = int(np.sum(~np.isfinite(arr)))
        if n_bad:
            raise DataError(f"{path.name}: {n_bad} non-finite values in '{col}'")

    t = raw["t"]
    dt = np.diff(t)
    if not np.all(dt > 0):
        raise DataError(f"{path.name}: timestamps are not strictly increasing")
    dt_med = float(np.median(dt))
    if float(np.max(np.abs(dt - dt_med))) > 0.1 * dt_med:
        raise DataError(f"{path.name}: sampling interval is not uniform (median {dt_med:.5f}s)")

    dims = [("v", 3), ("q", 4)]
    dims += [("w", 3), ("q_sp", 4), ("T_sp", 1)] if schema == "main" else [("pwm", 4)]
    for col, dim in dims:
        if raw[col].shape[1:] != (dim,):
            raise DataError(f"{path.name}: '{col}' has shape {raw[col].shape}, expected (T, {dim})")
    if raw["R"].shape[1:] not in ((9,), (3, 3)):
        raise DataError(f"{path.name}: 'R' has shape {raw['R'].shape}")

    qn = np.linalg.norm(raw["q"], axis=-1)
    if float(np.max(np.abs(qn - 1.0))) > 1e-2:
        raise DataError(f"{path.name}: quaternion norms deviate from 1 by {np.max(np.abs(qn-1)):.3g}")
    if schema == "main":
        qspn = np.linalg.norm(raw["q_sp"], axis=-1)
        if float(np.max(np.abs(qspn - 1.0))) > 5e-2:
            raise DataError(f"{path.name}: q_sp norms deviate from 1 by {np.max(np.abs(qspn-1)):.3g}")
        if float(np.abs(raw["T_sp"]).max()) > 10.0:
            raise DataError(f"{path.name}: T_sp out of the expected normalised-throttle range")
    else:
        if raw["pwm"].min() < 800.0 or raw["pwm"].max() > 2200.0:
            raise DataError(f"{path.name}: pwm outside the expected 800-2200 range")
    return dt_med


def load_flight(
    path: str | Path, decimate: int = 1, transfer_rate_mode: str = "backward_causal_v2"
) -> Flight:
    """Parse one Neural-Fly CSV into a :class:`Flight`.

    ``decimate`` keeps every n-th sample; the transfer drone logs at 100 Hz and
    is decimated to the protocol's 50 Hz so K/H mean the same wall-clock time.
    """
    path = Path(path)
    meta = parse_filename(path)
    if meta["condition"] not in WIND_MPS:
        raise DataError(f"{path.name}: unknown wind condition '{meta['condition']}'")

    raw = _read_raw(path)
    schema = "main" if "T_sp" in raw else "intel"
    dt = _check(raw, path, schema)
    if decimate > 1:
        raw = {k: a[::decimate] for k, a in raw.items()}
        dt *= decimate

    q = enforce_quat_hemisphere(raw["q"] / np.linalg.norm(raw["q"], axis=-1, keepdims=True))
    # The logged R matches quat_to_matrix(q) (not its transpose), i.e. body->world.
    # It is only orthonormal to ~1.5e-3, so rebuild it from the unit quaternion
    # and project, which removes the bias that would otherwise leak into
    # geodesic errors and so(3) increments.
    R = project_to_so3(quat_to_matrix_np(q))

    v = raw["v"]
    if schema == "main":
        omega = raw["w"]
        action = np.concatenate([raw["T_sp"], raw["q_sp"]], axis=-1)
    else:
        # No gyro: the v2 estimate at t uses only R[t-1], R[t]. The interval
        # rotation axis is unchanged by transport between its endpoint frames.
        # This is a delayed interval-average rate, not an instantaneous gyro.
        R_rel_seq = np.einsum("nji,njk->nik", R[:-1], R[1:])
        interval_rate = so3_log_np(R_rel_seq) / dt
        if transfer_rate_mode == "backward_causal_v2":
            omega = np.concatenate([np.zeros((1, 3)), interval_rate], axis=0)
        elif transfer_rate_mode == "forward_legacy":
            # Historical checkpoints only; contains lookahead and is not an
            # admissible online prediction protocol.
            omega = np.concatenate([interval_rate, interval_rate[-1:]], axis=0)
        else:
            raise ValueError(f"unknown transfer_rate_mode: {transfer_rate_mode}")
        # Motor PWMs are the actuation record; scale to O(1) and pad with a
        # constant column so ACTION_DIM matches the main platform (whose q_sp
        # yaw channel is itself identically zero).
        pwm = (raw["pwm"] - 1500.0) / 500.0
        action = np.concatenate([pwm, np.zeros_like(pwm[:, :1])], axis=-1)
    state = np.concatenate([v, matrix_to_rot6d_np(R), omega], axis=-1)
    v_body = np.einsum("nji,nj->ni", R, v)  # R^T v

    # Body-frame rotation increment: R_{t+1} = R_t @ exp(dtheta^)
    R_rel = np.einsum("nji,njk->nik", R[:-1], R[1:])
    delta_state = np.concatenate(
        [np.diff(v, axis=0), so3_log_np(R_rel), np.diff(omega, axis=0)], axis=-1
    )

    return Flight(
        name=path.stem,
        path=path,
        dt=dt,
        t=raw["t"],
        state=state.astype(np.float32),
        v_body=v_body.astype(np.float32),
        action=action.astype(np.float32),
        delta_state=delta_state.astype(np.float32),
        R=R.astype(np.float32),
        p=raw["p"].astype(np.float32),
        **meta,
    )


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def discover_flights(root: str | Path, pattern: str) -> list[Path]:
    """List CSVs under ``root`` whose stem matches the regex ``pattern``."""
    root = Path(root)
    if not root.exists():
        raise DataError(f"data root does not exist: {root}")
    rx = re.compile(pattern)
    return sorted(p for p in root.rglob("*.csv") if rx.fullmatch(p.stem))
