"""A deliberately small quadrotor + wind simulator for the closed-loop study.

Purpose (D15): the real logs cannot answer counterfactual questions ("what
would this rollout have done under a different action sequence?"), so the
closed-loop consequences of the LayerNorm finding are tested in simulation.
The simulator mirrors the physics the Neural-Fly setting exposes — thrust
along body z, attitude tracked by a fast inner loop, and an aerodynamic drag
force that depends on airspeed (velocity relative to wind) — with gust noise
so learning is non-trivial. It is calibrated to the same scales as the real
data (2 s history, 1 s horizon at 50 Hz; |v| of a couple m/s; winds
0-12 m/s), not fitted to it.

Episodes can be exported as CSVs in the *main-platform schema*, so the whole
existing pipeline (parser -> splits -> training -> evaluation) runs on sim
data unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..utils.rotation import so3_exp_np, so3_log_np

GRAVITY = np.array([0.0, 0.0, -9.81])


@dataclass
class SimParams:
    mass: float = 1.0
    t_max: float = 25.0  # max thrust [N]; hover at ~0.39 throttle
    tau_omega: float = 0.08  # inner-loop body-rate time constant [s]
    k_att: float = 9.0  # attitude P gain -> commanded body rate
    drag_lin: tuple = (0.25, 0.25, 0.40)  # linear drag [N s/m], body frame
    drag_quad: tuple = (0.020, 0.020, 0.035)  # quadratic drag [N s^2/m^2]
    gust_std: float = 0.08  # OU gust noise, fraction of mean wind speed
    gust_tau: float = 1.0  # gust correlation time [s]
    act_noise: float = 0.01  # actuation noise on throttle / rate command
    kp: float = 6.0  # position gain of the baseline controller
    kd: float = 4.0  # velocity gain
    dt: float = 0.02
    n_substeps: int = 10


def random_trajectory(rng: np.random.Generator, duration: float, dt: float) -> np.ndarray:
    """Smooth sum-of-sinusoids position reference, random3-like. (T, 3)."""
    t = np.arange(0.0, duration, dt)
    out = np.zeros((len(t), 3))
    for axis in range(3):
        amp_total = 1.5 if axis < 2 else 0.6
        for _ in range(3):
            f = rng.uniform(0.05, 0.35)
            a = rng.uniform(0.3, 1.0)
            phi = rng.uniform(0, 2 * np.pi)
            out[:, axis] += a * np.sin(2 * np.pi * f * t + phi)
        out[:, axis] *= amp_total / np.abs(out[:, axis]).max()
    out[:, 2] += 1.5  # hover height
    return out


def figure8_trajectory(duration: float, dt: float, period: float = 12.0) -> np.ndarray:
    t = np.arange(0.0, duration, dt)
    w = 2 * np.pi / period
    x = 1.5 * np.sin(w * t)
    y = 1.0 * np.sin(2 * w * t)
    z = np.full_like(t, 1.5)
    return np.stack([x, y, z], axis=-1)


class QuadrotorSim:
    """Rigid-body quadrotor with a first-order inner attitude/rate loop."""

    def __init__(self, params: SimParams | None = None, seed: int = 0):
        self.p_ = params or SimParams()
        self.rng = np.random.default_rng(seed)

    # ---------------------------------------------------------------- physics
    def aero_force(self, v: np.ndarray, R: np.ndarray, wind: np.ndarray) -> np.ndarray:
        """Drag on the airspeed vector, anisotropic in the body frame."""
        v_air_body = R.T @ (v - wind)
        d1 = np.asarray(self.p_.drag_lin)
        d2 = np.asarray(self.p_.drag_quad)
        f_body = -(d1 + d2 * np.abs(v_air_body)) * v_air_body
        return R @ f_body

    def step(self, state: dict, t_sp: float, R_sp: np.ndarray, wind: np.ndarray) -> dict:
        """Advance one control period (dt) with Euler substeps."""
        p, v, R, w = state["p"], state["v"], state["R"], state["w"]
        pp = self.p_
        h = pp.dt / pp.n_substeps
        t_sp = float(np.clip(t_sp + self.rng.normal(0, pp.act_noise), 0.0, 1.0))
        for _ in range(pp.n_substeps):
            w_cmd = pp.k_att * so3_log_np(R.T @ R_sp)
            w_cmd = w_cmd + self.rng.normal(0, pp.act_noise, 3)
            w = w + h * (w_cmd - w) / pp.tau_omega
            R = R @ so3_exp_np(w * h)
            thrust = t_sp * pp.t_max * R[:, 2]
            f = thrust + self.aero_force(v, R, wind) + pp.mass * GRAVITY
            v = v + h * f / pp.mass
            p = p + h * v
        return {"p": p, "v": v, "R": R, "w": w}

    # ------------------------------------------------------------- controller
    def baseline_controller(
        self, state: dict, p_ref: np.ndarray, v_ref: np.ndarray
    ) -> tuple[float, np.ndarray]:
        """PD position controller -> (normalized thrust, attitude setpoint)."""
        pp = self.p_
        a_des = pp.kp * (p_ref - state["p"]) + pp.kd * (v_ref - state["v"]) - GRAVITY
        f_des = pp.mass * a_des
        f_norm = np.linalg.norm(f_des)
        t_sp = float(np.clip(f_norm / pp.t_max, 0.05, 1.0))
        z_b = f_des / max(f_norm, 1e-6)
        # zero-yaw attitude from the desired body z (matches q_sp yaw == 0)
        x_c = np.array([1.0, 0.0, 0.0])
        y_b = np.cross(z_b, x_c)
        y_b /= max(np.linalg.norm(y_b), 1e-6)
        x_b = np.cross(y_b, z_b)
        R_sp = np.stack([x_b, y_b, z_b], axis=-1)
        return t_sp, R_sp

    # ---------------------------------------------------------------- episode
    def fly(
        self,
        traj: np.ndarray,
        wind_fn,
    ) -> dict[str, np.ndarray]:
        """Track ``traj`` with the baseline controller; log the main schema.

        ``wind_fn(t)`` returns the mean wind vector at time t; OU gusts are
        added on top, scaled by the mean wind speed.
        """
        pp = self.p_
        n = len(traj)
        v_ref = np.gradient(traj, pp.dt, axis=0)
        state = {"p": traj[0].copy(), "v": np.zeros(3), "R": np.eye(3), "w": np.zeros(3)}
        gust = np.zeros(3)
        logs: dict[str, list] = {k: [] for k in ("p", "v", "R", "w", "T_sp", "R_sp", "fa", "wind")}
        for i in range(n):
            t = i * pp.dt
            wind_mean = wind_fn(t)
            speed = np.linalg.norm(wind_mean)
            gust = (
                gust
                + pp.dt * (-gust / pp.gust_tau)
                + self.rng.normal(
                    0, pp.gust_std * max(speed, 0.3) * np.sqrt(2 * pp.dt / pp.gust_tau), 3
                )
            )
            wind = wind_mean + gust
            t_sp, R_sp = self.baseline_controller(state, traj[i], v_ref[i])
            logs["p"].append(state["p"].copy())
            logs["v"].append(state["v"].copy())
            logs["R"].append(state["R"].copy())
            logs["w"].append(state["w"].copy())
            logs["T_sp"].append(t_sp)
            logs["R_sp"].append(R_sp)
            logs["fa"].append(self.aero_force(state["v"], state["R"], wind))
            logs["wind"].append(wind.copy())
            state = self.step(state, t_sp, R_sp, wind)
        return {k: np.asarray(a) for k, a in logs.items()}
