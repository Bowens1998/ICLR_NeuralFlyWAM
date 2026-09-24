"""Closed-loop MPC evaluation of trained world models in the simulator (D15).

For each trained checkpoint, an MPPI controller uses the model as its
dynamics predictor to track a figure-8 under winds inside and beyond the
training envelope. The question the study answers: does the LayerNorm
effect found in open-loop prediction carry through to closed-loop control
performance?

References run alongside the learned models:
  pd        the baseline PD controller (no model, no preview)
  truth     MPPI planning through the real simulator dynamics (upper bound)

Usage:
  python scripts/sim/closed_loop.py [--models m1_history_gru,...]
      [--seeds 0-9] [--winds 0,2.5,4.9,6.1,8.5,12.1] [--duration 30]
Results append to reports/closed_loop/results.csv (one row per episode).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from latent_aero_wam.evaluation.evaluator import load_model_from_checkpoint  # noqa: E402
from latent_aero_wam.sim import QuadrotorSim, figure8_trajectory  # noqa: E402
from latent_aero_wam.sim.quadrotor import GRAVITY  # noqa: E402
from latent_aero_wam.utils.rotation import (  # noqa: E402
    matrix_to_rot6d_np,
    so3_exp_np,
    so3_log_np,
)

RUNS = REPO / "runs" / "results" / "sim_round1"
OUT = REPO / "reports" / "closed_loop"
K, H, DT = 100, 50, 0.02
WIND_DIR = np.array([1.0, 0.0, 0.0])


# --------------------------------------------------------------------- MPPI
class MPPI:
    """Plans in *perturbation space* around a stabilizing nominal that is
    re-anchored every step. Perturbations are (dT, tilt_x, tilt_y); the
    solution is a softmin-weighted mean of perturbations (never of
    quaternions), with a quadratic penalty pulling toward the nominal so the
    optimizer cannot wander off exploiting model error."""

    def __init__(
        self, predict, n_samples=192, sigma_t=0.04, sigma_tilt=0.08, lam=0.05, reg=0.05, rng=None
    ):
        self.predict = predict
        self.n = n_samples
        self.sigma = np.array([sigma_t, sigma_tilt, sigma_tilt])
        self.lam = lam
        self.reg = reg
        self.rng = rng or np.random.default_rng(0)
        self.prev: np.ndarray | None = None  # (H, 3) previous perturbation solution

    @staticmethod
    def apply(nominal: np.ndarray, pert: np.ndarray) -> np.ndarray:
        """nominal (H,5), pert (...,H,3) -> actions (...,H,5)."""
        b = pert.shape[:-2]
        h = nominal.shape[0]
        act = np.broadcast_to(nominal, b + (h, 5)).copy()
        act[..., 0] = np.clip(act[..., 0] + pert[..., 0], 0.05, 1.0)
        rotvec = np.zeros(b + (h, 3))
        rotvec[..., 0] = pert[..., 1]
        rotvec[..., 1] = pert[..., 2]
        R_nom = quat_to_matrix_batch(nominal[:, 1:5])  # (H,3,3)
        R_new = np.broadcast_to(R_nom, b + (h, 3, 3)) @ so3_exp_np(rotvec)
        act[..., 1:5] = matrix_to_quat_batch(R_new.reshape(-1, 3, 3)).reshape(b + (h, 4))
        return act

    def plan(self, nominal: np.ndarray, p0: np.ndarray, p_ref: np.ndarray, v_ref: np.ndarray):
        h = len(nominal)
        prev = (
            np.concatenate([self.prev[1:], np.zeros((1, 3))], axis=0)
            if self.prev is not None
            else np.zeros((h, 3))
        )
        eps = self.rng.normal(0, 1, (self.n, h, 3))
        for t in range(1, h):  # AR(1) smoothing keeps candidates dynamically plausible
            eps[:, t] = 0.7 * eps[:, t - 1] + 0.3 * eps[:, t]
        pert = prev[None] + self.sigma * eps
        pert[0] = 0.0  # the pure nominal
        pert[1] = prev  # the shifted previous solution

        v_hat = self.predict(self.apply(nominal, pert))  # (B, H, 3)
        p_hat = p0[None, None] + DT * np.cumsum(v_hat, axis=1)
        cost = (np.linalg.norm(p_hat - p_ref[None], axis=-1) ** 2).sum(1)
        cost += 0.2 * (np.linalg.norm(v_hat - v_ref[None], axis=-1) ** 2).sum(1)
        cost += self.reg * ((pert / self.sigma) ** 2).sum((1, 2)) / h
        w = np.exp(-(cost - cost.min()) / (self.lam * (cost.std() + 1e-9)))
        w /= w.sum()
        sol = (w[:, None, None] * pert).sum(0)
        self.prev = sol
        return self.apply(nominal, sol[None])[0][0]


def quat_to_matrix_batch(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
            np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
            np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
        ],
        axis=-2,
    )


def matrix_to_quat_batch(R: np.ndarray) -> np.ndarray:
    w = 0.5 * np.sqrt(np.maximum(1.0 + np.trace(R, axis1=-2, axis2=-1), 1e-12))
    return np.stack(
        [
            (R[:, 2, 1] - R[:, 1, 2]) / (4 * w),
            (R[:, 0, 2] - R[:, 2, 0]) / (4 * w),
            (R[:, 1, 0] - R[:, 0, 1]) / (4 * w),
            w,
        ],
        axis=-1,
    )


# ----------------------------------------------------------- model adapters
class LearnedDynamics:
    """Wraps a trained checkpoint as an MPPI predictor with online history.

    Buffer contract matches WindowDataset at centre t: history slices cover
    steps t-K..t-1 (states, v_body, actions, deltas), ``current_*`` is step t,
    and the candidate actions are ``future_action[t:t+H]``.
    """

    def __init__(self, ckpt: Path, device: torch.device):
        self.model, _, _ = load_model_from_checkpoint(ckpt, device)
        self.device = device
        self.states: list[np.ndarray] = []
        self.v_bodies: list[np.ndarray] = []
        self.Rs: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.deltas: list[np.ndarray] = []

    def observe_state(self, v, R, omega) -> None:
        state = np.concatenate([v, matrix_to_rot6d_np(R[None])[0], omega]).astype(np.float32)
        if self.states:
            prev, R_prev = self.states[-1], self.Rs[-1]
            self.deltas.append(
                np.concatenate(
                    [
                        state[:3] - prev[:3],
                        so3_log_np((R_prev.T @ R)[None])[0],
                        state[9:] - prev[9:],
                    ]
                ).astype(np.float32)
            )
        self.states.append(state)
        self.v_bodies.append((R.T @ v).astype(np.float32))
        self.Rs.append(R.copy())

    def observe_action(self, action) -> None:
        self.actions.append(np.asarray(action, dtype=np.float32))

    def ready(self) -> bool:
        return len(self.states) >= K + 1 and len(self.actions) >= K

    @torch.no_grad()
    def __call__(self, actions: np.ndarray) -> np.ndarray:
        b = len(actions)
        dev = self.device

        def rep(x, dims):
            return (
                torch.tensor(np.asarray(x, dtype=np.float32)).unsqueeze(0).repeat(b, *dims).to(dev)
            )

        batch = {
            "history_state": rep(self.states[-K - 1 : -1], (1, 1)),
            "history_v_body": rep(self.v_bodies[-K - 1 : -1], (1, 1)),
            "history_action": rep(self.actions[-K:], (1, 1)),
            "history_delta_state": rep(self.deltas[-K:], (1, 1)),
            "current_state": rep(self.states[-1], (1,)),
            "current_v_body": rep(self.v_bodies[-1], (1,)),
            "current_R": rep(self.Rs[-1], (1, 1)),
            "future_action": torch.tensor(actions.astype(np.float32)).to(dev),
        }
        out = self.model(batch)
        return out.state[..., 0:3].cpu().numpy()


class TruthDynamics:
    """MPPI predictor that rolls the real simulator forward, vectorized over
    candidates (mean wind, no gusts, no actuation noise — the idealized
    planning model)."""

    def __init__(self, sim: QuadrotorSim, wind_fn):
        self.p_ = sim.p_
        self.wind_fn = wind_fn
        self.t = 0.0
        self.state: dict | None = None

    def __call__(self, actions: np.ndarray) -> np.ndarray:
        pp = self.p_
        b, h, _ = actions.shape
        v = np.repeat(self.state["v"][None], b, axis=0)
        R = np.repeat(self.state["R"][None], b, axis=0)
        w = np.repeat(self.state["w"][None], b, axis=0)
        d1, d2 = np.asarray(pp.drag_lin), np.asarray(pp.drag_quad)
        hh = pp.dt / pp.n_substeps
        out = np.zeros((b, h, 3))
        for j in range(h):
            R_sp = quat_to_matrix_batch(actions[:, j, 1:5])
            t_sp = np.clip(actions[:, j, 0], 0.0, 1.0)[:, None]
            wind = self.wind_fn(self.t + j * DT)[None]
            for _ in range(pp.n_substeps):
                w_cmd = pp.k_att * so3_log_np(np.transpose(R, (0, 2, 1)) @ R_sp)
                w = w + hh * (w_cmd - w) / pp.tau_omega
                R = R @ so3_exp_np(w * hh)
                thrust = t_sp * pp.t_max * R[:, :, 2]
                v_air_body = np.einsum("bji,bj->bi", R, v - wind)
                f_body = -(d1 + d2 * np.abs(v_air_body)) * v_air_body
                f = thrust + np.einsum("bij,bj->bi", R, f_body) + pp.mass * GRAVITY
                v = v + hh * f / pp.mass
            out[:, j] = v
        return out


# ---------------------------------------------------------------- episodes
def pd_plan(sim, state, traj, v_ref_all, a_ref_all, i, horizon, decay=0.85):
    """Stabilizing nominal over the horizon: reference feedforward plus a
    geometrically decaying copy of the current PD correction. (H, 5)."""
    pp = sim.p_
    e = pp.kp * (traj[i] - state["p"]) + pp.kd * (v_ref_all[i] - state["v"])
    acts = np.zeros((horizon, 5))
    x_c = np.array([1.0, 0.0, 0.0])
    for h in range(horizon):
        a_des = a_ref_all[i + 1 + h] - GRAVITY + (decay**h) * e
        f_des = pp.mass * a_des
        f_norm = np.linalg.norm(f_des)
        acts[h, 0] = np.clip(f_norm / pp.t_max, 0.05, 1.0)
        z_b = f_des / max(f_norm, 1e-6)
        y_b = np.cross(z_b, x_c)
        y_b /= max(np.linalg.norm(y_b), 1e-6)
        x_b = np.cross(y_b, z_b)
        R_sp = np.stack([x_b, y_b, z_b], axis=-1)
        acts[h, 1:5] = matrix_to_quat_batch(R_sp[None])[0]
    return acts


def run_episode(kind: str, ckpt: Path | None, wind: float, duration: float, device, seed=0):
    sim = QuadrotorSim(seed=100 + seed)
    wind_fn = lambda t: wind * WIND_DIR  # noqa: E731
    traj = figure8_trajectory(duration + (K + 5) * DT, DT)
    v_ref_all = np.gradient(traj, DT, axis=0)
    a_ref_all = np.gradient(v_ref_all, DT, axis=0)
    state = {"p": traj[0].copy(), "v": np.zeros(3), "R": np.eye(3), "w": np.zeros(3)}
    gust = np.zeros(3)

    dyn = None
    ctrl = None
    if kind == "model":
        dyn = LearnedDynamics(ckpt, device)
        ctrl = MPPI(dyn, rng=np.random.default_rng(seed))
    elif kind == "truth":
        dyn = TruthDynamics(QuadrotorSim(seed=999), wind_fn)
        ctrl = MPPI(dyn, n_samples=64, rng=np.random.default_rng(seed))

    errs = []
    n = len(traj) - H - 1
    for i in range(n):
        t = i * DT
        wind_mean = wind_fn(t)
        speed = np.linalg.norm(wind_mean)
        gust = (
            gust
            + DT * (-gust / sim.p_.gust_tau)
            + sim.rng.normal(
                0, sim.p_.gust_std * max(speed, 0.3) * np.sqrt(2 * DT / sim.p_.gust_tau), 3
            )
        )
        if kind == "model":
            dyn.observe_state(state["v"], state["R"], state["w"])
        # nominal from the PD baseline along the reference
        t_sp, R_sp = sim.baseline_controller(state, traj[i], v_ref_all[i])
        q_sp = matrix_to_quat_batch(R_sp[None])[0]
        pd_action = np.concatenate([[t_sp], q_sp])

        use_mpc = kind != "pd" and (kind != "model" or dyn.ready())
        if use_mpc:
            if kind == "truth":
                dyn.state = {k: v.copy() for k, v in state.items()}
                dyn.t = t
            nominal = pd_plan(sim, state, traj, v_ref_all, a_ref_all, i, H)
            act = ctrl.plan(
                nominal, state["p"], traj[i + 1 : i + 1 + H], v_ref_all[i + 1 : i + 1 + H]
            )
        else:
            act = pd_action

        if kind == "model":
            dyn.observe_action(act)
        R_cmd = quat_to_matrix_batch(act[None, 1:5])[0]
        state = sim.step(state, float(act[0]), R_cmd, wind_fn(t) + gust)
        if i > K + 5:  # score only after the history warm-up
            errs.append(np.linalg.norm(state["p"] - traj[i + 1]))
    return float(np.sqrt(np.mean(np.square(errs))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--models",
        default="m1_history_gru,m2_latent_wam,m2_latent_wam_nonorm,"
        "m2_latent_wam_w64_nonorm,m2_latent_wam_magpass",
    )
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--winds", default="0,2.5,4.9,6.1,8.5,12.1")
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--references", default="pd,truth")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    winds = [float(w) for w in args.winds.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]

    OUT.mkdir(parents=True, exist_ok=True)
    out_csv = OUT / "results.csv"
    new = not out_csv.exists()
    with out_csv.open("a", newline="") as fh:
        wr = csv.writer(fh)
        if new:
            wr.writerow(["controller", "seed", "wind_mps", "rmse_m"])
        for ref in [r for r in args.references.split(",") if r]:
            for wind in winds:
                rmse = run_episode(ref, None, wind, args.duration, device)
                wr.writerow([ref, 0, wind, f"{rmse:.4f}"])
                fh.flush()
                print(f"{ref:28s} wind {wind:5.1f}  rmse {rmse:.3f}")
        for model in [m for m in args.models.split(",") if m]:
            for seed in seeds:
                ckpt = RUNS / f"{model}_seed{seed}" / "checkpoint_best.pt"
                if not ckpt.exists():
                    print(f"skip {model} seed {seed} (no checkpoint)")
                    continue
                for wind in winds:
                    rmse = run_episode("model", ckpt, wind, args.duration, device, seed)
                    wr.writerow([model, seed, wind, f"{rmse:.4f}"])
                    fh.flush()
                    print(f"{model:28s} seed {seed} wind {wind:5.1f}  rmse {rmse:.3f}")


if __name__ == "__main__":
    main()
