"""Rollout error metrics.

Primary metric: **normalised multi-step rollout error**. Each component is
divided by the standard deviation of the corresponding *H-step change* on the
D1 training split, so a score of 1.0 means "as wrong as predicting no change at
all" and the components can be averaged. Component errors are always reported
alongside the aggregate.

    velocity      per-component |v_hat - v| / sigma_dv
    orientation   geodesic angle(R_hat, R) / sigma_dtheta
    angular rate  per-component |omega_hat - omega| / sigma_domega
    displacement  |sum(v_hat) dt - sum(v) dt| / sigma_dp   (secondary)
"""

from __future__ import annotations

import numpy as np
import torch

from ..utils.rotation import rot6d_to_matrix

COMPONENTS = ("velocity", "orientation", "angular_rate", "displacement")
#: components entering the headline aggregate (displacement is derived from
#: velocity and would double-count it)
AGGREGATE_COMPONENTS = ("velocity", "orientation", "angular_rate")


def fit_metric_scales(dataset, max_windows: int = 20000) -> dict[str, list]:
    """Std of the H-step change over a split, per component."""
    H, dt = dataset.H, dataset.dt
    dv, dth, dw, dp = [], [], [], []
    for b in dataset.blocks:
        f = dataset.store.get(b.file)
        t = np.arange(b.start, b.stop)
        if t.size == 0:
            continue
        dv.append(f.state[t + H, 0:3] - f.state[t, 0:3])
        dw.append(f.state[t + H, 9:12] - f.state[t, 9:12])
        R0, RH = f.R[t], f.R[t + H]
        rel = np.einsum("nji,njk->nik", R0, RH)
        tr = np.clip((np.trace(rel, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
        dth.append(np.arccos(tr))
        # relative displacement obtained by integrating the logged velocity
        cs = np.cumsum(f.state[:, 0:3] * dt, axis=0)
        dp.append(cs[t + H] - cs[t])
    return {
        "velocity": np.concatenate(dv).std(axis=0).tolist(),
        "orientation": [float(np.concatenate(dth).std())],
        "angular_rate": np.concatenate(dw).std(axis=0).tolist(),
        "displacement": np.concatenate(dp).std(axis=0).tolist(),
    }


class MetricScales:
    """Torch-side holder for the scales, with a per-horizon error computation."""

    def __init__(self, scales: dict[str, list], device=None, dtype=torch.float32):
        self.raw = scales
        self.t = {
            k: torch.as_tensor(v, device=device, dtype=dtype).clamp_min(1e-6)
            for k, v in scales.items()
        }

    def to(self, device, dtype=torch.float32) -> MetricScales:
        return MetricScales(self.raw, device=device, dtype=dtype)


def rollout_errors(
    pred_state: torch.Tensor,
    pred_R: torch.Tensor,
    true_state: torch.Tensor,
    true_R: torch.Tensor,
    scales: MetricScales,
    dt: float,
) -> dict[str, torch.Tensor]:
    """Per-window, per-horizon normalised errors. Each value is (B, H)."""
    from ..utils.rotation import geodesic_angle

    v_err = (pred_state[..., 0:3] - true_state[..., 0:3]).abs() / scales.t["velocity"]
    w_err = (pred_state[..., 9:12] - true_state[..., 9:12]).abs() / scales.t["angular_rate"]
    theta = geodesic_angle(pred_R, true_R) / scales.t["orientation"]
    p_pred = torch.cumsum(pred_state[..., 0:3] * dt, dim=1)
    p_true = torch.cumsum(true_state[..., 0:3] * dt, dim=1)
    p_err = (p_pred - p_true).abs() / scales.t["displacement"]
    return {
        "velocity": v_err.mean(-1),
        "orientation": theta,
        "angular_rate": w_err.mean(-1),
        "displacement": p_err.mean(-1),
    }


def aggregate(errors: dict[str, torch.Tensor]) -> torch.Tensor:
    """Unweighted mean of the aggregate components, shape (B, H)."""
    return torch.stack([errors[c] for c in AGGREGATE_COMPONENTS], 0).mean(0)


def true_R_from_state(state: torch.Tensor) -> torch.Tensor:
    return rot6d_to_matrix(state[..., 3:9])
