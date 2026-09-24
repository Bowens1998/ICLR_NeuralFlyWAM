"""Training objective: one-step increment loss + multi-step rollout loss.

The one-step term is teacher-forced (ground-truth states in, increments out) and
keeps the transition model well conditioned; the rollout term is fully open-loop
and is what the evaluation actually measures. Both are computed on normalised
quantities so the nine increment channels contribute comparably.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..evaluation.metrics import MetricScales, aggregate, rollout_errors, true_R_from_state
from ..models.base import RolloutOutput


def horizon_weights(H: int, mode: str, device) -> torch.Tensor:
    if mode == "uniform":
        w = torch.ones(H, device=device)
    elif mode == "linear_decay":
        w = torch.linspace(1.0, 0.5, H, device=device)
    else:
        raise ValueError(f"unknown horizon weighting '{mode}'")
    return w / w.mean()


def compute_loss(
    model,
    batch: dict,
    out: RolloutOutput,
    scales: MetricScales,
    dt: float,
    w_onestep: float = 1.0,
    w_rollout: float = 1.0,
    w_jepa: float = 0.0,
    horizon_weighting: str = "uniform",
) -> tuple[torch.Tensor, dict[str, float]]:
    device = out.state.device
    H = out.state.shape[1]
    hw = horizon_weights(H, horizon_weighting, device)

    logs: dict[str, float] = {}
    loss = out.state.new_zeros(())

    if w_onestep > 0:
        target_delta = (batch["future_delta_state"] - model.delta_mean) / model.delta_std
        pred_delta = model.teacher_forced_delta(batch, out.context, out.memory_init)
        l_one = (F.mse_loss(pred_delta, target_delta, reduction="none").mean(-1) * hw).mean()
        loss = loss + w_onestep * l_one
        logs["loss_onestep"] = float(l_one.detach())

    if w_rollout > 0:
        true_R = true_R_from_state(batch["future_state"])
        errs = rollout_errors(out.state, out.R, batch["future_state"], true_R, scales, dt)
        # squared normalised error keeps the gradient scale sane across components
        l_roll = (aggregate({k: v.pow(2) for k, v in errs.items()}) * hw).mean()
        loss = loss + w_rollout * l_roll
        logs["loss_rollout"] = float(l_roll.detach())
        for k, v in errs.items():
            logs[f"train_{k}"] = float(v.mean().detach())

    if w_jepa > 0:
        if not hasattr(model, "jepa_loss"):
            raise TypeError(f"w_jepa > 0 but {type(model).__name__} has no jepa_loss")
        l_jepa = model.jepa_loss(batch, out.latent)
        loss = loss + w_jepa * l_jepa
        logs["loss_jepa"] = float(l_jepa.detach())

    logs["loss"] = float(loss.detach())
    return loss, logs
