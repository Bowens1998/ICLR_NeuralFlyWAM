"""Shared rollout machinery for M0-M3.

Every model integrates the same residual state update, so the only thing that
differs between them is how the per-step increment is produced:

    v_{h+1}     = v_h + dv
    R_{h+1}     = R_h @ exp(dtheta^)      <- so(3) tangent step, never 6D addition
    omega_{h+1} = omega_h + domega

Models own their normalisation buffers, so :meth:`predict_rollout` takes and
returns physical units. That is the interface a predictive controller will call.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..data.normalize import Normalizer
from ..data.parser import ACTION_DIM, DELTA_DIM, STATE_DIM, VBODY_DIM
from ..utils.rotation import matrix_to_rot6d, rot6d_to_matrix, so3_exp

ENCODER_INPUT_DIM = STATE_DIM + VBODY_DIM + ACTION_DIM + DELTA_DIM  # 29
ENCODER_INPUT_DIM_NO_ACTION = STATE_DIM + VBODY_DIM + DELTA_DIM  # 24
TRANSITION_INPUT_DIM = STATE_DIM + VBODY_DIM + ACTION_DIM  # 20


@dataclass
class RolloutOutput:
    """Physical-unit rollout results."""

    state: torch.Tensor  # (B, H, 12)
    R: torch.Tensor  # (B, H, 3, 3)
    delta_norm: torch.Tensor  # (B, H, 9) normalised increments the model emitted
    latent: torch.Tensor | None = None  # (B, latent_dim) if the model has one
    context: torch.Tensor | None = None  # conditioning handed to the transition
    memory_init: object = None  # recurrent state at the start of the rollout


def mlp(dims: list[int], activation: type[nn.Module] = nn.SiLU) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)


class RolloutModel(nn.Module):
    """Base class holding normalisation and the shared integrator."""

    #: set True on models that consume privileged metadata (the oracle)
    uses_privileged_metadata = False

    def __init__(self, normalizer: Normalizer):
        super().__init__()
        for group in Normalizer.GROUPS:
            s = normalizer.stats[group]
            self.register_buffer(f"{group}_mean", torch.as_tensor(s.mean), persistent=True)
            self.register_buffer(f"{group}_std", torch.as_tensor(s.std), persistent=True)

    # -- normalisation helpers -------------------------------------------
    def norm(self, x: torch.Tensor, group: str) -> torch.Tensor:
        return (x - getattr(self, f"{group}_mean")) / getattr(self, f"{group}_std")

    def denorm_delta(self, d: torch.Tensor) -> torch.Tensor:
        return d * self.delta_std + self.delta_mean

    # -- feature assembly -------------------------------------------------
    def history_features(self, batch: dict, include_action: bool = True) -> torch.Tensor:
        """(B, K, 29) or (B, K, 24) normalised interaction tuples."""
        parts = [
            self.norm(batch["history_state"], "state"),
            self.norm(batch["history_v_body"], "v_body"),
        ]
        if include_action:
            parts.append(self.norm(batch["history_action"], "action"))
        parts.append(self.norm(batch["history_delta_state"], "delta"))
        return torch.cat(parts, dim=-1)

    def transition_features(
        self, v: torch.Tensor, R: torch.Tensor, omega: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """(B, 20) normalised [state, v_body, action] at one rollout step."""
        state = torch.cat([v, matrix_to_rot6d(R), omega], dim=-1)
        v_body = torch.einsum("bji,bj->bi", R, v)
        return torch.cat(
            [
                self.norm(state, "state"),
                self.norm(v_body, "v_body"),
                self.norm(action, "action"),
            ],
            dim=-1,
        )

    # -- integrator -------------------------------------------------------
    @staticmethod
    def integrate(
        v: torch.Tensor, R: torch.Tensor, omega: torch.Tensor, delta_phys: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dv, dtheta, domega = delta_phys[..., 0:3], delta_phys[..., 3:6], delta_phys[..., 6:9]
        return v + dv, R @ so3_exp(dtheta), omega + domega

    @staticmethod
    def split_state(state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        v = state[..., 0:3]
        R = rot6d_to_matrix(state[..., 3:9])
        omega = state[..., 9:12]
        return v, R, omega

    # -- API --------------------------------------------------------------
    def step_delta(self, feat: torch.Tensor, context: torch.Tensor, memory) -> tuple:
        """Emit a normalised increment for one rollout step. Implemented by subclasses.

        Returns ``(delta_norm, memory)`` where ``memory`` is whatever recurrent
        state the model carries across rollout steps (``None`` for models whose
        conditioning is frozen over the horizon).
        """
        raise NotImplementedError

    def encode(self, batch: dict) -> tuple[torch.Tensor, object]:
        """Return ``(context, initial_memory)`` from the observed history."""
        raise NotImplementedError

    def forward(self, batch: dict) -> RolloutOutput:
        context, memory = self.encode(batch)
        return self.rollout(batch["current_state"], batch["future_action"], context, memory)

    def latent_of(self, out: RolloutOutput) -> torch.Tensor | None:
        return out.latent

    def rollout(
        self,
        current_state: torch.Tensor,
        future_action: torch.Tensor,
        context: torch.Tensor,
        memory: object = None,
    ) -> RolloutOutput:
        v, R, omega = self.split_state(current_state)
        memory_init = memory
        H = future_action.shape[1]
        states, Rs, deltas = [], [], []
        for h in range(H):
            feat = self.transition_features(v, R, omega, future_action[:, h])
            d_norm, memory = self.step_delta(feat, context, memory)
            v, R, omega = self.integrate(v, R, omega, self.denorm_delta(d_norm))
            states.append(torch.cat([v, matrix_to_rot6d(R), omega], dim=-1))
            Rs.append(R)
            deltas.append(d_norm)
        latent = context if getattr(self, "latent_is_context", False) else None
        return RolloutOutput(
            state=torch.stack(states, 1),
            R=torch.stack(Rs, 1),
            delta_norm=torch.stack(deltas, 1),
            latent=latent,
            context=context,
            memory_init=memory_init,
        )

    def teacher_forced_delta(
        self, batch: dict, context: torch.Tensor | None = None, memory: object = None
    ) -> torch.Tensor:
        """(B, H, 9) normalised increments predicted from *ground-truth* states.

        Used for the one-step term of the loss; it does not accumulate error.
        Pass the ``context``/``memory`` from a forward pass to avoid re-running
        the history encoder.
        """
        if context is None:
            context, memory = self.encode(batch)
        H = batch["future_action"].shape[1]
        true_state = torch.cat([batch["current_state"].unsqueeze(1), batch["future_state"]], dim=1)
        outs = []
        for h in range(H):
            v, R, omega = self.split_state(true_state[:, h])
            feat = self.transition_features(v, R, omega, batch["future_action"][:, h])
            d_norm, memory = self.step_delta(feat, context, memory)
            outs.append(d_norm)
        return torch.stack(outs, 1)

    @torch.no_grad()
    def predict_rollout(
        self,
        history_state: torch.Tensor,
        history_action: torch.Tensor,
        current_state: torch.Tensor,
        candidate_actions: torch.Tensor,
        history_v_body: torch.Tensor | None = None,
        history_delta_state: torch.Tensor | None = None,
    ) -> RolloutOutput:
        """Stable inference API. ``candidate_actions`` is (B, H, 5), physical units.

        ``history_v_body`` and ``history_delta_state`` are deterministic functions
        of the history and are recomputed when not supplied, so a controller only
        needs to hand over states and actions.
        """
        from .derive import derive_history_features

        if history_v_body is None or history_delta_state is None:
            history_v_body, history_delta_state = derive_history_features(
                history_state, current_state
            )
        batch = {
            "history_state": history_state,
            "history_v_body": history_v_body,
            "history_action": history_action,
            "history_delta_state": history_delta_state,
            "current_state": current_state,
            "future_action": candidate_actions,
        }
        return self.forward(batch)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
