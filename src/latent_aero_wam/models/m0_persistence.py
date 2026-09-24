"""M0 -- parameter-free sanity baseline.

``kinematic``: velocity and body rate are held constant and the attitude is
integrated with the held body rate (``R <- R exp(omega dt)``).
``constant``: the state is held fixed.

M0 exists to confirm that the rollout targets and the evaluator are sane. It is
not a competing model.
"""

from __future__ import annotations

import torch

from ..data.normalize import Normalizer
from ..utils.rotation import matrix_to_rot6d, so3_exp
from .base import RolloutModel, RolloutOutput


class PersistenceModel(RolloutModel):
    def __init__(self, normalizer: Normalizer, dt: float, mode: str = "kinematic"):
        super().__init__(normalizer)
        if mode not in ("kinematic", "constant"):
            raise ValueError(f"unknown persistence mode '{mode}'")
        self.mode = mode
        self.dt = float(dt)
        # a single unused parameter keeps optimiser/checkpoint code uniform
        self._dummy = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

    def encode(self, batch: dict):
        return None, None

    def forward(self, batch: dict) -> RolloutOutput:
        v, R, omega = self.split_state(batch["current_state"])
        H = batch["future_action"].shape[1]
        states, Rs = [], []
        for _ in range(H):
            if self.mode == "kinematic":
                R = R @ so3_exp(omega * self.dt)
            states.append(torch.cat([v, matrix_to_rot6d(R), omega], dim=-1))
            Rs.append(R)
        states = torch.stack(states, 1)
        return RolloutOutput(
            state=states,
            R=torch.stack(Rs, 1),
            delta_norm=torch.zeros(
                states.shape[0], H, 9, device=states.device, dtype=states.dtype
            ),
            latent=None,
        )

    def teacher_forced_delta(self, batch: dict, context=None, memory=None) -> torch.Tensor:
        B, H = batch["future_action"].shape[:2]
        return torch.zeros(B, H, 9, device=batch["current_state"].device)
