"""M2 -- Latent Aero-WAM, and M3 -- its state-only ablation.

The interaction encoder compresses the recent action-response history into a
low-dimensional ``z``. ``z`` is *frozen for the whole rollout horizon*: only the
motion state advances step to step. That two-timescale factorisation -- fast
state, slow hidden dynamics -- is the model's claim, and the shared residual
transition model ``f_phi(s_h, a_h, z)`` is what a controller re-uses to score
many candidate action sequences under one inferred ``z``.

M3 keeps everything except the action channel of the history, which isolates how
much of the OOD advantage comes from action-response *pairing* rather than from
trajectory appearance alone.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.normalize import Normalizer
from ..data.parser import DELTA_DIM
from ..utils.rotation import rot6d_to_matrix
from .base import (
    ENCODER_INPUT_DIM,
    ENCODER_INPUT_DIM_NO_ACTION,
    TRANSITION_INPUT_DIM,
    RolloutModel,
    mlp,
)


class LatentAeroWAM(RolloutModel):
    latent_is_context = True

    def __init__(
        self,
        normalizer: Normalizer,
        latent_dim: int = 16,
        encoder_hidden: int = 64,
        transition_hidden: int = 128,
        transition_layers: int = 2,
        use_action_history: bool = True,
        use_latent_norm: bool = True,
        latent_norm_mode: str | None = None,
        dropout: float = 0.0,
    ):
        super().__init__(normalizer)
        self.use_action_history = use_action_history
        self.latent_dim = latent_dim
        in_dim = ENCODER_INPUT_DIM if use_action_history else ENCODER_INPUT_DIM_NO_ACTION
        self.encoder = nn.GRU(in_dim, encoder_hidden, batch_first=True)
        self.to_latent = nn.Linear(encoder_hidden, latent_dim)
        # Rounds 4-5 mechanism tests (D13/D15). Normalization deletes the
        # latent's magnitude; the modes factor "normalization present" and
        # "magnitude channel present" into a 2x2:
        #   layernorm          norm,    no magnitude  (the original M2)
        #   none               no norm, magnitude     (round 4)
        #   magpass            norm,    magnitude     (LN(z) concat ||z||)
        #   rmsnorm            norm,    no magnitude  (direction kept, scale removed)
        #   layernorm_noaffine norm,    no magnitude  (rules out the affine params)
        if latent_norm_mode is None:
            latent_norm_mode = "layernorm" if use_latent_norm else "none"
        self.latent_norm_mode = latent_norm_mode
        self.latent_norm = {
            "layernorm": lambda: nn.LayerNorm(latent_dim),
            "none": lambda: nn.Identity(),
            "magpass": lambda: nn.LayerNorm(latent_dim),
            "rmsnorm": lambda: nn.RMSNorm(latent_dim),
            "layernorm_noaffine": lambda: nn.LayerNorm(latent_dim, elementwise_affine=False),
        }[latent_norm_mode]()
        context_dim = latent_dim + (1 if latent_norm_mode == "magpass" else 0)
        self.dropout = nn.Dropout(dropout)
        dims = [TRANSITION_INPUT_DIM + context_dim] + [transition_hidden] * transition_layers
        dims.append(DELTA_DIM)
        self.transition = mlp(dims)
        last = self.transition[-1]
        nn.init.zeros_(last.bias)
        nn.init.normal_(last.weight, std=1e-3)

    def encode(self, batch: dict):
        feats = self.history_features(batch, include_action=self.use_action_history)
        _, h = self.encoder(feats)
        pre = self.to_latent(self.dropout(h[-1]))
        z = self.latent_norm(pre)
        if self.latent_norm_mode == "magpass":
            z = torch.cat([z, pre.norm(dim=-1, keepdim=True)], dim=-1)
        return z, None

    def step_delta(self, feat: torch.Tensor, context: torch.Tensor, memory):
        return self.transition(torch.cat([feat, context], dim=-1)), memory


class JEPALatentWAM(LatentAeroWAM):
    """M2-JEPA: M2 plus a self-supervised latent-prediction objective.

    Round-1 diagnosis (`reports/GATE_B_ANALYSIS.md`): the encoder infers wind
    fine, but OOD latents land outside the training manifold and the transition
    model conditioned on them degrades. The JEPA term shapes the latent space
    directly: a predictor must map the history latent to the latent of the
    *future* window as seen by an EMA target encoder (BYOL-style stop-gradient,
    no reconstruction). The representation is trained to be predictive of
    where the dynamics are going, not merely discriminative of wind identity.
    """

    def __init__(
        self,
        normalizer: Normalizer,
        predictor_hidden: int = 64,
        ema_momentum: float = 0.996,
        **kwargs,
    ):
        super().__init__(normalizer, **kwargs)
        self.ema_momentum = float(ema_momentum)
        self.predictor = mlp([self.latent_dim, predictor_hidden, self.latent_dim])
        self.target_encoder = copy.deepcopy(self.encoder)
        self.target_to_latent = copy.deepcopy(self.to_latent)
        self.target_norm = copy.deepcopy(self.latent_norm)
        for p in self._target_parameters():
            p.requires_grad_(False)

    def _target_parameters(self):
        for m in (self.target_encoder, self.target_to_latent, self.target_norm):
            yield from m.parameters()

    def _online_parameters(self):
        for m in (self.encoder, self.to_latent, self.latent_norm):
            yield from m.parameters()

    @torch.no_grad()
    def _ema_update(self) -> None:
        m = self.ema_momentum
        for pt, po in zip(self._target_parameters(), self._online_parameters()):
            pt.mul_(m).add_(po.detach(), alpha=1.0 - m)

    def future_features(self, batch: dict) -> torch.Tensor:
        """(B, H, 29) interaction tuples of the *future* window, same layout as
        ``history_features`` so the target encoder mirrors the online one."""
        state = batch["future_state"]
        v = state[..., 0:3]
        R = rot6d_to_matrix(state[..., 3:9])
        v_body = torch.einsum("bhji,bhj->bhi", R, v)
        return torch.cat(
            [
                self.norm(state, "state"),
                self.norm(v_body, "v_body"),
                self.norm(batch["future_action"], "action"),
                self.norm(batch["future_delta_state"], "delta"),
            ],
            dim=-1,
        )

    def jepa_loss(self, batch: dict, z: torch.Tensor) -> torch.Tensor:
        if self.training:
            self._ema_update()
        with torch.no_grad():
            _, h = self.target_encoder(self.future_features(batch))
            z_tgt = self.target_norm(self.target_to_latent(h[-1]))
        return F.mse_loss(self.predictor(z), z_tgt)


class StateOnlyLatentWAM(LatentAeroWAM):
    """M3: identical to M2 with the action history removed from the encoder."""

    def __init__(self, normalizer: Normalizer, **kwargs):
        kwargs["use_action_history"] = False
        super().__init__(normalizer, **kwargs)


class OracleLatentWAM(LatentAeroWAM):
    """Diagnostic upper bound: ``z`` is an embedding of the true wind speed.

    This model reads a privileged label and is *never* a competitor to M1. It
    answers one question: if the hidden dynamics were handed to the transition
    model for free, how much of the OOD gap would close? If the oracle cannot
    close it, the gap is not an inference failure and no better encoder helps.
    """

    uses_privileged_metadata = True

    def __init__(self, normalizer: Normalizer, wind_scale: float = 6.0, **kwargs):
        super().__init__(normalizer, **kwargs)
        self.wind_scale = wind_scale
        self.wind_embed = mlp([1, 32, self.latent_dim])
        del self.encoder
        del self.to_latent

    def encode(self, batch: dict):
        if "wind_mps" not in batch:
            raise KeyError("OracleLatentWAM requires the privileged 'wind_mps' metadata")
        w = (batch["wind_mps"].reshape(-1, 1) / self.wind_scale).to(self.delta_std.dtype)
        return self.latent_norm(self.wind_embed(w)), None
