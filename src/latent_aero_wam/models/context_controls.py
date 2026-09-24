"""Round-9 controlled context pathways; hypotheses are in docs/ROUND9_PROTOCOL.md."""
from __future__ import annotations

import torch
from torch import nn

from .base import ENCODER_INPUT_DIM, TRANSITION_INPUT_DIM, RolloutModel
from .m2_latent_wam import LatentAeroWAM


class ContextTransform(nn.Module):
    """Non-affine transforms with explicit, epsilon-aware sufficient statistics."""

    def __init__(self, width: int, mode: str, eps: float = 1e-5):
        super().__init__()
        if mode not in ("none", "center", "rms", "ln", "ln_stats", "rms_stats"):
            raise ValueError(mode)
        self.mode, self.eps = mode, eps
        self.output_dim = width + {"ln_stats": 2, "rms_stats": 1}.get(mode, 0)

    def forward(self, z):
        if self.mode == "none":
            return z
        mean = z.mean(-1, keepdim=True)
        centered = z - mean
        if self.mode == "center":
            return centered
        if self.mode.startswith("rms"):
            scale = (z.square().mean(-1, keepdim=True) + self.eps).sqrt()
            unit = z / scale
            return torch.cat([unit, scale], -1) if self.mode == "rms_stats" else unit
        scale = (centered.square().mean(-1, keepdim=True) + self.eps).sqrt()
        unit = centered / scale
        return torch.cat([unit, mean, scale], -1) if self.mode == "ln_stats" else unit

    def reconstruct(self, encoded):
        if self.mode == "ln_stats":
            return encoded[..., :-2] * encoded[..., -1:] + encoded[..., -2:-1]
        if self.mode == "rms_stats":
            return encoded[..., :-1] * encoded[..., -1:]
        raise ValueError("reconstruction requires sufficient statistics")


class InformationControlWAM(LatentAeroWAM):
    def __init__(self, normalizer, context_mode="ln_stats", latent_dim=16,
                 transition_hidden=128, transition_layers=2, **kwargs):
        # Initialize all shared weights in the same order for seed pairing.
        super().__init__(normalizer, latent_dim=latent_dim,
                         transition_hidden=transition_hidden,
                         transition_layers=transition_layers,
                         latent_norm_mode="none", **kwargs)
        self.latent_norm = ContextTransform(latent_dim, context_mode)
        extra = self.latent_norm.output_dim - latent_dim
        if extra:
            old = self.transition[0]
            expanded = nn.Linear(old.in_features + extra, old.out_features)
            with torch.no_grad():
                expanded.weight[:, :old.in_features].copy_(old.weight)
                # Zero initial auxiliary weights retain the original output
                # at initialization; they remain fully trainable.
                expanded.weight[:, old.in_features:].zero_()
                expanded.bias.copy_(old.bias)
            self.transition[0] = expanded


class HistoryWindWAM(LatentAeroWAM):
    """Adds wind to the complete history pathway; not an oracle upper bound.

    On time-varying real logs the label is nominal, so comparisons are
    restricted to static D2 conditions in the round-9 protocol.
    """
    uses_privileged_metadata = True

    def __init__(self, normalizer, wind_scale=6.0, **kwargs):
        super().__init__(normalizer, **kwargs)
        self.wind_scale = wind_scale
        old = self.transition[0]
        expanded = nn.Linear(old.in_features + 1, old.out_features)
        with torch.no_grad():
            expanded.weight[:, :-1].copy_(old.weight)
            expanded.weight[:, -1].zero_()
            expanded.bias.copy_(old.bias)
        self.transition[0] = expanded

    def encode(self, batch):
        z, memory = super().encode(batch)
        return torch.cat([z, batch["wind_mps"].reshape(-1, 1) / self.wind_scale], -1), memory


class ContextMemoryControl(RolloutModel):
    """Identical encoder, GRUCell and head, with frozen or updated memory.

    The cell computes an output at every step in both arms; only whether its
    raw candidate becomes next-step memory differs. Normalization is applied
    at the memory-to-cell interface in both arms. This is a controlled family,
    not an exact reimplementation of either historical M1 or M2.
    """
    latent_is_context = True

    def __init__(self, normalizer, update_context=False, context_norm="none",
                 width=64, dropout=0.1, readout_norm="none"):
        super().__init__(normalizer)
        self.update_context = update_context
        self.encoder = nn.GRU(ENCODER_INPUT_DIM, width, batch_first=True)
        self.projection = nn.Linear(width, width)
        self.dropout = nn.Dropout(dropout)
        self.context_norm = ContextTransform(width, context_norm)
        if self.context_norm.output_dim != width:
            raise ValueError("memory controls require width-preserving transforms")
        self.readout_norm = ContextTransform(width, readout_norm)
        if self.readout_norm.output_dim != width:
            raise ValueError("readout controls require width-preserving transforms")
        self.decoder = nn.GRUCell(TRANSITION_INPUT_DIM, width)
        self.head = nn.Linear(width, 9)
        nn.init.zeros_(self.head.bias)
        nn.init.normal_(self.head.weight, std=1e-3)

    def encode(self, batch):
        _, h = self.encoder(self.history_features(batch))
        memory = self.projection(self.dropout(h[-1]))
        return self.context_norm(memory), memory

    def step_delta(self, feat, context, memory):
        candidate = self.decoder(feat, self.context_norm(memory))
        return self.head(self.readout_norm(candidate)), candidate if self.update_context else memory
