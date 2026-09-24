"""Normalisation statistics, computed from the D1 training split only.

Four groups are normalised independently:

    state    (12)  network inputs
    v_body   (3)   network input feature
    action   (5)   network input
    delta    (9)   the *prediction target*; the transition head emits normalised
                   increments which are rescaled by ``delta.std`` before being
                   integrated, so all nine components enter the loss on a
                   comparable scale.

``q_sp``'s yaw channel is identically zero in every Neural-Fly log, so a
variance floor is applied and the affected channels are recorded in
``constant_channels``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

STD_FLOOR = 1e-6


@dataclass
class GroupStats:
    mean: np.ndarray
    std: np.ndarray
    constant_channels: list[int]

    def to_json(self) -> dict:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "constant_channels": self.constant_channels,
        }

    @staticmethod
    def from_json(d: dict) -> GroupStats:
        return GroupStats(
            np.asarray(d["mean"], dtype=np.float32),
            np.asarray(d["std"], dtype=np.float32),
            list(d["constant_channels"]),
        )


def _fit_group(x: np.ndarray) -> GroupStats:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    constant = np.nonzero(std < STD_FLOOR)[0].tolist()
    std = np.where(std < STD_FLOOR, 1.0, std)
    return GroupStats(mean.astype(np.float32), std.astype(np.float32), constant)


class Normalizer:
    """Container for the four groups, with torch-side apply/invert helpers.

    It also carries ``metric_scales``: the per-component standard deviation of
    the *H-step* change on the training split. Reported errors are divided by
    these, so "1.0" means "as wrong as simply not predicting the change at all"
    and the components are comparable enough to average.
    """

    GROUPS = ("state", "v_body", "action", "delta")

    def __init__(self, stats: dict[str, GroupStats], metric_scales: dict[str, list] | None = None):
        missing = [g for g in self.GROUPS if g not in stats]
        if missing:
            raise ValueError(f"normalizer missing groups {missing}")
        self.stats = stats
        self.metric_scales = metric_scales or {}
        self._torch: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    # -- fitting ----------------------------------------------------------
    @classmethod
    def fit(cls, frames: dict[str, np.ndarray]) -> Normalizer:
        return cls({g: _fit_group(frames[g]) for g in cls.GROUPS})

    # -- persistence ------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "groups": {g: s.to_json() for g, s in self.stats.items()},
            "metric_scales": self.metric_scales,
        }
        path.write_text(json.dumps(payload, indent=2) + "\n")
        return path

    @classmethod
    def load(cls, path: str | Path) -> Normalizer:
        raw = json.loads(Path(path).read_text())
        groups = raw["groups"]
        return cls(
            {g: GroupStats.from_json(v) for g, v in groups.items()},
            raw.get("metric_scales", {}),
        )

    # -- use --------------------------------------------------------------
    def _tensors(self, group: str, ref: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        key = f"{group}:{ref.device}:{ref.dtype}"
        if key not in self._torch:
            s = self.stats[group]
            self._torch[key] = (
                torch.as_tensor(s.mean, device=ref.device, dtype=ref.dtype),
                torch.as_tensor(s.std, device=ref.device, dtype=ref.dtype),
            )
        return self._torch[key]

    def encode(self, x: torch.Tensor, group: str) -> torch.Tensor:
        mean, std = self._tensors(group, x)
        return (x - mean) / std

    def decode(self, x: torch.Tensor, group: str) -> torch.Tensor:
        mean, std = self._tensors(group, x)
        return x * std + mean

    def scale(self, group: str, ref: torch.Tensor) -> torch.Tensor:
        """The std vector alone -- used to rescale predicted increments."""
        return self._tensors(group, ref)[1]

    def std_np(self, group: str) -> np.ndarray:
        return self.stats[group].std
