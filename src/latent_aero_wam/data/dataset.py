"""PyTorch dataset over split manifests.

Flights are parsed once, cached as ``.npz``, and held in memory (the whole
Neural-Fly corpus is a few tens of MB). A dataset item is a dict of tensors; a
``metadata`` sub-dict carries wind labels that are used for evaluation grouping
and for the diagnostic oracle, and never by M0-M3.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler

from ..utils.config import resolve_path
from .normalize import Normalizer
from .parser import Flight, file_sha256, load_flight
from .splits import Block, manifest_blocks

_CACHE_FIELDS = ("t", "state", "v_body", "action", "delta_state", "R", "p")


class FlightStore:
    """Parses (and caches) the flights referenced by a manifest.

    The manifest's ``data_root`` is an absolute path from the machine that
    built it, so it is only a fallback: ``LATENT_WAM_DATA_ROOT`` (or, failing
    both, the repo-relative default) locates the CSVs on the current machine.
    """

    def __init__(self, manifest: dict, cache_root: str | Path | None = None):
        candidates = [
            os.environ.get("LATENT_WAM_DATA_ROOT"),
            manifest.get("data_root"),
            resolve_path("data/neural-fly/data"),
        ]
        roots = [Path(c) for c in candidates if c]
        self.data_root = next((r for r in roots if r.exists()), roots[0])
        if cache_root is None:
            cache_root = os.environ.get("LATENT_WAM_CACHE_ROOT") or resolve_path("data/cache")
        self.cache_root = Path(cache_root)
        self.decimate = int(manifest.get("decimate", 1))
        # Missing field intentionally preserves old checkpoint preprocessing.
        self.transfer_rate_mode = manifest.get("transfer_rate_mode", "forward_legacy")
        self.expected_hashes = {
            b["file"]: b["sha256"] for blocks in manifest["splits"].values() for b in blocks
        }
        self._flights: dict[str, Flight] = {}

    def get(self, name: str) -> Flight:
        if name not in self._flights:
            self._flights[name] = self._load(name)
        return self._flights[name]

    def _load(self, name: str) -> Flight:
        matches = list(self.data_root.rglob(f"{name}.csv"))
        if not matches:
            raise FileNotFoundError(f"{name}.csv not found under {self.data_root}")
        if file_sha256(matches[0]) != self.expected_hashes[name]:
            raise ValueError(f"raw file checksum changed: {name}")
        cache = self.cache_root / f"{name}_d{self.decimate}_{self.transfer_rate_mode}.npz"
        flight = load_flight(matches[0], decimate=self.decimate,
                             transfer_rate_mode=self.transfer_rate_mode)
        if not cache.exists():
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache, **{k: getattr(flight, k) for k in _CACHE_FIELDS})
        return flight


class WindowDataset(Dataset):
    """Windows of one split, drawn from the blocks in a manifest."""

    def __init__(
        self,
        manifest: dict,
        split: str,
        normalizer: Normalizer | None = None,
        store: FlightStore | None = None,
        stride: int = 1,
    ):
        self.manifest = manifest
        self.split = split
        self.K = int(manifest["history_steps"])
        self.H = int(manifest["horizon_steps"])
        self.dt = float(manifest["dt"])
        self.normalizer = normalizer
        self.store = store or FlightStore(manifest)
        self.blocks: list[Block] = manifest_blocks(manifest, split)
        self.index: list[tuple[str, int]] = [
            (b.file, t) for b in self.blocks for t in range(b.start, b.stop, max(1, stride))
        ]
        # stable, split-local integer id per flight, for per-condition grouping
        self._ids = {n: i for i, n in enumerate(dict.fromkeys(b.file for b in self.blocks))}

    def __len__(self) -> int:
        return len(self.index)

    def frames(self) -> dict[str, np.ndarray]:
        """Every raw frame touched by this split -- used to fit the normalizer."""
        acc: dict[str, list[np.ndarray]] = {g: [] for g in Normalizer.GROUPS}
        for b in self.blocks:
            f = self.store.get(b.file)
            lo, hi = b.start - self.K, b.stop - 1 + self.H
            acc["state"].append(f.state[lo : hi + 1])
            acc["v_body"].append(f.v_body[lo : hi + 1])
            acc["action"].append(f.action[lo : hi + 1])
            acc["delta"].append(f.delta_state[lo:hi])
        return {g: np.concatenate(v, axis=0) for g, v in acc.items()}

    def __getitem__(self, i: int) -> dict:
        name, t = self.index[i]
        f = self.store.get(name)
        K, H = self.K, self.H
        item = {
            "history_state": torch.from_numpy(f.state[t - K : t]),
            "history_v_body": torch.from_numpy(f.v_body[t - K : t]),
            "history_action": torch.from_numpy(f.action[t - K : t]),
            "history_delta_state": torch.from_numpy(f.delta_state[t - K : t]),
            "current_state": torch.from_numpy(f.state[t]),
            "current_v_body": torch.from_numpy(f.v_body[t]),
            "current_R": torch.from_numpy(f.R[t]),
            "future_action": torch.from_numpy(f.action[t : t + H]),
            "future_state": torch.from_numpy(f.state[t + 1 : t + H + 1]),
            "future_R": torch.from_numpy(f.R[t + 1 : t + H + 1]),
            "future_delta_state": torch.from_numpy(f.delta_state[t : t + H]),
            # metadata -- never passed to a model forward
            "wind_mps": torch.tensor(f.wind_mps, dtype=torch.float32),
            "centre_index": torch.tensor(t, dtype=torch.long),
            "flight_id": torch.tensor(self._flight_id(name), dtype=torch.long),
        }
        return item

    def _flight_id(self, name: str) -> int:
        return self._ids[name]

    @property
    def flight_names(self) -> list[str]:
        return list(self._ids)


def make_loader(
    dataset: WindowDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    num_samples: int | None = None,
) -> DataLoader:
    if num_samples is not None:
        if not shuffle or not 0 < num_samples <= len(dataset):
            raise ValueError("fixed-size sampling requires shuffled training and no replacement")
    sampler = RandomSampler(dataset, replacement=False, num_samples=num_samples) if num_samples else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
    )
