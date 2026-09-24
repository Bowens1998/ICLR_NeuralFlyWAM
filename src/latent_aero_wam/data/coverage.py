"""Paired training-only branch dataset; original splits/normalizers stay immutable."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .dataset import WindowDataset


class ActionCoverageDataset(Dataset):
    def __init__(self, base: WindowDataset, manifest_path, regime):
        if base.split != "d1_train" or regime not in ("log", "mixed", "recorded", "dense"):
            raise ValueError("training-only coverage configuration")
        self.base = base
        self.regime = regime
        path = Path(manifest_path)
        self.manifest = json.loads(path.read_text())
        m = self.manifest
        assert (
            m["schema"] == "round15-accepted-training-v1"
            and m["original_split_checksum"] == base.manifest["checksum"]
        )
        root = Path(os.environ.get("LATENT_WAM_COVERAGE_ROOT", m["data_root"]))
        self.arrays = []
        self.index = []
        base_indices = {pair: i for i, pair in enumerate(base.index)}
        assert len(m["files"]) == 16
        assert {f["flight"] for f in m["files"]} == {b.file for b in base.blocks}
        for f in m["files"]:
            p = root / f["npz"]
            assert hashlib.sha256(p.read_bytes()).hexdigest() == f["sha256"]
            with np.load(p, allow_pickle=False) as z:
                a = {k: z[k] for k in z.files}
            n = len(self.arrays)
            self.arrays.append(a)
            assert len(a["centers"]) == 96
            for j, t in enumerate(a["centers"]):
                for r in range(8):
                    self.index.append((n, j, r, base_indices[(f["flight"], int(t))]))
        assert len(self.index) == 12288

    def __len__(self):
        return len(self.base) if self.regime == "dense" else len(self.index)

    def __getitem__(self, i):
        if self.regime == "dense":
            return self.base[i]
        n, j, r, k = self.index[i]
        if self.regime == "recorded":
            return self.base[k]
        a = self.arrays[n]
        item = dict(self.base[k])
        label = "candidate" if self.regime == "mixed" and r >= 4 else "logged"
        action = a["candidate/actions"][j, r] if label == "candidate" else a["logged/actions"][j]
        item["future_action"] = torch.from_numpy(action)
        for target, source in [
            ("future_state", "state"),
            ("future_R", "R"),
            ("future_delta_state", "delta"),
        ]:
            item[target] = torch.from_numpy(a[label + "/" + source][j, r])
        return item
