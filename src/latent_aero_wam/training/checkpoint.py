"""Checkpoint save/load with full resume state and provenance binding."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import torch


def git_commit(repo_root: Path) -> str:
    try:
        sha = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )
        dirty = subprocess.check_output(
            ["git", "-C", str(repo_root), "status", "--porcelain"], stderr=subprocess.DEVNULL
        )
        return sha.decode().strip() + ("-dirty" if dirty.strip() else "")
    except Exception:
        return "unknown"


def save_checkpoint(path: str | Path, **state: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)  # atomic, so a killed job never leaves a truncated file
    return path


def load_checkpoint(path: str | Path, map_location="cpu") -> dict:
    return torch.load(path, map_location=map_location, weights_only=False)
