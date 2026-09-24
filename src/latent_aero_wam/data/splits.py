"""Deterministic windowing and immutable split manifests.

A *window* is identified by its centre index ``t`` in a flight. It spans raw
indices ``[t - K, t + H]``:

    history      interactions i_tau for tau in [t-K, t-1]   (uses s up to t)
    current      s_t
    future       actions a_{t:t+H-1}, states s_{t+1:t+H}

so a window never reads a state past ``t + H`` and the history never reads past
``t``. Two blocks drawn from the same flight are separated by more than
``K + H`` centres, which makes their raw index ranges disjoint -- this is what
the leakage test checks.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .parser import Flight, discover_flights, file_sha256, load_flight

SPLIT_NAMES = ("d0_smoke", "d1_train", "d1_val", "d2_static_ood", "d3_changing_ood")


@dataclass(frozen=True)
class Block:
    """A contiguous range of valid window centres inside one flight."""

    file: str
    condition: str
    trajectory: str
    sha256: str
    start: int  # first valid centre index (inclusive)
    stop: int  # last valid centre index (exclusive)

    @property
    def n_windows(self) -> int:
        return max(0, self.stop - self.start)


def valid_centre_range(n_steps: int, history_steps: int, horizon_steps: int) -> tuple[int, int]:
    """Return ``[first, last_exclusive)`` centre indices for a flight."""
    first = history_steps
    last = n_steps - horizon_steps  # centre t needs s_{t+H}, so t <= T-1-H
    return first, max(first, last)


def raw_index_span(block: Block, history_steps: int, horizon_steps: int) -> tuple[int, int]:
    """Raw sample indices touched by every window in ``block``."""
    return block.start - history_steps, block.stop - 1 + horizon_steps


def blocks_overlap(a: Block, b: Block, history_steps: int, horizon_steps: int) -> bool:
    if a.file != b.file:
        return False
    a0, a1 = raw_index_span(a, history_steps, horizon_steps)
    b0, b1 = raw_index_span(b, history_steps, horizon_steps)
    return a0 <= b1 and b0 <= a1


def build_manifest(cfg: dict) -> dict:
    """Construct the split manifest from a resolved data config."""
    from ..utils.config import resolve_path

    data_root = resolve_path(cfg["data_root"])
    K = int(round(cfg["history_seconds"] * cfg["sample_rate_hz"]))
    H = int(round(cfg["horizon_seconds"] * cfg["sample_rate_hz"]))

    pool_paths = discover_flights(data_root, cfg["development_pool_pattern"])
    ood_pattern = cfg.get("changing_ood_pattern")
    ood_paths = discover_flights(data_root, ood_pattern) if ood_pattern else []
    if not pool_paths:
        raise FileNotFoundError(f"no development flights under {data_root}")
    if ood_pattern and not ood_paths:
        raise FileNotFoundError(f"no changing-wind flights under {data_root}")

    decimate = int(cfg.get("decimate", 1))
    rate_mode = cfg.get("transfer_rate_mode", "forward_legacy")
    flights = {p.stem: load_flight(p, decimate=decimate, transfer_rate_mode=rate_mode)
               for p in pool_paths + ood_paths}
    for f in flights.values():
        expected_dt = 1.0 / cfg["sample_rate_hz"]
        if abs(f.dt - expected_dt) > 1e-4:
            raise ValueError(f"{f.name}: dt={f.dt:.5f}s but config says {expected_dt:.5f}s")

    holdout = set(cfg["static_ood_conditions"])
    gap = K + H + 1  # strictly more than K+H centres between blocks
    val_fraction = float(cfg["val_fraction"])

    splits: dict[str, list[Block]] = {name: [] for name in SPLIT_NAMES}

    def mk(f: Flight, start: int, stop: int) -> Block:
        return Block(
            file=f.name,
            condition=f.condition,
            trajectory=f.trajectory,
            sha256=file_sha256(f.path),
            start=int(start),
            stop=int(stop),
        )

    for path in pool_paths:
        f = flights[path.stem]
        lo, hi = valid_centre_range(f.n_steps, K, H)
        if f.condition in holdout:
            splits["d2_static_ood"].append(mk(f, lo, hi))
            continue
        # Blocked development split: an early train block and a late val block,
        # separated by a gap so no raw sample is shared.
        n = hi - lo
        n_val = int(round(val_fraction * n))
        val_start = hi - n_val
        train_stop = val_start - gap
        if train_stop <= lo:
            raise ValueError(f"{f.name}: too short for a blocked split at K={K}, H={H}")
        splits["d1_train"].append(mk(f, lo, train_stop))
        splits["d1_val"].append(mk(f, val_start, hi))

    for path in ood_paths:
        f = flights[path.stem]
        lo, hi = valid_centre_range(f.n_steps, K, H)
        splits["d3_changing_ood"].append(mk(f, lo, hi))

    # D0: a short slice of the first training block, for smoke tests only.
    first = splits["d1_train"][0]
    n_smoke = int(cfg.get("smoke_windows", 256))
    splits["d0_smoke"] = [Block(**{**asdict(first), "stop": min(first.stop, first.start + n_smoke)})]

    if not splits["d2_static_ood"]:
        raise ValueError(f"static_ood_conditions {sorted(holdout)} matched no flight")

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_root": str(data_root),
        "sample_rate_hz": cfg["sample_rate_hz"],
        "dt": 1.0 / cfg["sample_rate_hz"],
        "history_seconds": cfg["history_seconds"],
        "horizon_seconds": cfg["horizon_seconds"],
        "history_steps": K,
        "horizon_steps": H,
        "decimate": decimate,
        "static_ood_conditions": sorted(holdout),
        "splits": {k: [asdict(b) for b in v] for k, v in splits.items()},
    }
    if "transfer_rate_mode" in cfg:
        manifest["transfer_rate_mode"] = rate_mode
    manifest["checksum"] = manifest_checksum(manifest)
    return manifest


def manifest_checksum(manifest: dict) -> str:
    """Hash of everything that *defines* the split.

    ``data_root`` and the timestamp are machine/run specific and excluded; the
    per-file sha256 in each block already pins the data content itself.
    """
    payload = {k: v for k, v in manifest.items() if k not in ("created_utc", "checksum", "data_root")}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def save_manifest(manifest: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


def load_manifest(path: str | Path) -> dict:
    manifest = json.loads(Path(path).read_text())
    actual = manifest_checksum(manifest)
    if actual != manifest.get("checksum"):
        raise ValueError(
            f"split manifest {path} has been modified: checksum {actual} != {manifest.get('checksum')}"
        )
    return manifest


def manifest_blocks(manifest: dict, split: str) -> list[Block]:
    return [Block(**b) for b in manifest["splits"][split]]
