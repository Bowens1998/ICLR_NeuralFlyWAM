"""Evaluate a frozen checkpoint on D1/D2/D3.

Per-window, per-horizon errors are retained (not just their mean), because the
D3 story is *when* the error appears, not only how large it is. Sliding the
window with stride 1 re-estimates the latent at every 20 ms step, which is
exactly the streaming behaviour a controller would see.

Latents are exported so a disturbance-tracking analysis can be run later without
retraining.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..data.dataset import FlightStore, WindowDataset, make_loader
from ..data.normalize import Normalizer
from ..data.splits import load_manifest
from ..models import build_model
from ..training.checkpoint import load_checkpoint
from ..training.loop import move_batch
from ..utils.config import resolve_path
from ..utils.logging import get_logger
from .metrics import COMPONENTS, MetricScales, aggregate, rollout_errors, true_R_from_state

log = get_logger(__name__)


@dataclass
class SplitResult:
    split: str
    n_windows: int
    horizon_seconds: np.ndarray  # (H,)
    per_horizon: dict[str, np.ndarray]  # component -> (H,)
    scalar: dict[str, float]
    per_condition: dict[str, dict[str, float]]
    per_window: dict[str, np.ndarray]  # arrays for the time-aligned plots

    def summary(self) -> dict:
        return {
            "split": self.split,
            "n_windows": self.n_windows,
            "per_horizon": {k: v.tolist() for k, v in self.per_horizon.items()},
            "horizon_seconds": self.horizon_seconds.tolist(),
            "scalar": self.scalar,
            "per_condition": self.per_condition,
        }


def load_model_from_checkpoint(path: str | Path, device) -> tuple[torch.nn.Module, dict, dict]:
    state = load_checkpoint(path, map_location=device)
    cfg = state["config"]
    manifest = load_manifest(resolve_path(cfg["data"]["manifest_path"]))
    normalizer = Normalizer.load(resolve_path(cfg["data"]["normalizer_path"]))
    expected_norm = state["provenance"].get("normalizer_sha256")
    if expected_norm and expected_norm != hashlib.sha256(
        resolve_path(cfg["data"]["normalizer_path"]).read_bytes()
    ).hexdigest():
        raise ValueError("normalizer changed since training")
    if manifest["checksum"] != state["provenance"]["split_checksum"]:
        raise ValueError(
            "split manifest changed since training: "
            f"{manifest['checksum']} != {state['provenance']['split_checksum']}"
        )
    model = build_model(cfg["model"], normalizer, float(manifest["dt"])).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, state, {"manifest": manifest, "normalizer": normalizer}


@torch.no_grad()
def evaluate(
    checkpoint: str | Path,
    split: str,
    device: torch.device,
    batch_size: int = 256,
    stride: int = 1,
    amp: str = "off",
    export_latents: bool = True,
    wind_override: np.ndarray | None = None,
    max_windows: int | None = None,
) -> SplitResult:
    model, state, art = load_model_from_checkpoint(checkpoint, device)
    manifest, normalizer = art["manifest"], art["normalizer"]
    dt = float(manifest["dt"])
    H = int(manifest["horizon_steps"])
    scales = MetricScales(normalizer.metric_scales, device=device)
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "off": None}[amp]
    if device.type == "cpu":
        amp_dtype = None

    store = FlightStore(manifest)
    ds = WindowDataset(manifest, split, normalizer, store, stride=stride)
    if max_windows is not None:
        if max_windows <= 0:
            raise ValueError("max_windows must be positive")
        ds.index = ds.index[:max_windows]
    if wind_override is not None:
        wind_override = np.asarray(wind_override, dtype=np.float32)
        if wind_override.shape != (len(ds),) or not np.isfinite(wind_override).all():
            raise ValueError("wind_override must be finite and match canonical window order")
        if not model.uses_privileged_metadata:
            raise ValueError("wind intervention requires a privileged metadata model")
    if len(ds) == 0:
        raise ValueError(f"split '{split}' contains no windows")
    loader = make_loader(ds, batch_size, False, 0, pin_memory=device.type == "cuda")

    names = ds.flight_names
    cond_of_flight = {b.file: b.condition for b in ds.blocks}

    err_chunks: dict[str, list[np.ndarray]] = {c: [] for c in COMPONENTS}
    agg_chunks, flight_ids, centres, winds, latents = [], [], [], [], []

    torch.cuda.synchronize() if device.type == "cuda" else None
    t_start = time.perf_counter()
    n_seen = 0
    for batch in loader:
        batch = move_batch(batch, device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            model_batch = batch
            if wind_override is not None:
                model_batch = dict(batch)
                model_batch["wind_mps"] = torch.as_tensor(
                    wind_override[n_seen:n_seen + len(batch["wind_mps"])], device=device
                )
            out = model(model_batch)
        true_R = true_R_from_state(batch["future_state"])
        errs = rollout_errors(
            out.state.float(), out.R.float(), batch["future_state"], true_R, scales, dt
        )
        for c in COMPONENTS:
            err_chunks[c].append(errs[c].cpu().numpy())
        agg_chunks.append(aggregate(errs).cpu().numpy())
        flight_ids.append(batch["flight_id"].cpu().numpy())
        centres.append(batch["centre_index"].cpu().numpy())
        winds.append(batch["wind_mps"].cpu().numpy())
        if export_latents and out.latent is not None:
            latents.append(out.latent.float().cpu().numpy())
        n_seen += out.state.shape[0]
    torch.cuda.synchronize() if device.type == "cuda" else None
    elapsed = time.perf_counter() - t_start

    err = {c: np.concatenate(v, 0) for c, v in err_chunks.items()}  # (N, H)
    err["aggregate"] = np.concatenate(agg_chunks, 0)
    flight_ids = np.concatenate(flight_ids)
    centres = np.concatenate(centres)
    winds = np.concatenate(winds)

    per_horizon = {c: err[c].mean(0) for c in err}
    scalar = {f"{c}": float(err[c].mean()) for c in err}
    scalar.update({f"{c}_terminal": float(err[c][:, -1].mean()) for c in err})
    scalar["n_parameters"] = float(model.n_parameters())
    scalar["latency_ms_per_window"] = 1e3 * elapsed / max(n_seen, 1)
    if device.type == "cuda":
        scalar["peak_gpu_mb"] = torch.cuda.max_memory_allocated() / 1e6

    per_condition: dict[str, dict[str, float]] = {}
    for fid, name in enumerate(names):
        m = flight_ids == fid
        if not m.any():
            continue
        cond = cond_of_flight[name]
        per_condition[name] = {
            "condition": cond,
            "n_windows": int(m.sum()),
            **{c: float(err[c][m].mean()) for c in err},
            **{f"{c}_terminal": float(err[c][m][:, -1].mean()) for c in err},
        }

    per_window = {
        "flight_ids": flight_ids,
        "centre_index": centres,
        "wind_mps": winds,
        "flight_names": np.array(names),
        **{f"err_{c}": err[c] for c in err},
    }
    if wind_override is not None:
        per_window["applied_wind_mps"] = wind_override.copy()
    if latents:
        per_window["latent"] = np.concatenate(latents, 0)

    return SplitResult(
        split=split,
        n_windows=n_seen,
        horizon_seconds=(np.arange(1, H + 1) * dt),
        per_horizon=per_horizon,
        scalar=scalar,
        per_condition=per_condition,
        per_window=per_window,
    )


def save_result(result: SplitResult, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{result.split}_summary.json").write_text(json.dumps(result.summary(), indent=2) + "\n")
    np.savez_compressed(out_dir / f"{result.split}_per_window.npz", **result.per_window)
