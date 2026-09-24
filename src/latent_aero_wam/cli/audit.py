"""``python -m latent_aero_wam.cli.audit`` -- check the data and freeze the split.

Runs only the checks that would corrupt training if violated, then writes the
two immutable artifacts every run binds itself to: the split manifest and the
train-only normalisation statistics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..data.dataset import FlightStore, WindowDataset
from ..data.normalize import Normalizer
from ..data.parser import discover_flights, load_flight
from ..data.splits import (
    blocks_overlap,
    build_manifest,
    manifest_blocks,
    save_manifest,
)
from ..evaluation.metrics import fit_metric_scales
from ..utils.config import load_config, resolve_path
from ..utils.logging import get_logger

log = get_logger("audit")


def check_no_leakage(manifest: dict) -> None:
    K, H = manifest["history_steps"], manifest["horizon_steps"]
    names = list(manifest["splits"])
    for i, a in enumerate(names):
        for b in names[i:]:
            for ba in manifest_blocks(manifest, a):
                for bb in manifest_blocks(manifest, b):
                    if a == b and ba == bb:
                        continue
                    if a == "d0_smoke" or b == "d0_smoke":
                        continue  # D0 is a deliberate subset of D1-train
                    if blocks_overlap(ba, bb, K, H):
                        raise AssertionError(
                            f"window overlap between {a}:{ba.file}[{ba.start},{ba.stop}) "
                            f"and {b}:{bb.file}[{bb.start},{bb.stop})"
                        )


def report_flights(data_root: Path, patterns: list[str], decimate: int = 1,
                   transfer_rate_mode: str = "forward_legacy") -> list[dict]:
    rows = []
    for pattern in patterns:
        for p in discover_flights(data_root, pattern):
            f = load_flight(p, decimate=decimate, transfer_rate_mode=transfer_rate_mode)
            rows.append(
                {
                    "file": f.name,
                    "condition": f.condition,
                    "wind_mps": f.wind_mps,
                    "time_varying": f.wind_is_time_varying,
                    "n_steps": f.n_steps,
                    "duration_s": round(float(f.t[-1] - f.t[0]), 2),
                    "dt": round(f.dt, 5),
                    "v_max": round(float(np.linalg.norm(f.state[:, 0:3], axis=1).max()), 3),
                    "omega_max": round(float(np.linalg.norm(f.state[:, 9:12], axis=1).max()), 3),
                    "T_sp_range": [round(float(f.action[:, 0].min()), 3), round(float(f.action[:, 0].max()), 3)],
                }
            )
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Audit Neural-Fly data and freeze the split artifacts.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="config overrides, a.b=value")
    ap.add_argument("--force", action="store_true", help="overwrite an existing manifest")
    ap.add_argument("--report", default="reports/data_audit.json")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, args.set)
    data_root = resolve_path(cfg["data_root"])
    log.info("data root: %s", data_root)

    patterns = [p for p in (cfg["development_pool_pattern"], cfg.get("changing_ood_pattern")) if p]
    rows = report_flights(data_root, patterns, decimate=int(cfg.get("decimate", 1)),
                          transfer_rate_mode=cfg.get("transfer_rate_mode", "forward_legacy"))
    print(f"\n{'file':40s} {'wind m/s':>9s} {'N':>6s} {'dur s':>7s} {'dt':>7s} {'|v|max':>7s} {'|w|max':>7s}")
    for r in rows:
        tv = "*" if r["time_varying"] else " "
        print(
            f"{r['file']:40s} {r['wind_mps']:8.1f}{tv} {r['n_steps']:6d} "
            f"{r['duration_s']:7.2f} {r['dt']:7.4f} {r['v_max']:7.2f} {r['omega_max']:7.2f}"
        )
    print("  (* = wind varies within the flight)")

    manifest_path = resolve_path(cfg["manifest_path"])
    if manifest_path.exists() and not args.force:
        log.warning("%s already exists; not overwriting (use --force)", manifest_path)
        from ..data.splits import load_manifest

        manifest = load_manifest(manifest_path)
    else:
        manifest = build_manifest(cfg)
        check_no_leakage(manifest)
        save_manifest(manifest, manifest_path)
        log.info("wrote split manifest %s (checksum %s)", manifest_path, manifest["checksum"])

    print(f"\nsplit: K={manifest['history_steps']} steps history, H={manifest['horizon_steps']} steps horizon")
    for name, blocks in manifest["splits"].items():
        n = sum(b["stop"] - b["start"] for b in blocks)
        conds = sorted({b["condition"] for b in blocks})
        print(f"  {name:16s} {n:7d} windows  over {len(blocks)} block(s)  conditions={conds}")

    norm_path = resolve_path(cfg["normalizer_path"])
    store = FlightStore(manifest)
    train_ds = WindowDataset(manifest, "d1_train", store=store)
    normalizer = Normalizer.fit(train_ds.frames())
    normalizer.metric_scales = fit_metric_scales(train_ds)
    if norm_path.exists() and not args.force:
        existing = Normalizer.load(norm_path)
        for group in Normalizer.GROUPS:
            for attr in ("mean", "std"):
                if not np.array_equal(getattr(existing.stats[group], attr),
                                      getattr(normalizer.stats[group], attr)):
                    raise ValueError(f"normalizer differs: {norm_path}; create a new protocol version")
        if existing.metric_scales != normalizer.metric_scales:
            raise ValueError(f"metric scales differ: {norm_path}; create a new protocol version")
    else:
        normalizer.save(norm_path)
    log.info("wrote normalizer %s (fitted on d1_train only)", norm_path)

    const = {g: s.constant_channels for g, s in normalizer.stats.items() if s.constant_channels}
    if const:
        log.warning("constant channels (variance floored, kept for faithfulness): %s", const)
    print("\nmetric scales (std of the H-step change on d1_train):")
    for k, v in normalizer.metric_scales.items():
        print(f"  {k:14s} {np.round(v, 4).tolist()}")

    report = {"flights": rows, "manifest_checksum": manifest["checksum"],
              "metric_scales": normalizer.metric_scales}
    out = resolve_path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    log.info("audit report -> %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
