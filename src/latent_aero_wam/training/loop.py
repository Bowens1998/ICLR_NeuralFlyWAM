"""Config-driven training loop: AMP, seeding, early stopping, resume.

Model selection uses the D1 validation split only. D2 and D3 are never read
here -- they are evaluated once, after training, from a frozen checkpoint.
"""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

import torch

from ..data.dataset import FlightStore, WindowDataset, make_loader
from ..data.normalize import Normalizer
from ..evaluation.metrics import MetricScales, aggregate, rollout_errors, true_R_from_state
from ..models import build_model
from ..utils.config import REPO_ROOT, config_hash, resolve_path
from ..utils.logging import JsonlWriter, get_logger
from ..utils.seed import set_seed
from .checkpoint import git_commit, load_checkpoint, save_checkpoint
from .losses import compute_loss

log = get_logger(__name__)


def move_batch(batch: dict, device, dtype=torch.float32) -> dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True)
        if out[k].is_floating_point():
            out[k] = out[k].to(dtype)
    return out


@torch.no_grad()
def evaluate_split(model, loader, scales: MetricScales, dt: float, device, amp_dtype) -> dict:
    model.eval()
    totals: dict[str, float] = {}
    n = 0
    for batch in loader:
        batch = move_batch(batch, device)
        with torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None
        ):
            out = model(batch)
        true_R = true_R_from_state(batch["future_state"])
        errs = rollout_errors(
            out.state.float(), out.R.float(), batch["future_state"], true_R, scales, dt
        )
        errs["aggregate"] = aggregate(errs)
        b = out.state.shape[0]
        for k, v in errs.items():
            totals[k] = totals.get(k, 0.0) + float(v.mean()) * b
            totals[f"{k}_terminal"] = totals.get(f"{k}_terminal", 0.0) + float(v[:, -1].mean()) * b
        n += b
    model.train()
    return {k: v / max(n, 1) for k, v in totals.items()}


def train(cfg: dict, compute: dict, run_dir: Path, resume: bool = True) -> dict:
    device = torch.device(
        compute["device"] if torch.cuda.is_available() or compute["device"] == "cpu" else "cpu"
    )
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "off": None}[compute["amp"]]
    if device.type == "cpu":
        amp_dtype = None

    seed = int(cfg["seed"])
    set_seed(seed, deterministic=compute.get("deterministic", False))

    from ..data.splits import load_manifest

    manifest = load_manifest(resolve_path(cfg["data"]["manifest_path"]))
    normalizer = Normalizer.load(resolve_path(cfg["data"]["normalizer_path"]))
    dt = float(manifest["dt"])
    store = FlightStore(manifest)

    train_ds = WindowDataset(
        manifest, "d1_train", normalizer, store, stride=cfg["data"]["train_stride"]
    )
    val_ds = WindowDataset(manifest, "d1_val", normalizer, store, stride=cfg["data"]["eval_stride"])
    if cfg["data"].get("smoke", False):
        train_ds = WindowDataset(manifest, "d0_smoke", normalizer, store)
        val_ds = WindowDataset(manifest, "d0_smoke", normalizer, store)

    coverage_path = cfg["data"].get("coverage_manifest")
    if coverage_path:
        if cfg["data"].get("smoke", False):
            raise ValueError("coverage study cannot use legacy smoke split")
        from ..data.coverage import ActionCoverageDataset

        train_ds = ActionCoverageDataset(
            train_ds, resolve_path(coverage_path), cfg["data"]["coverage_regime"]
        )
        for key, config_key in [
            ("original_split_sha256", "manifest_path"),
            ("normalizer_sha256", "normalizer_path"),
        ]:
            assert (
                train_ds.manifest[key]
                == hashlib.sha256(resolve_path(cfg["data"][config_key]).read_bytes()).hexdigest()
            )

    train_loader = make_loader(
        train_ds,
        compute["batch_size"],
        True,
        compute["num_workers"],
        pin_memory=device.type == "cuda",
        drop_last=True,
        num_samples=cfg["data"].get("training_samples_per_epoch"),
    )
    val_loader = make_loader(
        val_ds,
        compute["eval_batch_size"],
        False,
        compute["num_workers"],
        pin_memory=device.type == "cuda",
    )

    model = build_model(cfg["model"], normalizer, dt).to(device)
    scales = MetricScales(normalizer.metric_scales, device=device)
    log.info(
        "model=%s params=%d train_windows=%d val_windows=%d device=%s amp=%s",
        cfg["model"]["name"],
        model.n_parameters(),
        len(train_ds),
        len(val_ds),
        device,
        compute["amp"],
    )

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        # parameter-free baselines (M0): a frozen placeholder group, never stepped
        trainable or list(model.parameters()),
        lr=cfg["optim"]["lr"],
        weight_decay=cfg["optim"]["weight_decay"],
    )
    total_steps = cfg["optim"]["max_epochs"] * max(1, len(train_loader) // compute["grad_accum"])
    sched = (
        torch.optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr=cfg["optim"]["lr"],
            total_steps=max(total_steps, 1),
            pct_start=0.1,
            div_factor=10.0,
            final_div_factor=10.0,
        )
        if cfg["optim"]["schedule"] == "onecycle"
        else None
    )
    scaler = torch.amp.GradScaler(device.type, enabled=amp_dtype is torch.float16)

    ckpt_last = run_dir / "checkpoint_last.pt"
    ckpt_best = run_dir / "checkpoint_best.pt"
    metrics_log = JsonlWriter(run_dir / "metrics.jsonl")
    start_epoch, best, bad_epochs = 0, float("inf"), 0
    optimizer_steps = 0

    if resume and ckpt_last.exists():
        state = load_checkpoint(ckpt_last, map_location=device)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        if sched and state.get("scheduler"):
            sched.load_state_dict(state["scheduler"])
        if state.get("scaler"):
            scaler.load_state_dict(state["scaler"])
        # map_location may have moved the RNG byte tensors onto the GPU
        torch.set_rng_state(state["rng"]["cpu"].cpu().to(torch.uint8))
        if device.type == "cuda" and state["rng"].get("cuda") is not None:
            torch.cuda.set_rng_state_all([s.cpu().to(torch.uint8) for s in state["rng"]["cuda"]])
        start_epoch, best, bad_epochs = state["epoch"] + 1, state["best"], state["bad_epochs"]
        optimizer_steps = state.get("optimizer_steps", 0)
        log.info("resumed from %s at epoch %d (best=%.4f)", ckpt_last, start_epoch, best)

    provenance = {
        "git_commit": os.environ.get("LATENT_WAM_SOURCE_COMMIT", git_commit(REPO_ROOT)),
        "normalizer_sha256": hashlib.sha256(
            resolve_path(cfg["data"]["normalizer_path"]).read_bytes()
        ).hexdigest(),
        "config_hash": config_hash(cfg),
        "split_checksum": manifest["checksum"],
        "seed": seed,
        "model_name": cfg["model"]["name"],
        "n_parameters": model.n_parameters(),
    }
    if coverage_path:
        provenance["coverage_manifest_sha256"] = hashlib.sha256(
            resolve_path(coverage_path).read_bytes()
        ).hexdigest()
        provenance["coverage_regime"] = cfg["data"]["coverage_regime"]
        provenance["training_windows"] = len(train_ds)
        if cfg["data"].get("training_samples_per_epoch") is not None:
            provenance["training_samples_per_epoch"] = cfg["data"]["training_samples_per_epoch"]

    (run_dir / "provenance.json").write_text(__import__("json").dumps(provenance, indent=2) + "\n")

    grad_accum = compute["grad_accum"]
    patience = cfg["optim"]["early_stopping_patience"]

    for epoch in range(start_epoch, cfg["optim"]["max_epochs"]):
        t0 = time.time()
        model.train()
        run_logs: dict[str, float] = {}
        n_batches = 0
        opt.zero_grad(set_to_none=True)
        # a parameter-free model has nothing to fit and its loss carries no grad;
        # skip straight to validation + checkpointing
        for i, batch in enumerate(train_loader if trainable else ()):
            batch = move_batch(batch, device)
            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None
            ):
                out = model(batch)
                loss, logs = compute_loss(
                    model,
                    batch,
                    out,
                    scales,
                    dt,
                    w_onestep=cfg["loss"]["w_onestep"],
                    w_rollout=cfg["loss"]["w_rollout"],
                    w_jepa=cfg["loss"].get("w_jepa", 0.0),
                    horizon_weighting=cfg["loss"]["horizon_weighting"],
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite loss at epoch {epoch}, batch {i}")
            scaler.scale(loss / grad_accum).backward()
            if (i + 1) % grad_accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["optim"]["grad_clip"])
                scaler.step(opt)
                optimizer_steps += 1
                scaler.update()
                opt.zero_grad(set_to_none=True)
                if sched and sched.last_epoch < sched.total_steps - 1:
                    sched.step()
            for k, v in logs.items():
                run_logs[k] = run_logs.get(k, 0.0) + v
            n_batches += 1

        train_metrics = {k: v / max(n_batches, 1) for k, v in run_logs.items()}
        val_metrics = evaluate_split(model, val_loader, scales, dt, device, amp_dtype)
        record = {
            "epoch": epoch,
            "lr": opt.param_groups[0]["lr"],
            "seconds": round(time.time() - t0, 2),
            **{f"train/{k}": v for k, v in train_metrics.items()},
            **{f"val/{k}": v for k, v in val_metrics.items()},
        }
        if coverage_path:
            record.update(optimizer_steps=optimizer_steps, train_batches=n_batches)
        metrics_log.write(record)
        log.info(
            "epoch %3d | loss %.4f | val agg %.4f | val vel %.4f ori %.4f | %.1fs",
            epoch,
            train_metrics.get("loss", float("nan")),
            val_metrics["aggregate"],
            val_metrics["velocity"],
            val_metrics["orientation"],
            record["seconds"],
        )

        score = val_metrics["aggregate"]
        if not __import__("math").isfinite(score):
            raise FloatingPointError(f"nonfinite validation score at epoch {epoch}")
        improved = score < best - cfg["optim"]["min_delta"]
        if improved:
            best, bad_epochs = score, 0
        else:
            bad_epochs += 1

        state = {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict() if sched else None,
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best": best,
            "bad_epochs": bad_epochs,
            "config": cfg,
            "compute": compute,
            "provenance": provenance,
            "val_metrics": val_metrics,
            "rng": {
                "cpu": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
            },
        }
        if coverage_path:
            state["optimizer_steps"] = optimizer_steps
        save_checkpoint(ckpt_last, **state)
        if improved:
            save_checkpoint(ckpt_best, **state)
        if patience and bad_epochs >= patience:
            log.info("early stopping at epoch %d (no improvement for %d epochs)", epoch, bad_epochs)
            break

    return {"best_val_aggregate": best, "run_dir": str(run_dir), **provenance}
