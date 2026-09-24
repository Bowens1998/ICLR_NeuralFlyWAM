"""FP32 forward/backward compatibility and timings; no scientific model selection."""
# ruff: noqa: E402 -- standalone entry point resolves repository imports first.

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from latent_aero_wam.data.dataset import WindowDataset, make_loader
from latent_aero_wam.data.normalize import Normalizer
from latent_aero_wam.data.splits import load_manifest
from latent_aero_wam.evaluation.metrics import MetricScales
from latent_aero_wam.models import build_model
from latent_aero_wam.training.loop import move_batch
from latent_aero_wam.training.losses import compute_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(70)
    norm = Normalizer.load(ROOT / "artifacts/round11b_v1/data0_norm.json")
    m = load_manifest(ROOT / "artifacts/round11b_v1/data0_split.json")
    ds = WindowDataset(m, "d1_train", norm)
    b = next(iter(make_loader(ds, 256, False)))
    device = torch.device("cuda")
    records = []
    for arm in ["frozen", "updated"]:
        cfg = dict(
            family="context_memory_control",
            name=arm,
            update_context=arm == "updated",
            context_norm="none",
            readout_norm="none",
            width=64,
            dropout=0.1,
        )
        model = build_model(cfg, norm, 0.02).to(device)
        assert model.n_parameters() == 39497
        batch = move_batch(b, device)
        scales = MetricScales(norm.metric_scales, device=device)
        opt = torch.optim.AdamW(model.parameters(), lr=0.001)
        elapsed = []
        losses = []
        for _step in range(6):
            torch.cuda.synchronize()
            t = time.monotonic()
            opt.zero_grad()
            out = model(batch)
            loss, _ = compute_loss(model, batch, out, scales, 0.02)
            assert torch.isfinite(loss)
            loss.backward()
            assert all(
                torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
            )
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            torch.cuda.synchronize()
            elapsed.append(time.monotonic() - t)
            losses.append(float(loss.detach()))
        records.append(
            dict(
                arm=arm,
                parameters=model.n_parameters(),
                losses=losses,
                median_seconds_after_warmup=float(np.median(elapsed[1:])),
            )
        )
    result = dict(
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        capability=torch.cuda.get_device_capability(),
        amp="off",
        tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
        records=records,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        f.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
