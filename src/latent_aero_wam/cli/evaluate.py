"""``python -m latent_aero_wam.cli.evaluate`` -- evaluate a frozen checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ..evaluation.evaluator import evaluate, save_result
from ..utils.logging import get_logger

log = get_logger("evaluate")

ALL_SPLITS = ("d1_val", "d2_static_ood", "d3_changing_ood")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate a checkpoint on one or more splits.")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", nargs="+", default=list(ALL_SPLITS))
    ap.add_argument("--out-dir", default=None, help="default: <checkpoint dir>/eval")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--amp", default="off", choices=["off", "bf16", "fp16"])
    args = ap.parse_args(argv)

    ckpt = Path(args.checkpoint)
    out_dir = Path(args.out_dir) if args.out_dir else ckpt.parent / "eval"
    device = torch.device(args.device)

    for split in args.split:
        res = evaluate(ckpt, split, device, args.batch_size, args.stride, args.amp)
        save_result(res, out_dir)
        log.info(
            "%-16s n=%6d  aggregate=%.4f  terminal=%.4f  vel=%.4f  ori=%.4f  rate=%.4f",
            split, res.n_windows, res.scalar["aggregate"], res.scalar["aggregate_terminal"],
            res.scalar["velocity"], res.scalar["orientation"], res.scalar["angular_rate"],
        )
        for name, m in sorted(res.per_condition.items()):
            log.info("    %-42s cond=%-10s aggregate=%.4f", name, m["condition"], m["aggregate"])
    log.info("results -> %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
