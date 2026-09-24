"""``python -m latent_aero_wam.cli.train`` -- config-driven training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from ..training.loop import train
from ..utils.config import load_config, resolve_path
from ..utils.logging import get_logger

log = get_logger("train")


def run_dir_for(cfg: dict, output_root: Path) -> Path:
    return output_root / cfg["experiment_name"] / f"{cfg['model']['name']}_seed{cfg['seed']}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Train one model / one seed.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--compute", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--output-root", default=None)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, args.set)
    compute = load_config(args.compute)
    if args.seed is not None:
        cfg["seed"] = args.seed

    output_root = Path(args.output_root) if args.output_root else resolve_path(
        cfg.get("output_root", "runs")
    )
    run_dir = run_dir_for(cfg, output_root)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (run_dir / "resolved_compute.yaml").write_text(yaml.safe_dump(compute, sort_keys=False))
    log.info("run dir: %s", run_dir)

    result = train(cfg, compute, run_dir, resume=not args.no_resume)
    (run_dir / "train_result.json").write_text(json.dumps(result, indent=2) + "\n")
    log.info("done: best val aggregate = %.4f", result["best_val_aggregate"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
