"""Print the run directory for a (config, seed). Used by the Slurm templates."""

from __future__ import annotations

import argparse

from ..utils.config import load_config, resolve_path
from .train import run_dir_for


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Resolve the run directory for a config and seed.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--output-root", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    cfg["seed"] = args.seed
    root = resolve_path(args.output_root or cfg.get("output_root", "runs"))
    print(run_dir_for(cfg, root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
