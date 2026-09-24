"""``python -m latent_aero_wam.cli.summarize`` -- collate runs into one table.

Reads every ``eval/<split>_summary.json`` under an experiment root, writes a
tidy CSV, the primary plots, and a short Markdown results file. Seeds are
aggregated with mean and standard deviation; nothing here selects a model.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from ..evaluation.plots import plot_error_vs_horizon, plot_split_comparison, plot_time_aligned
from ..utils.logging import get_logger

log = get_logger("summarize")

PRIMARY_SPLITS = ["d1_val", "d2_static_ood", "d3_changing_ood"]


def collect_rows(root: Path) -> list[dict]:
    rows = []
    for summary in sorted(root.rglob("eval/*_summary.json")):
        run_dir = summary.parent.parent
        prov_path = run_dir / "provenance.json"
        prov = json.loads(prov_path.read_text()) if prov_path.exists() else {}
        s = json.loads(summary.read_text())
        rows.append(
            {
                "run": run_dir.name,
                "model": prov.get("model_name", run_dir.name.rsplit("_seed", 1)[0]),
                "seed": prov.get("seed", -1),
                "split": s["split"],
                "n_windows": s["n_windows"],
                "n_parameters": prov.get("n_parameters", s["scalar"].get("n_parameters")),
                "git_commit": prov.get("git_commit", ""),
                "split_checksum": prov.get("split_checksum", ""),
                **{k: v for k, v in s["scalar"].items()},
            }
        )
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    keys = sorted({k for r in rows for k in r})
    ordered = ["model", "seed", "split", "n_windows", "n_parameters", "aggregate"]
    keys = ordered + [k for k in keys if k not in ordered]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def markdown_table(rows: list[dict], splits: list[str]) -> str:
    models = sorted({r["model"] for r in rows})
    out = ["| model | params | " + " | ".join(splits) + " |",
           "|---|---|" + "---|" * len(splits)]
    for m in models:
        params = next((r["n_parameters"] for r in rows if r["model"] == m), "")
        cells = []
        for sp in splits:
            vals = [r["aggregate"] for r in rows if r["model"] == m and r["split"] == sp]
            cells.append(f"{np.mean(vals):.4f} ± {np.std(vals):.4f} (n={len(vals)})" if vals else "-")
        out.append(f"| {m} | {params} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Collate evaluation results across runs and seeds.")
    ap.add_argument("--runs-root", required=True, help="e.g. runs/main_round1")
    ap.add_argument("--out-dir", default="reports")
    ap.add_argument("--splits", nargs="+", default=PRIMARY_SPLITS)
    ap.add_argument("--time-aligned-flight", default="custom_figure8_baseline_70p20sint")
    args = ap.parse_args(argv)

    root, out_dir = Path(args.runs_root), Path(args.out_dir)
    rows = collect_rows(root)
    if not rows:
        log.error("no evaluation summaries under %s", root)
        return 1
    log.info("collected %d (run, split) results from %s", len(rows), root)

    checksums = {r["split_checksum"] for r in rows if r["split_checksum"]}
    if len(checksums) > 1:
        raise SystemExit(f"runs disagree on the split manifest: {checksums}")

    write_csv(rows, out_dir / "results.csv")
    plot_split_comparison(rows, out_dir / "split_comparison.png", args.splits)

    # error-vs-horizon, seed-averaged, one panel per split
    for split in args.splits:
        curves: dict[str, dict] = {}
        for model in sorted({r["model"] for r in rows}):
            per_h, hs = [], None
            for summary in sorted(root.rglob(f"eval/{split}_summary.json")):
                prov = summary.parent.parent / "provenance.json"
                if not prov.exists() or json.loads(prov.read_text()).get("model_name") != model:
                    continue
                s = json.loads(summary.read_text())
                per_h.append(s["per_horizon"]["aggregate"])
                hs = s["horizon_seconds"]
            if per_h:
                curves[model] = {"horizon_seconds": hs, "per_horizon": {"aggregate": np.mean(per_h, 0)}}
        if curves:
            plot_error_vs_horizon(curves, out_dir / f"error_vs_horizon_{split}.png")

    # time-aligned view on the changing-wind flight, seed 0 of each model
    per_window: dict[str, dict] = {}
    dt = 0.02
    for npz in sorted(root.rglob("eval/d3_changing_ood_per_window.npz")):
        prov = npz.parent.parent / "provenance.json"
        if not prov.exists():
            continue
        p = json.loads(prov.read_text())
        if p.get("seed", 0) != 0:
            continue
        d = dict(np.load(npz, allow_pickle=True))
        d["flight_names"] = [str(x) for x in d["flight_names"]]
        per_window[p["model_name"]] = d
    if per_window:
        plot_time_aligned(
            per_window, out_dir / "time_aligned_changing_wind.png",
            flight=args.time_aligned_flight, dt=dt,
        )

    md = [
        "# Results -- normalised aggregate rollout error (lower is better)",
        "",
        f"runs root: `{root}`  |  split checksum: `{sorted(checksums)[0] if checksums else 'n/a'}`",
        "",
        markdown_table(rows, args.splits),
        "",
        "Errors are normalised by the standard deviation of the H-step change on",
        "the D1 training split, so 1.0 is the score of predicting no change at all.",
        "D2/D3 were evaluated once from frozen checkpoints and were never used for",
        "model selection.",
    ]
    (out_dir / "RESULTS.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[4:5]))
    log.info("wrote %s, %s and plots", out_dir / "results.csv", out_dir / "RESULTS.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
