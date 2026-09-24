"""Full acceptance gate for the locked Round18 training roster, no inference."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from latent_aero_wam.data.normalize import Normalizer
from latent_aero_wam.models import build_model
from latent_aero_wam.utils.config import config_hash, load_config


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--source", required=True)
    args = ap.parse_args()
    SOURCE = args.source
    assert len(SOURCE) == 40
    plan = load_config(args.plan)
    subpath = Path("runs/evidence") / plan["plan_name"] / "submission_manifest.json"
    sub = json.loads(subpath.read_text())
    assert sub["git_commit"] == SOURCE
    expected = [(m["config"], seed) for m in plan["models"] for seed in plan["seeds"]]
    assert [(t["config"], t["seed"]) for t in sub["tasks"]] == expected
    reference = {}
    files = {str(subpath): sha(subpath)}
    runs = []
    npzs = 0
    for task in sub["tasks"]:
        cfg = load_config(task["config"])
        cfg["seed"] = task["seed"]
        assert config_hash(cfg) == task["config_hash"]
        run = (
            Path("runs/results")
            / plan["experiment_name"]
            / f"{cfg['model']['name']}_seed{task['seed']}"
        )
        prov = json.loads((run / "provenance.json").read_text())
        assert prov["seed"] == task["seed"] and prov["model_name"] == cfg["model"]["name"]
        assert prov["git_commit"] == SOURCE and prov["split_checksum"] == sub["split_checksum"]
        assert prov["normalizer_sha256"] == sha(cfg["data"]["normalizer_path"])
        assert load_config(run / "resolved_config.yaml") == cfg
        compute = load_config(plan["compute_config"])
        assert load_config(run / "resolved_compute.yaml") == compute
        rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
        epochs = cfg["optim"]["max_epochs"]
        assert epochs in (120, 445)
        assert len(rows) == epochs and all(
            np.isfinite(v) for r in rows for v in r.values() if isinstance(v, (int, float))
        )
        assert [r["epoch"] for r in rows] == list(range(epochs))
        assert all(
            r["train_batches"] == 48 and r["optimizer_steps"] == 48 * (i + 1)
            for i, r in enumerate(rows)
        )
        assert prov["coverage_manifest_sha256"] == sha(cfg["data"]["coverage_manifest"])
        assert (
            prov["coverage_regime"] == cfg["data"]["coverage_regime"]
            and prov["training_windows"] == (45600 if cfg["data"]["coverage_regime"] == "dense" else 12288)
        )
        selected = 0
        best = float("inf")
        for i, r in enumerate(rows):
            if r["val/aggregate"] < best - cfg["optim"]["min_delta"]:
                selected = i
                best = r["val/aggregate"]
        result = json.loads((run / "train_result.json").read_text())
        assert np.isfinite(result["best_val_aggregate"])
        cp = torch.load(run / "checkpoint_best.pt", map_location="cpu", weights_only=False)
        assert cp["epoch"] == selected and np.isclose(cp["best"], best)
        assert np.isclose(result["best_val_aggregate"], best)
        assert cp["optimizer_steps"] == 48 * (selected + 1)
        assert cp["config"] == cfg and cp["compute"] == compute and cp["provenance"] == prov
        norm = Normalizer.load(cfg["data"]["normalizer_path"])
        model = build_model(cfg["model"], norm, 0.02)
        model.load_state_dict(cp["model"], strict=True)
        assert model.n_parameters() == 39497
        assert all(torch.isfinite(v).all() for v in cp["model"].values())
        for split in plan["eval_splits"]:
            p = run / "eval" / f"{split}_per_window.npz"
            sp = p.with_name(f"{split}_summary.json")
            summary = json.loads(sp.read_text())
            n = summary["n_windows"]
            h = len(summary["horizon_seconds"])
            assert h == 50
            assert summary["scalar"]["n_parameters"] == 39497
            if split == "d2_static_ood":
                assert n == 85500
            with np.load(p, allow_pickle=False) as z:
                for k in z.files:
                    a = z[k]
                    if np.issubdtype(a.dtype, np.number):
                        assert np.isfinite(a).all(), (run, split, k)
                    if k != "flight_names":
                        assert len(a) == n
                for metric in [
                    "velocity",
                    "orientation",
                    "angular_rate",
                    "displacement",
                    "aggregate",
                ]:
                    a = z["err_" + metric]
                    assert a.shape == (n, h) and (a >= 0).all()
                    assert np.isclose(a.mean(), summary["scalar"][metric], rtol=1e-5, atol=1e-7)
                    assert np.allclose(
                        a.mean(0), summary["per_horizon"][metric], rtol=1e-5, atol=1e-7
                    )
                assert np.allclose(
                    z["err_aggregate"],
                    sum(z["err_" + m] for m in ["velocity", "orientation", "angular_rate"]) / 3,
                    rtol=2e-6,
                    atol=1e-7,
                )
                ids = tuple(
                    z[k].copy() for k in ["flight_ids", "centre_index", "wind_mps", "flight_names"]
                )
                if split not in reference:
                    reference[split] = ids
                else:
                    assert all(np.array_equal(a, b) for a, b in zip(reference[split], ids))
            files[str(p)] = sha(p)
            files[str(sp)] = sha(sp)
            npzs += 1
        for name in [
            "checkpoint_best.pt",
            "provenance.json",
            "resolved_config.yaml",
            "resolved_compute.yaml",
            "train_result.json",
            "metrics.jsonl",
        ]:
            p = run / name
            files[str(p)] = sha(p)
        runs.append(
            dict(
                model=cfg["model"]["name"],
                seed=task["seed"],
                epochs=len(rows),
                parameters=model.n_parameters(),
            )
        )
    report = dict(
        source=SOURCE,
        plan=plan["plan_name"],
        accepted_runs=len(runs),
        accepted_npz=npzs,
        runs=runs,
        sha256=files,
        pilot="pilot" in plan["plan_name"],
        checks=[
            "exact source/config/compute/provenance",
            "best checkpoint strict load and finite tensors",
            "finite training",
            "all NPZ finite/shapes/nonnegative/errors/aggregate/summary/horizon",
            "paired window identities",
        ],
    )
    target = Path("runs/evidence/round18_bundle") / (plan["plan_name"] + "_acceptance.json")
    with target.open("x") as f:
        f.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "sha256"}, indent=2))


if __name__ == "__main__":
    main()
