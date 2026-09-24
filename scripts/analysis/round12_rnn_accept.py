"""Accept the exact Round12 RNN roster before analysis or pilot budget approval.

Example: python scripts/analysis/round12_rnn_accept.py --runs-root runs/results
  --source <40-character-immutable-commit> --stage pilot --output acceptance.json

Only known local/compute environment roots are normalized. Scientific config fields,
raw saved-config hashes, data identities and every accepted output remain bound.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from latent_aero_wam.data.normalize import Normalizer  # noqa: E402
from latent_aero_wam.data.parser import WIND_MPS  # noqa: E402
from latent_aero_wam.data.splits import load_manifest  # noqa: E402
from latent_aero_wam.models import build_model  # noqa: E402
from latent_aero_wam.utils.config import config_hash, load_config  # noqa: E402

ARMS = ("frozen_r0", "frozen_r1", "updated_r0", "updated_r1")
SPLITS = ("d1_val", "d2_static_ood", "d3_changing_ood")
METRICS = ("velocity", "orientation", "angular_rate", "displacement", "aggregate")
REMOTE_PROJECT = Path(".")
REMOTE_OUTPUT = Path("runs")
BASE_FILES = ("resolved_config.yaml", "resolved_compute.yaml", "provenance.json",
              "train_result.json", "metrics.jsonl", "checkpoint_best.pt")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def finite_tree(value, label):
    if isinstance(value, dict):
        for key, child in value.items():
            finite_tree(child, f"{label}.{key}")
    elif isinstance(value, list | tuple):
        for index, child in enumerate(value):
            finite_tree(child, f"{label}[{index}]")
    elif isinstance(value, int | float | np.number):
        require(np.isfinite(value), f"nonfinite {label}")
    elif isinstance(value, torch.Tensor):
        require(bool(torch.isfinite(value).all()), f"nonfinite tensor {label}")


def close(actual, expected, label, rtol=1e-5, atol=1e-7):
    require(np.shape(actual) == np.shape(expected), f"shape mismatch: {label}")
    require(np.allclose(actual, expected, rtol=rtol, atol=atol), f"disagreement: {label}")


def local_artifact(value, repo=REPO):
    """Map only the known checkout/release prefix to a repository-relative path."""
    path = Path(value)
    require(".." not in path.parts, "parent traversal in artifact path")
    if not path.is_absolute():
        return path.as_posix()
    for root in (repo.resolve(), REMOTE_PROJECT):
        try:
            relative = path.relative_to(root)
        except ValueError:
            continue
        if root == REMOTE_PROJECT and relative.parts[:1] == ("releases",):
            require(len(relative.parts) >= 3, "incomplete compute release path")
            relative = Path(*relative.parts[2:])
        return relative.as_posix()
    raise ValueError(f"unrecognized artifact root: {value}")


def normalize_config(cfg, runs_root, repo=REPO):
    normalized = copy.deepcopy(cfg)
    value = Path(normalized["output_root"])
    allowed = {REMOTE_OUTPUT, Path("runs"), Path("runs/results"),
               repo.resolve() / "runs", repo.resolve() / "runs/results",
               Path(runs_root), Path(runs_root).resolve()}
    require(value in allowed, f"unrecognized output_root: {value}")
    normalized["output_root"] = "<known-output-root>"
    for field in ("manifest_path", "normalizer_path"):
        normalized["data"][field] = local_artifact(normalized["data"][field], repo)
    return normalized


def roster(stage, repo=REPO):
    require(stage in ("pilot", "formal"), "unknown stage")
    tasks = []
    for dataset in ((0, 4) if stage == "pilot" else range(5)):
        suffix = "_pilot" if stage == "pilot" else ""
        plan_path = f"configs/evidence/round12rnn_data{dataset}{suffix}.yaml"
        plan = load_config(repo / plan_path)
        seeds = [79] if stage == "pilot" else [80, 81, 82]
        splits = ["d1_val"] if stage == "pilot" else list(SPLITS)
        require(plan["seeds"] == seeds and plan["eval_splits"] == splits, "changed plan roster")
        expected_names = [f"r12rnn_d{dataset}_{arm}" for arm in ARMS]
        require([e["key"] for e in plan["models"]] == expected_names, "changed arm roster")
        for arm, entry in zip(ARMS, plan["models"], strict=True):
            for seed in seeds:
                cfg = load_config(repo / entry["config"])
                cfg["seed"] = seed
                require(cfg["experiment_name"] == plan["experiment_name"], "experiment mismatch")
                require(cfg["model"]["name"] == entry["key"], "model name mismatch")
                require(cfg["model"] == dict(name=entry["key"], family="context_rnn_control",
                        update_context=arm.startswith("updated"), context_norm="ln" if arm.endswith("1")
                        else "none", readout_norm="none", width=64, dropout=0.1), "changed model design")
                require(cfg["optim"]["max_epochs"] == (3 if stage == "pilot" else 120), "epoch cap")
                require(cfg["data"]["manifest_path"] == f"artifacts/round11b_v1/data{dataset}_split.json",
                        "wrong dataset manifest")
                require(cfg["data"]["normalizer_path"] == f"artifacts/round11b_v1/data{dataset}_norm.json",
                        "wrong normalizer")
                tasks.append(dict(dataset=dataset, arm=arm, seed=seed, config=cfg,
                    config_path=entry["config"], plan_path=plan_path, compute_path=plan["compute_config"],
                    splits=splits, run=f"{cfg['experiment_name']}/{entry['key']}_seed{seed}"))
    return tasks


def expected_identity(manifest, split):
    blocks = manifest["splits"][split]
    names = list(dict.fromkeys(b["file"] for b in blocks))
    ids = {name: i for i, name in enumerate(names)}
    return dict(
        flight_names=np.asarray(names),
        flight_ids=np.concatenate([np.full(b["stop"] - b["start"], ids[b["file"]], dtype=np.int64)
                                   for b in blocks]),
        centre_index=np.concatenate([np.arange(b["start"], b["stop"]) for b in blocks]),
        wind_mps=np.concatenate([np.full(b["stop"] - b["start"], WIND_MPS[b["condition"]],
                                        dtype=np.float32) for b in blocks]),
    )


def validate_npz(path, summary, manifest, split):
    identity = expected_identity(manifest, split)
    n, h = len(identity["flight_ids"]), int(manifest["horizon_steps"])
    require(summary["split"] == split and summary["n_windows"] == n, "split/window count mismatch")
    finite_tree(summary, "summary")
    close(summary["horizon_seconds"], np.arange(1, h + 1) * manifest["dt"], "horizon seconds")
    require(set(summary["per_horizon"]) == set(METRICS), "missing or extra horizon metrics")
    require(set(summary["per_condition"]) == set(identity["flight_names"]), "per-flight summary roster")
    require(summary["scalar"]["n_parameters"] == 28489, "wrong parameter count")
    latency = summary["scalar"]["latency_ms_per_window"]
    require(latency > 0, "invalid measured evaluation latency")
    allowed = {*identity, "latent", *(f"err_{m}" for m in METRICS)}
    with np.load(path, allow_pickle=False) as data:
        require(set(data.files) == allowed, "missing or unexpected NPZ arrays")
        for key in data.files:
            value = data[key]
            require(value.dtype.kind != "O", f"object array: {key}")
            if key != "flight_names":
                require(np.issubdtype(value.dtype, np.number) and np.isfinite(value).all(),
                        f"nonfinite/nonnumeric array: {key}")
                require(len(value) == n, f"window count: {key}")
        for key, expected in identity.items():
            require(np.array_equal(data[key], expected), f"canonical window identity mismatch: {key}")
        require(data["flight_ids"].dtype.kind in "iu" and data["centre_index"].dtype.kind in "iu",
                "window IDs must be integers")
        require(data["latent"].shape == (n, 64), "latent shape")
        errors = {metric: data[f"err_{metric}"] for metric in METRICS}
        for metric, values in errors.items():
            require(values.shape == (n, h) and (values >= 0).all(), f"error shape/sign: {metric}")
            close(summary["scalar"][metric], values.mean(), f"{metric} scalar")
            close(summary["scalar"][f"{metric}_terminal"], values[:, -1].mean(), f"{metric} terminal")
            close(summary["per_horizon"][metric], values.mean(0), f"{metric} per horizon")
        close(errors["aggregate"], (errors["velocity"] + errors["orientation"]
              + errors["angular_rate"]) / 3, "aggregate composition")
        conditions = {block["file"]: block["condition"] for block in manifest["splits"][split]}
        for fid, name in enumerate(identity["flight_names"]):
            mask = identity["flight_ids"] == fid
            record = summary["per_condition"][str(name)]
            require(record["condition"] == conditions[name] and record["n_windows"] == int(mask.sum()),
                    "per-flight summary identity/count")
            for metric, values in errors.items():
                close(record[metric], values[mask].mean(), f"{name}/{metric}")
                close(record[f"{metric}_terminal"], values[mask, -1].mean(), f"{name}/{metric} terminal")
    # The evaluator's measured loop excludes checkpoint loading and compression/I/O.
    return dict(split=split, n_windows=n, horizon=h, eval_loop_seconds=latency * n / 1000,
                latency_ms_per_window=latency)


def validate_training(rows, cfg, stage):
    max_epochs = cfg["optim"]["max_epochs"]
    require(0 < len(rows) <= max_epochs, "missing/excess training epochs")
    require([row["epoch"] for row in rows] == list(range(len(rows))), "duplicate/missing training epochs")
    if stage == "pilot":
        require(len(rows) == 3, "pilot must finish all three epochs")
    best, best_epoch, bad = float("inf"), None, 0
    for index, row in enumerate(rows):
        finite_tree(row, f"epoch {index}")
        required = {"epoch", "lr", "seconds", "train/loss", "train/loss_onestep", "train/loss_rollout"}
        required |= {f"val/{m}{suffix}" for m in METRICS for suffix in ("", "_terminal")}
        require(required <= row.keys(), "missing training/validation metrics")
        require(row["seconds"] > 0 and 0 < row["lr"] <= cfg["optim"]["lr"] * (1 + 1e-6),
                "invalid epoch duration/LR")
        require(all(row[k] >= 0 for k in required if k.startswith(("val/", "train/"))),
                "negative error/loss")
        close(row["train/loss"], row["train/loss_onestep"] + row["train/loss_rollout"], "training loss composition")
        close(row["val/aggregate"], sum(row[f"val/{m}"] for m in METRICS[:3]) / 3, "ID aggregate composition")
        if row["val/aggregate"] < best - cfg["optim"]["min_delta"]:
            best, best_epoch, bad = row["val/aggregate"], index, 0
        else:
            bad += 1
        require(bad < cfg["optim"]["early_stopping_patience"] or index == len(rows) - 1,
                "training continued after early-stopping criterion")
    require(len(rows) == max_epochs or bad >= cfg["optim"]["early_stopping_patience"],
            "training truncated before epoch cap/early stopping")
    return dict(epochs=len(rows), best_epoch=best_epoch, best_val_aggregate=best,
                bad_epochs=bad, first_epoch_seconds=rows[0]["seconds"],
                max_steady_epoch_seconds=max(row["seconds"] for row in rows[1:]),
                logged_training_seconds=sum(row["seconds"] for row in rows))


def validate_checkpoint(path, saved_cfg, compute, provenance, training, rows, normalizer, manifest, last=False):
    state = torch.load(path, map_location="cpu", weights_only=True)
    require(set(state) == {"model", "optimizer", "scheduler", "scaler", "epoch", "best",
                           "bad_epochs", "config", "compute", "provenance", "val_metrics", "rng"},
            "checkpoint schema incomplete or unexpected")
    require(state["config"] == saved_cfg and state["compute"] == compute, "checkpoint config/compute mismatch")
    require(state["provenance"] == provenance, "checkpoint provenance mismatch")
    epoch = training["epochs"] - 1 if last else training["best_epoch"]
    require(state["epoch"] == epoch, "checkpoint epoch does not match ID selection")
    close(state["best"], training["best_val_aggregate"], "checkpoint selected ID score", rtol=1e-10)
    require(state["bad_epochs"] == (training["bad_epochs"] if last else 0), "checkpoint stopping counter")
    require(set(state["val_metrics"]) == {key.removeprefix("val/") for key in rows[epoch] if key.startswith("val/")},
            "checkpoint validation metric schema")
    for key, value in state["val_metrics"].items():
        close(value, rows[epoch][f"val/{key}"], f"checkpoint validation metric {key}", rtol=1e-10)
    finite_tree(state, "checkpoint")
    model = build_model(saved_cfg["model"], normalizer, 0.02)
    reference = model.state_dict()
    require(state["model"].keys() == reference.keys(), "checkpoint model key mismatch")
    require(all(state["model"][key].shape == value.shape and state["model"][key].dtype == value.dtype
                for key, value in reference.items()), "checkpoint model shape/FP32 dtype mismatch")
    model.load_state_dict(state["model"], strict=True)
    require(model.n_parameters() == 28489, "checkpoint parameter count")
    optimizer = state["optimizer"]
    require(len(optimizer["param_groups"]) == 1, "checkpoint optimizer groups")
    group = optimizer["param_groups"][0]
    require(len(group["params"]) == len(list(model.parameters()))
            and set(group["params"]) == set(optimizer["state"]), "incomplete optimizer parameter state")
    close(group["weight_decay"], saved_cfg["optim"]["weight_decay"], "optimizer decay", rtol=0, atol=0)
    close(group["lr"], rows[epoch]["lr"], "optimizer recorded LR", rtol=1e-10)
    n_train = sum(len(range(b["start"], b["stop"], saved_cfg["data"]["train_stride"]))
                  for b in manifest["splits"]["d1_train"])
    steps_per_epoch = (n_train // compute["batch_size"]) // compute["grad_accum"]
    expected_steps = (epoch + 1) * steps_per_epoch
    for parameter in optimizer["state"].values():
        require(float(parameter["step"]) == expected_steps, "optimizer step/epoch mismatch")
    total_steps = saved_cfg["optim"]["max_epochs"] * max(1, steps_per_epoch)
    require(state["scheduler"]["total_steps"] == total_steps
            and state["scheduler"]["last_epoch"] == min(expected_steps, total_steps - 1),
            "OneCycle step/epoch mismatch")
    require(state["scaler"] == {}, "FP32 checkpoint must not contain an AMP scaler state")
    for group in Normalizer.GROUPS:
        close(getattr(model, f"{group}_mean").numpy(), normalizer.stats[group].mean, f"{group} mean", rtol=0, atol=0)
        close(getattr(model, f"{group}_std").numpy(), normalizer.stats[group].std, f"{group} std", rtol=0, atol=0)


def run_file_names(splits, include_last=False):
    names = list(BASE_FILES)
    if include_last:
        names.append("checkpoint_last.pt")
    for split in splits:
        names.extend([f"eval/{split}_summary.json", f"eval/{split}_per_window.npz"])
    return names


def validate_run(task, runs_root, source, stage, manifest, repo=REPO):
    run = Path(runs_root) / task["run"]
    saved_cfg = yaml.safe_load((run / "resolved_config.yaml").read_text())
    require(normalize_config(saved_cfg, runs_root, repo) == normalize_config(task["config"], runs_root, repo),
            f"resolved scientific configuration differs: {task['run']}")
    compute = yaml.safe_load((run / "resolved_compute.yaml").read_text())
    require(compute == load_config(repo / task["compute_path"]) and compute["amp"] == "off",
            "resolved compute/FP32 profile mismatch")
    norm_path = repo / task["config"]["data"]["normalizer_path"]
    provenance = read_json(run / "provenance.json")
    expected = dict(git_commit=source, normalizer_sha256=sha256(norm_path),
                    config_hash=config_hash(saved_cfg), split_checksum=manifest["checksum"],
                    seed=task["seed"], model_name=task["config"]["model"]["name"], n_parameters=28489)
    require(provenance == expected, "source/seed/model/config/data provenance mismatch")
    result = read_json(run / "train_result.json")
    require({key: result[key] for key in expected} == expected, "training result provenance mismatch")
    relative = Path(task["run"])
    original_root = Path(saved_cfg["output_root"])
    if not original_root.is_absolute():
        original_root = repo.resolve() / original_root
    require(Path(result["run_dir"]) == original_root / relative
            or Path(result["run_dir"]) == Path(runs_root).resolve() / relative,
            "training result run directory mismatch")
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines() if line.strip()]
    training = validate_training(rows, saved_cfg, stage)
    close(result["best_val_aggregate"], training["best_val_aggregate"], "training result best ID score", rtol=1e-10)
    normalizer = Normalizer.load(norm_path)
    validate_checkpoint(run / "checkpoint_best.pt", saved_cfg, compute, provenance, training, rows, normalizer, manifest)
    has_last = (run / "checkpoint_last.pt").exists()
    if has_last:
        validate_checkpoint(run / "checkpoint_last.pt", saved_cfg, compute, provenance, training, rows, normalizer, manifest, last=True)
    require({p.name for p in (run / "eval").glob("*_per_window.npz")}
            == {f"{split}_per_window.npz" for split in task["splits"]}, "unexpected/missing evaluation split")
    evaluations = []
    for split in task["splits"]:
        evaluations.append(validate_npz(run / "eval" / f"{split}_per_window.npz",
            read_json(run / "eval" / f"{split}_summary.json"), manifest, split))
    files = [dict(path=f"{task['run']}/{name}", sha256=sha256(run / name))
             for name in run_file_names(task["splits"], has_last)]
    return dict(dataset=task["dataset"], arm=task["arm"], seed=task["seed"], run=task["run"],
                training=training, evaluations=evaluations, files=files)


def timing_report(records, manifests):
    first = max(row["training"]["first_epoch_seconds"] for row in records)
    steady = max(row["training"]["max_steady_epoch_seconds"] for row in records)
    projection = 2 * (first + 119 * steady) + 300
    rate = max(e["latency_ms_per_window"] for row in records for e in row["evaluations"]) / 1000
    formal_windows = max(sum(b["stop"] - b["start"] for split in SPLITS for b in m["splits"][split])
                         for m in manifests.values())
    eval_projection = 2 * rate * formal_windows
    return dict(max_first_epoch_seconds=first, max_steady_epoch_seconds=steady,
                formula="2*(max_first+119*max_steady)+300", training_projection_seconds=projection,
                training_gate_below_7200=projection < 7200,
                max_measured_eval_loop_seconds=max(sum(e["eval_loop_seconds"] for e in r["evaluations"])
                                                  for r in records),
                full_formal_eval_windows=formal_windows,
                formal_eval_loop_projection_seconds=eval_projection,
                combined_projection_seconds=projection + eval_projection,
                combined_gate_below_7200=projection + eval_projection < 7200,
                full_job_overhead_measured=False,
                caveat="Evaluation timing is the measured evaluator loop, excluding checkpoint load, "
                       "NPZ compression/write and scheduler overhead. Its full-roster projection uses "
                       "2x the maximum observed seconds/window; 300s remains an unmeasured overhead "
                       "reserve. Inspect job walltime before formal submission; no runtime guarantee.")


def input_paths(tasks, repo=REPO):
    paths = {"artifacts/round11b_v1/acceptance.json", "configs/sim/round11_generation.json"}

    def add_yaml(name):
        if name in paths:
            return
        paths.add(name)
        value = yaml.safe_load((repo / name).read_text())
        for parent in value.get("defaults", []):
            add_yaml(parent)

    for task in tasks:
        for name in (task["config_path"], task["compute_path"], task["plan_path"]):
            add_yaml(name)
        paths.update(task["config"]["data"][k] for k in ("manifest_path", "normalizer_path"))
    return sorted(paths)


def accept(runs_root, source, stage, repo=REPO):
    require(bool(re.fullmatch(r"[0-9a-f]{40}", source)), "source must be a full immutable commit SHA")
    tasks = roster(stage, repo)
    generation = read_json(repo / "artifacts/round11b_v1/acceptance.json")
    require(generation["accepted_flights"] == 370, "Round11B generation not accepted")
    manifests = {}
    for dataset in sorted({task["dataset"] for task in tasks}):
        m = load_manifest(repo / f"artifacts/round11b_v1/data{dataset}_split.json")
        original = generation["replicates"][dataset]
        require(m["checksum"] == original["split_checksum"], "manifest differs from accepted Round11B data")
        require(sha256(repo / f"artifacts/round11b_v1/data{dataset}_norm.json") == original["normalizer_sha256"],
                "normalizer differs from accepted Round11B data")
        require(m["dataset_replicate"] == dataset and m["horizon_steps"] == 50 and m["history_steps"] == 100
                and m["dt"] == 0.02, "data protocol mismatch")
        roster_hash = hashlib.sha256(json.dumps(
            read_json(repo / "configs/sim/round11_generation.json"), sort_keys=True).encode()).hexdigest()
        require(m["generation_roster_sha256"] == roster_hash,
                "generation roster hash mismatch")
        manifests[dataset] = m
    for experiment in {task["config"]["experiment_name"] for task in tasks}:
        expected = {Path(task["run"]).name for task in tasks if task["config"]["experiment_name"] == experiment}
        present = {p.name for p in (Path(runs_root) / experiment).iterdir() if p.is_dir() and "_seed" in p.name}
        require(present == expected, f"missing/unplanned run directories: {experiment}")
    records = [validate_run(task, runs_root, source, stage, manifests[task["dataset"]], repo) for task in tasks]
    return dict(schema="round12_rnn_acceptance_v1", accepted=True, stage=stage, source=source,
                accepted_runs=len(records), accepted_npz=sum(len(row["evaluations"]) for row in records),
                protocol="docs/ROUND12D_RNN.md", records=records,
                repository_inputs=[dict(path=p, sha256=sha256(repo / p)) for p in input_paths(tasks, repo)],
                timing=timing_report(records, manifests), primary_analysis_run=False,
                checks=["exact full roster/config/source/seed/compute/normalizer/split",
                        "ID-selected checkpoint and completed epoch/early-stopping trace",
                        "all NPZ arrays finite, canonical stride-one windows paired across all arms/seeds",
                        "full summary/scalar/terminal/per-horizon/per-flight agreement",
                        "all accepted input/output files bound by SHA256"],
                evaluation_precision_evidence="Training compute is FP32. Immutable train template specifies "
                    "evaluate --amp off; evaluator summaries do not independently record runtime AMP flags.")


def write_new(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--stage", choices=("pilot", "formal"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = accept(args.runs_root, args.source, args.stage)
    write_new(args.output, report)
    print(json.dumps({key: report[key] for key in ("stage", "accepted_runs", "accepted_npz", "timing")}))
    return 2 if args.stage == "pilot" and not report["timing"]["combined_gate_below_7200"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
