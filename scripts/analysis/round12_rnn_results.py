"""Descriptive RNN replication, gated on the SHA256-bound full formal acceptance."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.stats import t

sys.path.insert(0, str(Path(__file__).resolve().parent))
import round12_rnn_accept as acceptance  # noqa: E402

ARMS = acceptance.ARMS
CONDITIONS = ("static_30wind", "static_50wind", "static_70wind", "slow_35wind", "fast_35wind")
SEEDS = (80, 81, 82)


def describe(values):
    array = np.asarray(values, dtype=float)
    acceptance.require(array.shape == (5,) and np.isfinite(array).all(), "five finite dataset means required")
    mean = float(array.mean())
    se = float(array.std(ddof=1) / np.sqrt(5))
    interval = t.interval(0.95, 4, loc=mean, scale=se) if se else (mean, mean)
    return dict(dataset_values=array.tolist(), mean=mean, descriptive_t95ci=list(interval),
                positive_datasets=int((array > 0).sum()), negative_datasets=int((array < 0).sum()),
                zero_datasets=int((array == 0).sum()))


def summarize_replicates(scores):
    acceptance.require(set(scores) == set(ARMS), "all four fixed arms required")
    arrays = {arm: np.asarray(values, dtype=float) for arm, values in scores.items()}
    arms = {arm: describe(values) for arm, values in arrays.items()}
    frozen = arrays["frozen_r1"] - arrays["frozen_r0"]
    updated = arrays["updated_r1"] - arrays["updated_r0"]
    contrasts = {
        "frozen_recurrence_LN_effect": frozen,
        "updated_recurrence_LN_effect": updated,
        "recurrence_effect_updated_minus_frozen": updated - frozen,
        "updated_minus_frozen_r0": arrays["updated_r0"] - arrays["frozen_r0"],
        "updated_minus_frozen_r1": arrays["updated_r1"] - arrays["frozen_r1"],
    }
    return dict(arms=arms, contrasts={name: describe(value) for name, value in contrasts.items()})


def verify_acceptance(path, runs_root, source, repo=acceptance.REPO):
    acceptance.require(bool(re.fullmatch(r"[0-9a-f]{40}", source)), "source must be a full immutable commit SHA")
    report = acceptance.read_json(path)
    acceptance.require(report.get("schema") == "round12_rnn_acceptance_v1" and report.get("accepted") is True,
                       "unrecognized/unaccepted artifact report")
    acceptance.require(report["source"] == source and report["stage"] == "formal", "source/formal acceptance required")
    acceptance.require(report["accepted_runs"] == 60 and report["accepted_npz"] == 180, "incomplete formal acceptance")
    tasks = acceptance.roster("formal", repo)
    records = report["records"]
    acceptance.require(len(records) == 60 and len({r["run"] for r in records}) == 60, "duplicate/missing accepted runs")
    record_map = {r["run"]: r for r in records}
    acceptance.require(set(record_map) == {task["run"] for task in tasks}, "accepted run roster differs")
    # Verify every accepted output (including ID data/checkpoints), before reading outcomes.
    for task in tasks:
        record = record_map[task["run"]]
        for field in ("dataset", "arm", "seed"):
            acceptance.require(record[field] == task[field], f"accepted {field} mismatch")
        paths = [entry["path"] for entry in record["files"]]
        required = {f"{task['run']}/{name}" for name in acceptance.run_file_names(task["splits"])}
        last = f"{task['run']}/checkpoint_last.pt"
        acceptance.require(len(paths) == len(set(paths)) and set(paths) in (required, required | {last}),
                           "accepted file roster incomplete or duplicated")
        acceptance.require([e["split"] for e in record["evaluations"]] == list(acceptance.SPLITS),
                           "accepted evaluation roster mismatch")
        for entry in record["files"]:
            acceptance.require(acceptance.sha256(Path(runs_root) / entry["path"]) == entry["sha256"],
                               f"accepted file changed: {entry['path']}")
    inputs = report["repository_inputs"]
    expected_inputs = set(acceptance.input_paths(tasks, repo))
    acceptance.require(len(inputs) == len(expected_inputs) and {e["path"] for e in inputs} == expected_inputs,
                       "accepted repository-input roster differs")
    for entry in inputs:
        acceptance.require(acceptance.sha256(repo / entry["path"]) == entry["sha256"],
                           f"accepted repository input changed: {entry['path']}")
    return report, tasks


def aggregate_flights(records):
    """Equal weight: ten flights -> each seed -> three seeds -> each dataset."""
    indexed = {}
    for row in records:
        key = (row["dataset"], row["arm"], row["seed"], row["condition"], row["trajectory_seed"])
        acceptance.require(key not in indexed, "duplicate flight/seed/arm record")
        indexed[key] = row
    acceptance.require(len(indexed) == 5 * 4 * 3 * 5 * 10, "full 3000 flight records required")
    metrics = (*acceptance.METRICS, "aggregate_terminal")
    scores = {metric: {c: {arm: [] for arm in ARMS} for c in CONDITIONS} for metric in metrics}
    seed_means = []
    for dataset in range(5):
        expected_trajectories = set(range(110200 + dataset * 1000, 110210 + dataset * 1000))
        for condition in CONDITIONS:
            for arm in ARMS:
                per_seed = {metric: [] for metric in metrics}
                for seed in SEEDS:
                    rows = [row for key, row in indexed.items() if key[:4] == (dataset, arm, seed, condition)]
                    acceptance.require({r["trajectory_seed"] for r in rows} == expected_trajectories
                                       and len(rows) == 10, "ten matching flights required for each seed/condition")
                    means = {metric: float(np.mean([row[metric] for row in rows])) for metric in metrics}
                    acceptance.finite_tree(means, "flight means")
                    seed_means.append(dict(dataset=dataset, condition=condition, arm=arm, seed=seed, **means))
                    for metric in metrics:
                        per_seed[metric].append(means[metric])
                for metric in metrics:
                    scores[metric][condition][arm].append(float(np.mean(per_seed[metric])))
    return scores, seed_means


def analyze(runs_root, acceptance_path, source, repo=acceptance.REPO):
    accepted, tasks = verify_acceptance(acceptance_path, runs_root, source, repo)
    generation = acceptance.read_json(repo / "configs/sim/round11_generation.json")
    metadata = {row["name"]: row for row in generation["rows"]}
    records = []
    for task in tasks:
        for split in ("d2_static_ood", "d3_changing_ood"):
            path = Path(runs_root) / task["run"] / "eval" / f"{split}_per_window.npz"
            with np.load(path, allow_pickle=False) as data:
                arrays = {metric: data[f"err_{metric}"] for metric in acceptance.METRICS}
                for fid, name in enumerate(data["flight_names"]):
                    row = metadata[str(name)]
                    condition = f"{row['group']}_{row['condition']}"
                    acceptance.require(row["replicate"] == task["dataset"] and condition in CONDITIONS,
                                       "wrong dataset/condition in accepted outcome")
                    mask = data["flight_ids"] == fid
                    acceptance.require(int(mask.sum()) == 2850, "full 2850 windows per test flight required")
                    scores = {metric: float(values[mask].mean()) for metric, values in arrays.items()}
                    records.append(dict(dataset=task["dataset"], arm=task["arm"], seed=task["seed"],
                        condition=condition, flight=str(name), trajectory_seed=row["trajectory_seed"],
                        n_windows=int(mask.sum()), **scores,
                        aggregate_terminal=float(arrays["aggregate"][mask, -1].mean())))
    scores, seed_means = aggregate_flights(records)
    summaries = {metric: {c: summarize_replicates(values) for c, values in conditions.items()}
                 for metric, conditions in scores.items()}
    return dict(schema="round12_rnn_results_v1", source=source, protocol="docs/ROUND12D_RNN.md",
        acceptance_sha256=acceptance.sha256(acceptance_path), accepted_runs=accepted["accepted_runs"],
        accepted_npz=accepted["accepted_npz"], conditions=list(CONDITIONS), model_seeds=list(SEEDS),
        uncertainty_unit="independent training-data replicate (n=5), after equal averaging over 3 seeds and 10 flights",
        caveat="Descriptive t95 intervals, not simultaneous; no p-values, equivalence claim, OOD selection, "
               "or closed-loop inference. Existing simulation data and its train-only per-replicate metric "
               "scales are reused. Static/slow/fast conditions remain separate; history encoder is still a GRU.",
        contrast_direction="first named quantity minus reference; negative error difference is improvement",
        dataset_scores=scores["aggregate"], metric_dataset_scores=scores,
        summaries=summaries, seed_means=seed_means, flight_records=records)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--acceptance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = analyze(args.runs_root, args.acceptance, args.source)
    acceptance.write_new(args.output, report)
    print(json.dumps(report["summaries"]["aggregate"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
