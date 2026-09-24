"""Locked five-dataset factorial analysis; never pool windows as replicates."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import t


def interval(values):
    x = np.asarray(values, dtype=float)
    assert x.shape == (5,) and np.isfinite(x).all()
    half = t.ppf(0.975, 4) * x.std(ddof=1) / np.sqrt(5)
    return dict(
        dataset_differences=x.tolist(),
        mean=float(x.mean()),
        descriptive_t95ci=[float(x.mean() - half), float(x.mean() + half)],
        negative_datasets=int((x < 0).sum()),
        positive_datasets=int((x > 0).sum()),
    )


def factorial(records, metric):
    summary = {}
    for wind in [4.9, 6.1, 8.5]:
        means = {}
        for coverage in ["log", "mixed"]:
            for arm in ["frozen", "updated"]:
                ds = []
                for d in range(5):
                    seeds = []
                    for seed in [70, 71, 72]:
                        selected = [
                            x
                            for x in records
                            if (x["wind"], x["dataset"], x["coverage"], x["arm"], x["seed"])
                            == (wind, d, coverage, arm, seed)
                        ]
                        assert len(selected) == 10 and sorted(
                            x["episode"] for x in selected
                        ) == list(range(10))
                        seeds.append(np.mean([x[metric] for x in selected]))
                    ds.append(np.mean(seeds))
                means[coverage + "/" + arm] = np.asarray(ds)
        gaps = {c: means[c + "/updated"] - means[c + "/frozen"] for c in ["log", "mixed"]}
        summary[str(wind)] = dict(
            arm_dataset_scores={k: v.tolist() for k, v in means.items()},
            arm_means={k: float(v.mean()) for k, v in means.items()},
            contrasts={
                **{c + "_updated_minus_frozen": interval(v) for c, v in gaps.items()},
                "coverage_by_memory_interaction": interval(gaps["mixed"] - gaps["log"]),
                **{
                    a + "_mixed_minus_log": interval(means["mixed/" + a] - means["log/" + a])
                    for a in ["frozen", "updated"]
                },
            },
        )
    return summary


def control_record(r):
    """Use the protocol's failure-aware score, never a similarly named raw RMSE."""
    row = r["row"]
    return dict(
        wind=row["wind"],
        dataset=row["dataset_replicate"],
        coverage=row["coverage"],
        arm=row["arm"],
        seed=row["model_seed"],
        episode=row["episode"],
        rmse_m=r["capped_tracking_rmse_m"],
        failed=r["failed"],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", choices=["query", "control"], required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--acceptance", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    acceptance = json.loads(args.acceptance.read_text())
    assert not acceptance["pilot"]
    expected = 30 if args.panel == "query" else 1800
    assert acceptance.get("accepted_anchors", acceptance.get("accepted_episodes")) == expected
    accepted = {x["index"]: x for x in acceptance.get("records", acceptance.get("files", []))}
    assert set(accepted) == set(range(expected))
    records = []
    nonfinite = []
    for i in range(expected):
        path = args.root / (("anchor_" if args.panel == "query" else "episode_") + f"{i:04d}.json")
        r = json.loads(path.read_text())
        assert not r["pilot"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == accepted[i]["sha256"]
        row = r["row"]
        if args.panel == "control":
            records.append(control_record(r))
        else:
            for x in r["anchor"]["models"]:
                if x["status"] != "complete":
                    nonfinite.append(dict(anchor=i, model=x))
                    continue
                records.append(
                    dict(
                        wind=row["wind"],
                        dataset=x["dataset_replicate"],
                        coverage=x["coverage"],
                        arm=x["arm"],
                        seed=x["model_seed"],
                        episode=row["episode"],
                        physical_solution_cost=x["physical_solution_cost"],
                        query_E=x["horizon_metrics"]["50"]["original_E"]["prefix_mean"],
                        position_m=x["horizon_metrics"]["50"]["physical_position_l2_m"][
                            "prefix_mean"
                        ],
                    )
                )
    if nonfinite:
        args.output.write_text(
            json.dumps(
                dict(status="incomplete_finite_factorial", nonfinite=nonfinite, records=records),
                indent=2,
            )
            + "\n"
        )
        raise ValueError(
            "Nonfinite queries retained; do not report a finite-only factorial interaction"
        )
    metrics = (
        ["rmse_m"]
        if args.panel == "control"
        else ["physical_solution_cost", "query_E", "position_m"]
    )
    result = dict(
        panel=args.panel,
        accepted_records=len(records),
        acceptance_sha256=hashlib.sha256(args.acceptance.read_bytes()).hexdigest(),
        summary={m: factorial(records, m) for m in metrics},
        records=records,
        caveat="Result-informed fixed-roster follow-up; n=5 training datasets, descriptive nonsimultaneous t intervals, no p-values. Interaction is mixed-minus-log change in Updated-minus-Frozen gap; zero-crossing does not establish equivalence.",
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(args.panel, len(records), "records analyzed")


if __name__ == "__main__":
    main()
