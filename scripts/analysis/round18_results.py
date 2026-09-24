"""Locked recipe/budget contrasts with five training datasets as replication unit."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from round15_results import control_record, interval

RECIPES = ["log", "mixed"]
CELLS = [f"{r}_{b}" for r in RECIPES for b in ["short", "long"]]


def factorial(records, metric):
    assert len(records) == 4800
    grouped = defaultdict(list)
    for r in records:
        grouped[(r["wind"], r["coverage"], r["arm"], r["dataset"], r["seed"])].append(r)
    assert len(grouped) == 480
    summary = {}
    for wind in [0.0, 4.9, 6.1, 8.5]:
        means = {}
        for cell in CELLS:
            for arm in ["frozen", "updated"]:
                ds = []
                for d in range(5):
                    ss = []
                    for seed in [70, 71, 72]:
                        rr = grouped[(wind, cell, arm, d, seed)]
                        assert len(rr) == 10 and sorted(x["episode"] for x in rr) == list(range(10))
                        values = np.asarray([x[metric] for x in rr], dtype=float)
                        assert np.isfinite(values).all()
                        ss.append(values.mean())
                    ds.append(np.mean(ss))
                means[f"{cell}/{arm}"] = np.asarray(ds)
        summary[str(wind)] = summarize(means, CELLS)
    return summary


def summarize(means, cells):
    gaps = {c: means[c + "/updated"] - means[c + "/frozen"] for c in cells}
    contrasts = {c + "/U-F": interval(v) for c, v in gaps.items()}
    for b in ["short", "long"]:
        for left, right in [("mixed", "log")]:
            key = f"{b}/{left}-{right}"
            contrasts[key + "/change_in_U-F"] = interval(gaps[f"{left}_{b}"] - gaps[f"{right}_{b}"])
            for arm in ["frozen", "updated"]:
                contrasts[key + "/" + arm] = interval(
                    means[f"{left}_{b}/{arm}"] - means[f"{right}_{b}/{arm}"]
                )
    for recipe in RECIPES:
        contrasts[recipe + "/long-short/change_in_U-F"] = interval(
            gaps[recipe + "_long"] - gaps[recipe + "_short"]
        )
        for arm in ["frozen", "updated"]:
            contrasts[recipe + "/long-short/" + arm] = interval(
                means[f"{recipe}_long/{arm}"] - means[f"{recipe}_short/{arm}"]
            )
    contrasts["budget_change_in_coverage_interaction"] = interval(
        gaps["mixed_long"] - gaps["log_long"] - gaps["mixed_short"] + gaps["log_short"]
    )
    return dict(
        arm_dataset_scores={k: v.tolist() for k, v in means.items()},
        arm_means={k: float(v.mean()) for k, v in means.items()},
        contrasts=contrasts,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", choices=["query", "control"], required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--acceptance", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    gate = json.loads(a.acceptance.read_text())
    assert not gate["pilot"]
    expected = 40 if a.panel == "query" else 4800
    assert gate.get("accepted_anchors", gate.get("accepted_episodes")) == expected
    accepted = {x["index"]: x for x in gate.get("records", gate.get("files", []))}
    assert set(accepted) == set(range(expected))
    records, nonfinite = [], []
    for i in range(expected):
        path = a.root / (("anchor_" if a.panel == "query" else "episode_") + f"{i:04d}.json")
        assert hashlib.sha256(path.read_bytes()).hexdigest() == accepted[i]["sha256"]
        r = json.loads(path.read_text())
        assert not r["pilot"] and r["manifest_sha256"] == gate["manifest_sha256"]
        if a.panel == "control":
            records.append(control_record(r))
        else:
            for x in r["anchor"]["models"]:
                if x["status"] != "complete":
                    nonfinite.append(dict(anchor=i, model=x))
                    continue
                records.append(
                    dict(
                        wind=r["row"]["wind"],
                        dataset=x["dataset_replicate"],
                        coverage=x["coverage"],
                        arm=x["arm"],
                        seed=x["model_seed"],
                        episode=r["row"]["episode"],
                        physical_solution_cost=x["physical_solution_cost"],
                        query_E=x["horizon_metrics"]["50"]["original_E"]["prefix_mean"],
                        position_m=x["horizon_metrics"]["50"]["physical_position_l2_m"][
                            "prefix_mean"
                        ],
                    )
                )
    if nonfinite:
        a.output.write_text(
            json.dumps(
                dict(status="incomplete_finite_factorial", nonfinite=nonfinite, records=records),
                indent=2,
            )
            + "\n"
        )
        raise ValueError("Nonfinite queries retained; no finite-only factorial inference")
    metrics = (
        ["rmse_m"] if a.panel == "control" else ["physical_solution_cost", "query_E", "position_m"]
    )
    result = dict(
        panel=a.panel,
        accepted_records=len(records),
        acceptance_sha256=hashlib.sha256(a.acceptance.read_bytes()).hexdigest(),
        summary={m: factorial(records, m) for m in metrics},
        records=records,
        caveat="Independent confirmation of a result-selected finding; original and new datasets analyzed separately; n=5 training datasets; descriptive nonsimultaneous t95 intervals, no p-values. Budget includes schedule length/selection opportunities. Conditional contrasts do not identify a unique latent mechanism.",
    )
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(a.panel, len(records), "records analyzed")


if __name__ == "__main__":
    main()
