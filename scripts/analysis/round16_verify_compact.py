"""Independently reconstruct Round16 means and contrasts from compact records."""

import argparse
import hashlib
import itertools
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np


def verify(report):
    assert report["panel"] in ("original_static_logged_actions", "query", "control")
    offline = report["panel"] == "original_static_logged_actions"
    recipes = ["dense", "recorded", "log", "mixed"]
    cells = [f"{r}_{b}" for r in recipes for b in ["short", "long"]]
    winds = [3.7, 6.1, 8.5] if offline else [0.0, 4.9, 6.1, 8.5]
    if not offline:
        cells.append("original")
    arms = ["frozen", "updated"]
    records = report["records"]
    fields = ["wind", "coverage", "arm", "dataset", "seed"]
    domains = [winds, cells, arms, range(5), [70, 71, 72]]
    if not offline:
        fields.append("episode")
        domains.append(range(10))
    identities = [tuple(r[k] for k in fields) for r in records]
    assert len(identities) == len(set(identities))
    assert set(identities) == set(itertools.product(*domains))
    if offline:
        assert all(r["windows"] == 28500 for r in records)
    else:
        assert report["accepted_records"] == 10800
    metrics = (
        ["original_E"]
        if offline
        else (
            ["rmse_m"]
            if report["panel"] == "control"
            else ["physical_solution_cost", "query_E", "position_m"]
        )
    )
    checked = 0
    for metric in metrics:
        grouped = defaultdict(list)
        for r in records:
            value = float(r[metric])
            assert math.isfinite(value)
            grouped[(r["wind"], r["coverage"], r["arm"], r["dataset"])].append(value)
        for wind in winds:
            # Balanced cells allow an independent flat mean within each dataset.
            means = {}
            for c, a in itertools.product(cells, arms):
                scores = []
                for d in range(5):
                    values = grouped[(wind, c, a, d)]
                    assert len(values) == (3 if offline else 30)
                    scores.append(statistics.fmean(values))
                means[c + "/" + a] = np.asarray(scores)
            block = (
                report["summary"][str(wind)] if offline else report["summary"][metric][str(wind)]
            )
            assert set(block["arm_means"]) == set(block["arm_dataset_scores"]) == set(means)
            for key, values in means.items():
                np.testing.assert_allclose(
                    block["arm_dataset_scores"][key], values, rtol=1e-11, atol=1e-13
                )
                assert math.isclose(
                    block["arm_means"][key], statistics.fmean(values), rel_tol=1e-11, abs_tol=1e-13
                )

            def gap(c, values=means):
                return values[c + "/updated"] - values[c + "/frozen"]

            expected = {c + "/U-F": gap(c) for c in cells}
            for b in ["short", "long"]:
                for left, right in zip(recipes[1:], recipes[:-1]):
                    key = f"{b}/{left}-{right}"
                    expected[key + "/change_in_U-F"] = gap(left + "_" + b) - gap(right + "_" + b)
                    for a in arms:
                        expected[key + "/" + a] = (
                            means[f"{left}_{b}/{a}"] - means[f"{right}_{b}/{a}"]
                        )
            for r in recipes:
                expected[r + "/long-short/change_in_U-F"] = gap(r + "_long") - gap(r + "_short")
                for a in arms:
                    expected[r + "/long-short/" + a] = (
                        means[r + "_long/" + a] - means[r + "_short/" + a]
                    )
            expected["budget_change_in_coverage_interaction"] = (
                expected["long/mixed-log/change_in_U-F"] - expected["short/mixed-log/change_in_U-F"]
            )
            assert set(expected) == set(block["contrasts"])
            for key, values in expected.items():
                actual = block["contrasts"][key]
                mean = statistics.fmean(values)
                # Fixed df=4, two-sided t95 critical value; no analysis helper import.
                half = 2.7764451051977987 * statistics.stdev(values) / math.sqrt(5)
                np.testing.assert_allclose(
                    actual["dataset_differences"], values, rtol=1e-11, atol=1e-13
                )
                np.testing.assert_allclose(actual["mean"], mean, rtol=1e-11, atol=1e-13)
                np.testing.assert_allclose(
                    actual["descriptive_t95ci"], [mean - half, mean + half], rtol=1e-10, atol=1e-12
                )
                assert actual["negative_datasets"] == sum(v < 0 for v in values)
                assert actual["positive_datasets"] == sum(v > 0 for v in values)
                checked += 1
    return dict(
        panel=report["panel"],
        complete_records=len(records),
        independently_checked_contrasts=checked,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    result = verify(json.loads(args.input.read_text()))
    result["input_sha256"] = hashlib.sha256(args.input.read_bytes()).hexdigest()
    result["scope"] = (
        "Independent compact-record statistical reconstruction; full raw numerical acceptance remains separately required."
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
