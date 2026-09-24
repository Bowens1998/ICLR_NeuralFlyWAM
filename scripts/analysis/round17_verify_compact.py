"""Independent flat-within-dataset reconstruction of matched baseline contrasts."""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


def verify(report, learned):
    rr = report["records"]
    assert len(rr) == 80 and {(r["kind"], r["wind"], r["episode"]) for r in rr} == {
        (k, w, e) for k in ["nominal", "true_mean"] for w in [0.0, 4.9, 6.1, 8.5] for e in range(10)
    }
    checked = 0
    for wind, block in report["summary"].items():
        for kind, cell in block.items():
            scores = [
                r["rmse_m"]
                for r in sorted(rr, key=lambda r: r["episode"])
                if r["kind"] == kind and r["wind"] == float(wind)
            ]
            assert scores == cell["episode_scores"] and len(scores) == 10
            base = statistics.fmean(scores)
            assert math.isclose(base, cell["mean"], rel_tol=1e-12, abs_tol=1e-14)
            assert cell["failures"] == sum(
                r["failed"] for r in rr if r["kind"] == kind and r["wind"] == float(wind)
            )
            assert set(cell["learned_minus_reference"]) == set(
                learned["summary"]["rmse_m"][wind]["arm_means"]
            )
            for key, actual in cell["learned_minus_reference"].items():
                coverage, arm = key.split("/")
                differences = []
                for d in range(5):
                    rows = [
                        r
                        for r in learned["records"]
                        if (r["wind"], r["coverage"], r["arm"], r["dataset"])
                        == (float(wind), coverage, arm, d)
                    ]
                    assert len(rows) == 30 and {(r["seed"], r["episode"]) for r in rows} == {
                        (s, e) for s in [70, 71, 72] for e in range(10)
                    }
                    differences.append(statistics.fmean(r["rmse_m"] for r in rows) - base)
                mean = statistics.fmean(differences)
                half = 2.7764451051977987 * statistics.stdev(differences) / math.sqrt(5)
                expected = differences + [mean, mean - half, mean + half]
                values = (
                    actual["dataset_differences"] + [actual["mean"]] + actual["descriptive_t95ci"]
                )
                assert len(expected) == len(values)
                assert all(
                    math.isclose(x, y, rel_tol=1e-10, abs_tol=1e-12)
                    for x, y in zip(expected, values)
                )
                assert actual["negative_datasets"] == sum(v < 0 for v in differences)
                assert actual["positive_datasets"] == sum(v > 0 for v in differences)
                checked += 1
    assert checked == 144
    return dict(
        accepted_episodes=80,
        failures=sum(r["failed"] for r in rr),
        independently_recomputed_contrasts=checked,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--round16", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    r = verify(json.loads(a.input.read_text()), json.loads(a.round16.read_text()))
    r.update(
        input_sha256=hashlib.sha256(a.input.read_bytes()).hexdigest(),
        round16_sha256=hashlib.sha256(a.round16.read_bytes()).hexdigest(),
    )
    a.output.write_text(json.dumps(r, indent=2) + "\n")
    print(json.dumps(r))
