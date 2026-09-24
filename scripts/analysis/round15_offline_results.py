"""Descriptive original-static-test errors for all accepted coverage arms."""

import hashlib
import json
from pathlib import Path

import numpy as np
from round15_results import interval


def main():
    records = []
    gates = {}
    windows = {}
    for d in range(5):
        gate = Path(f"runs/evidence/round15_bundle/round15_coverage_data{d}_acceptance.json")
        a = json.loads(gate.read_text())
        assert a["accepted_runs"] == 12 and a["accepted_npz"] == 24 and not a["pilot"]
        gates[str(gate)] = hashlib.sha256(gate.read_bytes()).hexdigest()
        for coverage in ["log", "mixed"]:
            for arm in ["frozen", "updated"]:
                for seed in [70, 71, 72]:
                    p = Path(f"runs/results/round15_coverage_data{d}_v1") / (
                        f"r15_d{d}_{coverage}_{arm}_seed{seed}/eval/d2_static_ood_per_window.npz"
                    )
                    assert hashlib.sha256(p.read_bytes()).hexdigest() == a["sha256"][str(p)]
                    with np.load(p, allow_pickle=False) as z:
                        error = z["err_aggregate"]
                        assert error.shape == (85500, 50) and np.isfinite(error).all()
                        for wind in [3.7, 6.1, 8.5]:
                            mask = np.isclose(z["wind_mps"], wind, rtol=0, atol=1e-5)
                            n = int(mask.sum())
                            assert n > 0
                            key = (d, wind)
                            assert windows.setdefault(key, n) == n
                            records.append(
                                dict(
                                    dataset=d,
                                    coverage=coverage,
                                    arm=arm,
                                    seed=seed,
                                    wind=wind,
                                    windows=n,
                                    original_E=float(error[mask].mean()),
                                )
                            )
    summary = {}
    for wind in [3.7, 6.1, 8.5]:
        means = {}
        for c in ["log", "mixed"]:
            for arm in ["frozen", "updated"]:
                ds = []
                for d in range(5):
                    rr = [
                        r
                        for r in records
                        if (r["wind"], r["dataset"], r["coverage"], r["arm"]) == (wind, d, c, arm)
                    ]
                    assert sorted(r["seed"] for r in rr) == [70, 71, 72]
                    ds.append(np.mean([r["original_E"] for r in rr]))
                means[c + "/" + arm] = np.asarray(ds)
        gaps = {c: means[c + "/updated"] - means[c + "/frozen"] for c in ["log", "mixed"]}
        summary[str(wind)] = dict(
            arm_dataset_scores={k: v.tolist() for k, v in means.items()},
            arm_means={k: float(v.mean()) for k, v in means.items()},
            contrasts={
                **{c + "_updated_minus_frozen": interval(v) for c, v in gaps.items()},
                "coverage_by_memory_interaction": interval(gaps["mixed"] - gaps["log"]),
                **{
                    arm + "_mixed_minus_log": interval(means["mixed/" + arm] - means["log/" + arm])
                    for arm in ["frozen", "updated"]
                },
            },
        )
    Path("reports/ROUND15_OFFLINE_RESULTS.json").write_text(
        json.dumps(
            dict(
                panel="original_static_logged_actions",
                metric="original_E_H50_prefix",
                acceptance_sha256=gates,
                records=records,
                summary=summary,
                caveat="Descriptive original test-set audit. Average windows/horizons within checkpoint, then seeds within dataset; n=5 training datasets and nonsimultaneous t intervals. No additional p-values or selection. Not a literal replication of the old overlapping-window training recipe.",
            ),
            indent=2,
        )
        + "\n"
    )
    print("Accepted offline descriptive records", len(records))


if __name__ == "__main__":
    main()
