"""Original static logged-action prediction audit; all 240 matched models retained."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from round16_results import RECIPES, summarize


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("runs/results"))
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    records, gates, hashes, ids = [], {}, {}, {}
    cells = [f"{r}_{b}" for r in RECIPES for b in ["short", "long"]]
    for d in range(5):
        accepted = {}
        for n, family, count in [(15, "coverage", 12), (16, "bridge", 36)]:
            p = Path(f"runs/evidence/round{n}_bundle/round{n}_{family}_data{d}_acceptance.json")
            g = json.loads(p.read_text())
            assert g["accepted_runs"] == count and g["accepted_npz"] == 2 * count and not g["pilot"]
            accepted[n] = g
            gates[str(p)] = sha(p)
        for c in cells:
            recipe, budget = c.split("_")
            n = 15 if c in ["log_short", "mixed_short"] else 16
            family = "coverage" if n == 15 else "bridge"
            for arm in ["frozen", "updated"]:
                for seed in [70, 71, 72]:
                    name = (
                        f"r15_d{d}_{recipe}_{arm}"
                        if n == 15
                        else f"r16_d{d}_{recipe}_{budget}_{arm}"
                    )
                    rel = f"round{n}_{family}_data{d}_v1/{name}_seed{seed}/eval/d2_static_ood_per_window.npz"
                    path = a.root / rel
                    digest = sha(path)
                    assert digest == accepted[n]["sha256"]["runs/results/" + rel]
                    hashes[rel] = digest
                    with np.load(path, allow_pickle=False) as z:
                        err = z["err_aggregate"]
                        assert err.shape == (85500, 50) and np.isfinite(err).all()
                        identity = tuple(
                            z[k].copy()
                            for k in ["flight_ids", "centre_index", "wind_mps", "flight_names"]
                        )
                        expected = ids.setdefault(d, identity)
                        assert all(np.array_equal(x, y) for x, y in zip(identity, expected))
                        for wind in [3.7, 6.1, 8.5]:
                            mask = np.isclose(z["wind_mps"], wind, rtol=0, atol=1e-5)
                            assert int(mask.sum()) == 28500
                            records.append(
                                dict(
                                    dataset=d,
                                    coverage=c,
                                    arm=arm,
                                    seed=seed,
                                    wind=wind,
                                    windows=int(mask.sum()),
                                    original_E=float(err[mask].mean()),
                                )
                            )
    assert len(records) == 720
    summaries = {}
    for wind in [3.7, 6.1, 8.5]:
        means = {}
        for c in cells:
            for arm in ["frozen", "updated"]:
                ds = []
                for d in range(5):
                    rr = [
                        r
                        for r in records
                        if (r["wind"], r["coverage"], r["arm"], r["dataset"]) == (wind, c, arm, d)
                    ]
                    assert sorted(r["seed"] for r in rr) == [70, 71, 72]
                    ds.append(np.mean([r["original_E"] for r in rr]))
                means[f"{c}/{arm}"] = np.asarray(ds)
        summaries[str(wind)] = summarize(means, cells)
    a.output.write_text(
        json.dumps(
            dict(
                panel="original_static_logged_actions",
                metric="original_E_H50_prefix",
                acceptance_sha256=gates,
                raw_npz_sha256=hashes,
                records=records,
                summary=summaries,
                caveat="Static logged tests are 3.7/6.1/8.5, not the feedback zero/4.9 boundary conditions. n=5 training datasets, seed/window averages within dataset, descriptive nonsimultaneous intervals. Full240matched models, no outcome selection.",
            ),
            indent=2,
        )
        + "\n"
    )
    print("accepted offline records", len(records))


if __name__ == "__main__":
    main()
