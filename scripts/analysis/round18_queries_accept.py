"""Recompute new-model MPPI weighted solutions and costs from retained traces."""
# ruff: noqa: E402 -- resolve repository imports before standalone analysis.

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/sim"))
import round18_evaluate as runner

D = runner.diag


def accept(path, manifest, source, pilot=False):
    m = json.loads(Path(manifest).read_text())
    runner.validate(m)
    r = json.loads(Path(path).read_text())
    assert (
        r["status"] == "complete"
        and r["manifest_sha256"] == D.sha256(manifest)
        and r["execution_source"] == source
    )
    assert r["panel"] == "query" and r["index"] in range(40)
    assert len(source) == 40 and r["schema"] == m["schema"]
    assert r["source_sha256"] == m["source_sha256"]
    assert r["pilot"] == pilot
    if pilot:
        assert r["index"] in m["query_pilot"]
    claim = json.loads(Path(path).with_suffix(".claim").read_text())
    assert claim == {
        k: r[k] for k in ["index", "manifest_sha256", "execution_source", "pilot", "panel"]
    }
    expected_snapshot, expected_bank, expected_row = runner.fresh_anchor(r["index"])
    assert r["row"] == expected_row
    assert r["anchor"]["noise_seed_words"] == {
        phase: D.protocol.noise_seed_words("a1", 180000 + r["index"], 100, phase)
        for phase in ["construct", "evaluate"]
    }
    trace = Path(path).with_suffix(".npz")
    assert r["trace_sha256"] == D.sha256(trace)
    records = r["anchor"]["models"]
    assert len(records) == 120
    assert [
        (x["dataset_replicate"], x["coverage"], x["arm"], x["model_seed"]) for x in records
    ] == [(x["dataset"], x["coverage"], x["arm"], x["seed"]) for x in m["models"]]
    with np.load(trace, allow_pickle=False) as z:
        assert all(np.isfinite(z[k]).all() for k in z.files)
        np.testing.assert_array_equal(z["bank/actions"], expected_bank["actions"])
        for key in ["p", "v", "R", "w"]:
            np.testing.assert_allclose(
                z["snapshot/state/" + key], expected_snapshot["state"][key], rtol=0, atol=1e-12
            )
        for key in expected_snapshot["history"]:
            np.testing.assert_array_equal(
                z["snapshot/history/" + key], expected_snapshot["history"][key]
            )
        for key in ["gust", "p_ref", "v_ref"]:
            np.testing.assert_array_equal(z["snapshot/" + key], expected_snapshot[key])
        np.testing.assert_array_equal(
            z["evaluation/actions"],
            D.legacy.MPPI.apply(expected_bank["nominal"], z["evaluation/perturbations"]),
        )
        for phase in ["construct", "evaluate"]:
            noise = D.noise_bundle(
                D.protocol.noise_seed_words("a1", 180000 + r["index"], 100, phase), 8
            )
            for key, value in noise.items():
                np.testing.assert_array_equal(z["noise/" + phase + "/" + key], value)
        replay = D.simulate(expected_snapshot, z["evaluation/actions"], noise, expected_row["wind"])
        for key in ["p", "v", "R", "w", "p_planner"]:
            np.testing.assert_allclose(replay[key], z["evaluation/" + key], rtol=1e-11, atol=1e-12)
        snap = dict(
            state={"p": z["snapshot/state/p"]}, p_ref=z["snapshot/p_ref"], v_ref=z["snapshot/v_ref"]
        )
        truth = {k: z["evaluation/" + k] for k in ["p", "v", "R", "w", "p_planner"]}
        pert = z["evaluation/perturbations"]
        complete = [x for x in records if x["status"] == "complete"]
        assert all(x["status"] in ["complete", "nonfinite_prediction"] for x in records)
        assert truth["p"].shape == (8, 65 + len(complete), 50, 3)
        costs = D.costs(truth, snap, pert)
        for label in ["physical", "planner"]:
            np.testing.assert_allclose(
                costs[label], z["evaluation_cost/" + label], rtol=1e-10, atol=1e-12
            )
        for i, rec in enumerate(records):
            assert rec["checkpoint_sha256"] == m["models"][i]["checkpoint_sha256"]
            if rec["status"] != "complete":
                continue
            state = z[f"model_{i}/state"]
            rotation = z[f"model_{i}/R"]
            assert state.shape == (64, 50, 12)
            pred = D.planner_cost(
                state[..., :3],
                snap["state"]["p"],
                snap["p_ref"],
                snap["v_ref"],
                z["bank/perturbations"],
                np.array([0.04, 0.08, 0.08]),
            )
            weights, solution = D.softmin_solution(pred, z["bank/perturbations"])
            np.testing.assert_allclose(pred, z[f"model_{i}/predicted_cost"], rtol=1e-10, atol=1e-12)
            np.testing.assert_allclose(weights, z[f"model_{i}/weights"], rtol=1e-10, atol=1e-12)
            np.testing.assert_allclose(
                solution, z[f"model_{i}/solution_perturbation"], rtol=1e-10, atol=1e-12
            )
            si = 64 + rec["solution_index"]
            np.testing.assert_allclose(pert[si], solution, rtol=1e-10, atol=1e-12)
            np.testing.assert_allclose(
                rec["first_action"], z["evaluation/actions"][si, 0], rtol=0, atol=1e-12
            )
            for label in ["physical", "planner"]:
                assert np.isclose(
                    costs[label][:, si].mean(),
                    rec[label + "_solution_cost"],
                    rtol=1e-10,
                    atol=1e-12,
                )
                assert np.isclose(
                    (costs[label][:, si] - costs[label][:, 64]).mean(),
                    rec[label + "_solution_cost_excess_vs_oracle_softmin"],
                    rtol=1e-10,
                    atol=1e-12,
                )
            scales = D.Normalizer.load(
                ROOT / m["models"][i]["config"]["data"]["normalizer_path"]
            ).metric_scales
            metrics = D.horizon_metrics(
                state, rotation, {k: v[:, :64] for k, v in truth.items()}, snap, scales
            )
            for h, row in metrics.items():
                for metric, values in row.items():
                    if isinstance(values, dict):
                        for stat, v in values.items():
                            assert np.isclose(
                                v, rec["horizon_metrics"][h][metric][stat], rtol=1e-10, atol=1e-12
                            )
        assert sorted(x["solution_index"] for x in complete) == list(range(1, len(complete) + 1))
    return dict(
        index=r["index"],
        sha256=D.sha256(path),
        trace_sha256=r["trace_sha256"],
        nonfinite_models=len(records) - len(complete),
        elapsed_seconds=r["seconds"],
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    indices = [0, 10, 20, 30] if a.pilot else range(40)
    records = [
        accept(a.root / f"anchor_{i:04d}.json", a.manifest, a.source, a.pilot) for i in indices
    ]
    with a.output.open("x") as f:
        f.write(
            json.dumps(
                dict(
                    accepted_anchors=len(records),
                    records=records,
                    source=a.source,
                    pilot=a.pilot,
                    manifest_sha256=D.sha256(a.manifest),
                ),
                indent=2,
            )
            + "\n"
        )
    print("accepted anchors", len(records))
