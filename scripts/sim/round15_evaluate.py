"""Fresh query/control factorial; unchanged MPPI and physical scoring algorithms."""

# ruff: noqa: E402 -- standalone entry point resolves repository imports first.
import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import round12_diagnostics as diag

from latent_aero_wam.evaluation.evaluator import load_model_from_checkpoint
from latent_aero_wam.sim import QuadrotorSim, random_trajectory

WINDS = [4.9, 6.1, 8.5]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sources():
    paths = list((ROOT / "src").rglob("*.py")) + [
        ROOT / "scripts/sim" / n
        for n in [
            "round15_evaluate.py",
            "round12_diagnostics.py",
            "round12_diagnostic_protocol.py",
            "round11_closed_loop.py",
            "round11_closed_loop_protocol.py",
            "closed_loop.py",
        ]
    ]
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def rows():
    out = []
    for d in range(5):
        for coverage in ["log", "mixed"]:
            for arm in ["frozen", "updated"]:
                for seed in [70, 71, 72]:
                    for wind in WINDS:
                        for e in range(10):
                            out.append(
                                dict(
                                    index=len(out),
                                    kind="model",
                                    dataset_replicate=d,
                                    coverage=coverage,
                                    arm=arm,
                                    model=f"r15_d{d}_{coverage}_{arm}",
                                    model_seed=seed,
                                    wind=wind,
                                    episode=e,
                                    environment_seed=610000 + e,
                                    trajectory_seed=620000 + e,
                                    planning_seed=630000 + e,
                                )
                            )
    return out


def fresh_anchor(index):
    wind = WINDS[index // 10]
    e = index % 10
    sim = QuadrotorSim(seed=610000 + e)
    traj = random_trajectory(np.random.default_rng(620000 + e), 33.02, 0.02)
    vel = np.gradient(traj, 0.02, axis=0)
    acc = np.gradient(vel, 0.02, axis=0)
    state = dict(p=traj[0].copy(), v=np.zeros(3), R=np.eye(3), w=np.zeros(3))
    gust = np.zeros(3)
    history = diag.empty_history()
    for i in range(101):
        history.observe_state(state["v"], state["R"], state["w"])
        if i == 100:
            break
        pp = sim.p_
        gust += -0.02 * gust / pp.gust_tau + sim.rng.normal(
            0, pp.gust_std * max(wind, 0.3) * np.sqrt(0.04 / pp.gust_tau), 3
        )
        throttle, R = sim.baseline_controller(state, traj[i], vel[i])
        action = np.r_[throttle, diag.legacy.matrix_to_quat_batch(R[None])[0]]
        history.observe_action(action)
        state = sim.step(
            state,
            float(action[0]),
            diag.legacy.quat_to_matrix_batch(action[None, 1:])[0],
            np.array([wind, 0.0, 0.0]) + gust,
        )
    snap = dict(
        step=100,
        state=state,
        gust=gust,
        history=diag.history_arrays(history),
        p_ref=traj[101:151],
        v_ref=vel[101:151],
    )
    nominal = diag.legacy.pd_plan(sim, state, traj, vel, acc, 100, 50)
    bank = diag.TraceMPPI(None, n_samples=64, rng=np.random.default_rng(630000 + e)).sample_bank(
        nominal
    )
    return (
        snap,
        bank,
        dict(
            stage="a1",
            index=150000 + index,
            wind=wind,
            episode=e,
            environment_seed=610000 + e,
            trajectory_seed=620000 + e,
            planning_seed=630000 + e,
        ),
    )


def make_manifest(root):
    models = []
    accepted = {}
    for d in range(5):
        ap = ROOT / f"runs/evidence/round15_bundle/round15_coverage_data{d}_acceptance.json"
        a = json.loads(ap.read_text())
        assert a["accepted_runs"] == 12 and a["accepted_npz"] == 24
        accepted[str(ap.relative_to(ROOT))] = sha(ap)
        for c in ["log", "mixed"]:
            for arm in ["frozen", "updated"]:
                for seed in [70, 71, 72]:
                    rel = f"round15_coverage_data{d}_v1/r15_d{d}_{c}_{arm}_seed{seed}/checkpoint_best.pt"
                    p = Path(root) / rel
                    assert sha(p) == a["sha256"]["runs/results/" + rel]
                    cp = torch.load(p, map_location="cpu", weights_only=False)
                    assert cp["provenance"]["git_commit"] == a["source"]
                    models.append(
                        dict(
                            dataset=d,
                            coverage=c,
                            arm=arm,
                            seed=seed,
                            path=rel,
                            checkpoint_sha256=sha(p),
                            config=cp["config"],
                            provenance=cp["provenance"],
                            normalizer_sha256=sha(ROOT / cp["config"]["data"]["normalizer_path"]),
                            split_sha256=sha(ROOT / cp["config"]["data"]["manifest_path"]),
                        )
                    )
    rr = rows()
    return dict(
        schema="round15-evaluation-v1",
        models=models,
        rows=rr,
        source_sha256=sources(),
        protocol_sha256={
            n: sha(ROOT / "docs" / n) for n in ["ROUND15_PROTOCOL.md", "ROUND15_EVALUATION.md"]
        },
        training_acceptance=accepted,
        query_pilot=[0, 10, 20],
        control_pilot=[
            r["index"]
            for r in rr
            if r["dataset_replicate"] == 0 and r["model_seed"] == 70 and r["episode"] == 0
        ],
    )


def validate(m):
    assert (
        m["schema"] == "round15-evaluation-v1"
        and m["source_sha256"] == sources()
        and m["rows"] == rows()
    )
    assert m["protocol_sha256"] == {
        n: sha(ROOT / "docs" / n) for n in ["ROUND15_PROTOCOL.md", "ROUND15_EVALUATION.md"]
    }
    assert [(x["dataset"], x["coverage"], x["arm"], x["seed"]) for x in m["models"]] == [
        (d, c, a, s)
        for d in range(5)
        for c in ["log", "mixed"]
        for a in ["frozen", "updated"]
        for s in [70, 71, 72]
    ]
    assert m["query_pilot"] == [0, 10, 20] and m["control_pilot"] == [
        r["index"]
        for r in rows()
        if r["dataset_replicate"] == 0 and r["model_seed"] == 70 and r["episode"] == 0
    ]
    for x in m["models"]:
        assert (
            sha(ROOT / x["config"]["data"]["normalizer_path"]) == x["normalizer_sha256"]
            and sha(ROOT / x["config"]["data"]["manifest_path"]) == x["split_sha256"]
        )


def load_model(spec, root, device):
    p = Path(root) / spec["path"]
    assert sha(p) == spec["checkpoint_sha256"]
    model, state, art = load_model_from_checkpoint(p, torch.device(device))
    assert (
        state["config"] == spec["config"]
        and state["provenance"] == spec["provenance"]
        and model.n_parameters() == 39497
    )
    return (
        model,
        art["normalizer"].metric_scales,
        dict(
            checkpoint_sha256=spec["checkpoint_sha256"], arm=spec["arm"], coverage=spec["coverage"]
        ),
    )


def execute(m, index, root, device, panel):
    validate(m)
    start = time.monotonic()
    if panel == "control":
        row = m["rows"][index]
        x = next(
            x
            for x in m["models"]
            if (x["dataset"], x["coverage"], x["arm"], x["seed"])
            == (row["dataset_replicate"], row["coverage"], row["arm"], row["model_seed"])
        )
        model, _, _ = load_model(x, root, "cpu")
        del model
        r = diag.original_runner.run(row, Path(root) / x["path"], device, scored_steps=1500)
        r.update(checkpoint_sha256=x["checkpoint_sha256"])
        return r, None
    snap, bank, row = fresh_anchor(index)
    specs = [
        dict(
            model=x["config"]["model"]["name"],
            dataset_replicate=x["dataset"],
            model_seed=x["seed"],
            identity=x,
        )
        for x in m["models"]
    ]
    result, trace = diag.evaluate_anchor(
        snap, bank, specs, lambda x: load_model(x["identity"], root, device), row, 8
    )
    return dict(
        status="complete", row=row, index=index, anchor=result, seconds=time.monotonic() - start
    ), trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--write-manifest", action="store_true")
    ap.add_argument("--checkpoint-root", default="runs/results")
    ap.add_argument("--panel", choices=["query", "control"])
    ap.add_argument("--index", type=int)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--output", type=Path)
    ap.add_argument("--pilot", action="store_true")
    a = ap.parse_args()
    if a.write_manifest:
        with a.manifest.open("x") as f:
            f.write(json.dumps(make_manifest(a.checkpoint_root), indent=2) + "\n")
        return
    m = json.loads(a.manifest.read_text())
    validate(m)
    if a.pilot:
        assert a.index in m[a.panel + "_pilot"]
    a.output.mkdir(parents=True, exist_ok=True)
    p = a.output / (("anchor_" if a.panel == "query" else "episode_") + f"{a.index:04d}")
    identity = dict(
        index=a.index,
        manifest_sha256=sha(a.manifest),
        execution_source=os.environ.get("LATENT_WAM_SOURCE_COMMIT", "local-development"),
        pilot=a.pilot,
        panel=a.panel,
    )
    with p.with_suffix(".claim").open("x") as f:
        f.write(json.dumps(identity))
    try:
        torch.set_num_threads(1)
        r, z = execute(m, a.index, a.checkpoint_root, a.device, a.panel)
        r.update(
            identity,
            schema=m["schema"],
            source_sha256=m["source_sha256"],
            device=a.device,
            gpu=torch.cuda.get_device_name() if a.device.startswith("cuda") else None,
        )
        if z is not None:
            np.savez_compressed(p.with_suffix(".npz"), **z)
            r["trace_sha256"] = sha(p.with_suffix(".npz"))
        with p.with_suffix(".json").open("x") as f:
            f.write(json.dumps(r, indent=2) + "\n")
        print(a.panel, a.index, r.get("rmse_m", r.get("seconds")))
    except Exception:
        p.with_suffix(".error").write_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
