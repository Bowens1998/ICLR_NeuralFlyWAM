"""Generate paired training-only log/candidate branches without touching old data."""
# ruff: noqa: E402 -- standalone entry point resolves repository imports first.

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from ast import literal_eval
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import round12_diagnostics as diag

from latent_aero_wam.data.parser import load_flight
from latent_aero_wam.sim import QuadrotorSim, SimParams, random_trajectory
from latent_aero_wam.utils.rotation import matrix_to_quat_np, matrix_to_rot6d_np, so3_log_np


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def design():
    old = json.loads((ROOT / "configs/sim/round18_generation.json").read_text())
    rows = []
    for d in range(5):
        split = ROOT / f"artifacts/round18/data{d}_split.json"
        m = json.loads(split.read_text())
        blocks = m["splits"]["d1_train"]
        for b in blocks:
            row = next(r for r in old["rows"] if r["name"] == b["file"])
            assert row["group"] == "train" and row["replicate"] == d
            rows.append(
                dict(
                    index=len(rows),
                    dataset=d,
                    original_row=row,
                    block=b,
                    centers=np.linspace(b["start"], b["stop"] - 1, 96, dtype=int).tolist(),
                    split_sha256=sha(split),
                    normalizer_sha256=sha(ROOT / f"artifacts/round18/data{d}_norm.json"),
                )
            )
    assert len(rows) == 80
    return dict(
        schema="round18-branches-v1",
        params=old["params"],
        rows=rows,
        protocol_sha256=sha(ROOT / "docs/ROUND18_PROTOCOL.md"),
    )


def targets(truth, snapshot, choice):
    # 8 noise draws paired with one chosen action per draw.
    rr = np.arange(8)
    v = truth["v"][rr, choice]
    R = truth["R"][rr, choice]
    w = truth["w"][rr, choice]
    prior_v = np.concatenate(
        [np.broadcast_to(snapshot["state"]["v"], (8, 1, 3)), v[:, :-1]], axis=1
    )
    prior_R = np.concatenate(
        [np.broadcast_to(snapshot["state"]["R"], (8, 1, 3, 3)), R[:, :-1]], axis=1
    )
    prior_w = np.concatenate(
        [np.broadcast_to(snapshot["state"]["w"], (8, 1, 3)), w[:, :-1]], axis=1
    )
    return dict(
        state=np.concatenate([v, matrix_to_rot6d_np(R), w], axis=-1).astype("float32"),
        R=R.astype("float32"),
        delta=np.concatenate(
            [v - prior_v, so3_log_np(np.swapaxes(prior_R, -2, -1) @ R), w - prior_w], axis=-1
        ).astype("float32"),
        p=truth["p"][rr, choice].astype("float32"),
        threshold=truth["threshold_exceeded"][rr, choice],
    )


def generate(m, index, data_root, out):
    assert m == design()
    spec = m["rows"][index]
    row = spec["original_row"]
    path = Path(data_root) / (row["name"] + ".csv")
    assert sha(path) == spec["block"]["sha256"]
    provenance = json.loads(path.with_suffix(".generation.json").read_text())
    assert provenance["row"] == row and provenance["params"] == m["params"]
    raw = pd.read_csv(path)
    flight = load_flight(path)
    pp = SimParams(**m["params"])
    sim = QuadrotorSim(pp, seed=row["noise_seed"])
    traj = random_trajectory(np.random.default_rng(row["trajectory_seed"]), 60.0, pp.dt)
    assert hashlib.sha256(traj.tobytes()).hexdigest() == provenance["trajectory_sha256"]
    vel = np.gradient(traj, pp.dt, axis=0)
    acc = np.gradient(vel, pp.dt, axis=0)
    state = dict(p=traj[0].copy(), v=np.zeros(3), R=np.eye(3), w=np.zeros(3))
    gust = np.zeros(3)
    errors = {k: 0.0 for k in ["p", "v", "R", "w", "T_sp", "q_sp"]}
    snapshots = {}
    for i in range(max(spec["centers"]) + 1):
        if i in spec["centers"]:
            snapshots[i] = dict(
                state={k: v.copy() for k, v in state.items()},
                gust=gust.copy(),
                p_ref=traj[i + 1 : i + 51],
                v_ref=vel[i + 1 : i + 51],
            )
        gust += -pp.dt * gust / pp.gust_tau + sim.rng.normal(
            0, pp.gust_std * max(row["mean_wind"], 0.3) * np.sqrt(2 * pp.dt / pp.gust_tau), 3
        )
        throttle, Rsp = sim.baseline_controller(state, traj[i], vel[i])
        for k in errors:
            actual = (
                np.asarray([throttle])
                if k == "T_sp"
                else matrix_to_quat_np(Rsp[None])[0]
                if k == "q_sp"
                else state[k]
            )
            errors[k] = max(
                errors[k], float(np.max(np.abs(actual - np.asarray(literal_eval(raw.iloc[i][k])))))
            )
        if max(errors.values()) > 5.1e-9:
            raise ValueError(f"CSV replay mismatch at{i}: {errors}")
        state = sim.step(state, throttle, Rsp, np.array([row["mean_wind"], 0.0, 0.0]) + gust)
    arrays = {}
    identities = []
    started = time.monotonic()
    for center, snap in snapshots.items():
        seed = [18, spec["dataset"], row["index"], center]
        nominal = diag.legacy.pd_plan(sim, snap["state"], traj, vel, acc, center, 50)
        bank = diag.TraceMPPI(
            None, n_samples=64, rng=np.random.default_rng(np.random.SeedSequence(seed + [0]))
        ).sample_bank(nominal)
        ids = np.random.default_rng(np.random.SeedSequence(seed + [1])).choice(64, 8, replace=False)
        actions = np.concatenate(
            [flight.action[center : center + 50][None], bank["actions"][ids]], axis=0
        ).astype("float32")
        noise = diag.noise_bundle(seed + [2], 8)
        truth = diag.simulate(snap, actions, noise, row["mean_wind"], pp)
        for label, choice in [("logged", np.zeros(8, dtype=int)), ("candidate", np.arange(1, 9))]:
            for k, v in targets(truth, snap, choice).items():
                arrays.setdefault(label + "/" + k, []).append(v)
        arrays.setdefault("logged/actions", []).append(actions[0])
        arrays.setdefault("candidate/actions", []).append(actions[1:])
        arrays.setdefault("candidate_indices", []).append(ids)
        identities.append(center)
    arrays = {k: np.stack(v) for k, v in arrays.items()}
    arrays["centers"] = np.asarray(identities)
    # Store original current values for checking delta alignment against full-state replay.
    arrays["current_v"] = np.stack([snapshots[t]["state"]["v"] for t in identities])
    arrays["current_R"] = np.stack([snapshots[t]["state"]["R"] for t in identities])
    arrays["current_w"] = np.stack([snapshots[t]["state"]["w"] for t in identities])
    for k, v in arrays.items():
        assert np.isfinite(v).all(), k
    np.savez_compressed(out.with_suffix(".npz"), **arrays)
    result = dict(
        status="complete",
        spec=spec,
        replay_errors=errors,
        trace_sha256=sha(out.with_suffix(".npz")),
        centers=len(identities),
        branches_per_regime=len(identities) * 8,
        seconds=time.monotonic() - started,
        source=os.environ.get("LATENT_WAM_SOURCE_COMMIT", "local-development"),
        protocol_sha256=m["protocol_sha256"],
    )
    out.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", type=Path, required=True)
    ap.add_argument("--write-design", action="store_true")
    ap.add_argument("--index", type=int)
    ap.add_argument("--data-root")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    if args.write_design:
        with args.design.open("x") as f:
            f.write(json.dumps(design(), indent=2) + "\n")
        return
    args.output.mkdir(parents=True, exist_ok=True)
    stem = args.output / f"flight_{args.index:03d}"
    with stem.with_suffix(".claim").open("x") as f:
        f.write(str(os.getpid()))
    try:
        print(
            json.dumps(
                generate(json.loads(args.design.read_text()), args.index, args.data_root, stem),
                indent=2,
            )
        )
    except Exception:
        stem.with_suffix(".error").write_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
