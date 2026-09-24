"""Round12B/C matched-budget, plant-only control sensitivity evaluator.

This module never mutates the original Round11 evaluator, checkpoints or results.
Formal and pilot execution use separately frozen protocols and output namespaces.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import closed_loop as legacy  # noqa: E402
import round11_closed_loop_protocol as scoring  # noqa: E402
import round12_control_protocol as protocol  # noqa: E402

from latent_aero_wam.data.splits import load_manifest  # noqa: E402
from latent_aero_wam.sim import QuadrotorSim, random_trajectory  # noqa: E402
from latent_aero_wam.sim.quadrotor import SimParams  # noqa: E402
from latent_aero_wam.training.checkpoint import load_checkpoint  # noqa: E402
from latent_aero_wam.utils.config import config_hash, resolve_path  # noqa: E402


class NonfinitePrediction(FloatingPointError):
    """A controller failure, distinct from software/provenance execution failures."""


def finite_predictor(dynamics):
    def predict(actions):
        if not np.isfinite(actions).all():
            raise NonfinitePrediction("nonfinite_candidate_action")
        result = dynamics(actions)
        if np.asarray(result).shape != (*actions.shape[:2], 3):
            raise ValueError("predictor violated candidate batch/horizon contract")
        if not np.isfinite(result).all():
            raise NonfinitePrediction("nonfinite_prediction")
        return result
    return predict


def parameter_pair(changes: dict) -> tuple[SimParams, SimParams]:
    """Return independent plant/design parameters; only the plant is perturbed."""
    if changes not in [setting["plant_changes"] for setting in protocol.SETTINGS.values()]:
        raise ValueError("unregistered physics perturbation")
    baseline = SimParams()
    updates = dict(changes)
    drag_factor = updates.pop("drag_factor", None)
    if drag_factor is not None:
        updates["drag_lin"] = tuple(drag_factor * x for x in baseline.drag_lin)
        updates["drag_quad"] = tuple(drag_factor * x for x in baseline.drag_quad)
    return replace(baseline, **updates), SimParams()


def check_checkpoint(path: Path, entry: dict) -> dict:
    """Check bytes, full configuration, training provenance and current data artifacts."""
    actual_hash = protocol.file_hash(path)
    if actual_hash != entry["checkpoint_sha256"]:
        raise ValueError("checkpoint bytes differ from frozen manifest")
    state = load_checkpoint(path, map_location="cpu")
    cfg, provenance = state["config"], state["provenance"]
    expected = dict(git_commit=entry["source_commit"], model_name=entry["model_name"],
                    seed=entry["model_seed"], config_hash=entry["config_hash"],
                    split_checksum=entry["split_checksum"],
                    normalizer_sha256=entry["normalizer_sha256"])
    if any(provenance.get(k) != value for k, value in expected.items()):
        raise ValueError("checkpoint provenance differs from frozen manifest")
    if (cfg != entry["config"] or protocol.object_hash(cfg) != entry["config_sha256"]
            or config_hash(cfg) != entry["config_hash"]):
        raise ValueError("checkpoint config differs from frozen manifest")
    data = load_manifest(resolve_path(cfg["data"]["manifest_path"]))
    if data["checksum"] != entry["split_checksum"]:
        raise ValueError("training split changed since protocol lock")
    if protocol.file_hash(resolve_path(cfg["data"]["normalizer_path"])) != entry["normalizer_sha256"]:
        raise ValueError("training normalizer changed since protocol lock")
    if (float(data["dt"]) != .02 or data["history_steps"] != 100
            or data["horizon_steps"] != 50):
        raise ValueError("checkpoint training timing is incompatible with control protocol")
    return dict(checkpoint_sha256=actual_hash, provenance=expected,
                config_sha256=entry["config_sha256"], data_artifacts_checked=True)


def run(row, checkpoint=None, device="cpu", *, scored_steps=1500, dynamics_factory=None):
    """One episode. Short duration and injected dynamics are for tests/pilots only."""
    if not 1 <= scored_steps <= 1500:
        raise ValueError("invalid scored duration")
    setting = protocol.SETTINGS[row["setting"]]
    if any(row.get(key) != value for key, value in setting.items()):
        raise ValueError("row settings differ from registered intervention")
    dev = torch.device(device)
    plant_params, design_params = parameter_pair(row["plant_changes"])
    plant = QuadrotorSim(plant_params, seed=row["environment_seed"])
    design_sim = QuadrotorSim(design_params, seed=0)  # no design RNG is consumed
    horizon = row["horizon"]
    traj = random_trajectory(np.random.default_rng(row["trajectory_seed"]), 33.02, .02)
    velocity = np.gradient(traj, .02, axis=0)
    acceleration = np.gradient(velocity, .02, axis=0)
    state = dict(p=traj[0].copy(), v=np.zeros(3), R=np.eye(3), w=np.zeros(3))
    gust = np.zeros(3)
    kind = row["kind"]
    if kind not in scoring.KINDS:
        raise ValueError("unknown controller")
    dynamics = controller = None
    if kind == "model":
        dynamics = (dynamics_factory or legacy.LearnedDynamics)(checkpoint, dev)
    elif kind in ("nominal", "true_mean"):
        mean_wind = row["wind"] if kind == "true_mean" else 0.
        dynamics = legacy.TruthDynamics(design_sim, lambda t: np.array([mean_wind, 0., 0.]))
    if dynamics is not None:
        controller = legacy.MPPI(finite_predictor(dynamics), n_samples=row["candidates"],
                                 rng=np.random.default_rng(row["planning_seed"]))
    errors, actions, latencies, effort_rows = [], [], [], []
    failure = failed_step = warmup_state = warmup_seconds = None
    start = time.monotonic()
    for i in range(100 + scored_steps):
        try:
            if kind == "model":
                dynamics.observe_state(state["v"], state["R"], state["w"])
            if i == 100:
                warmup_state = {key: value.tolist() for key, value in state.items()}
                warmup_seconds = time.monotonic() - start
                if kind == "model" and not dynamics.ready():
                    raise ValueError("learned history contract violated")
            pp = plant.p_
            gust += -.02 * gust / pp.gust_tau + plant.rng.normal(
                0, pp.gust_std * max(row["wind"], .3) * np.sqrt(.04 / pp.gust_tau), 3)
            if i < 100 or kind == "pd":
                throttle, rotation = design_sim.baseline_controller(state, traj[i], velocity[i])
                action = np.r_[throttle, legacy.matrix_to_quat_batch(rotation[None])[0]]
            else:
                if dev.type == "cuda":
                    torch.cuda.synchronize(dev)
                tic = time.monotonic()
                nominal = legacy.pd_plan(design_sim, state, traj, velocity, acceleration, i, horizon)
                if kind == "preview_pd":
                    action = nominal[0]
                else:
                    if kind in ("nominal", "true_mean"):
                        dynamics.state = {key: value.copy() for key, value in state.items()}
                        dynamics.t = i * .02
                    action = controller.plan(nominal, state["p"], traj[i+1:i+1+horizon],
                                             velocity[i+1:i+1+horizon])
                if dev.type == "cuda":
                    torch.cuda.synchronize(dev)
                latencies.append(time.monotonic() - tic)
            if not np.isfinite(action).all():
                raise NonfinitePrediction("nonfinite_action")
            actions.append(action.tolist())
            if kind == "model":
                dynamics.observe_action(action)
            rotation = legacy.quat_to_matrix_batch(action[None, 1:])[0]
            state = plant.step(state, float(action[0]), rotation,
                               np.array([row["wind"], 0., 0.]) + gust)
            failure = scoring.state_failure(state, traj[i+1])
            if i >= 100:
                error = np.linalg.norm(state["p"] - traj[i+1])
                errors.append(float(error) if np.isfinite(error) else None)
                effort_rows.append([float(action[0] - .3924),
                                    float(np.arccos(np.clip(rotation[2, 2], -1., 1.)))])
            if failure:
                failed_step = i
                break
        except NonfinitePrediction as exc:
            failure, failed_step = str(exc), i
            break
    clean_errors = [np.nan if value is None else value for value in errors]
    result = scoring.score(clean_errors, failure is not None, steps=scored_steps)
    effort = np.asarray(effort_rows)
    result.update(row=row, failure_reason=failure, failure_step=failed_step,
                  errors_m=errors, actions=actions, planning_seconds=latencies,
                  warmup_state=warmup_state, warmup_seconds=warmup_seconds,
                  elapsed_seconds=time.monotonic()-start, observed_effort_steps=len(effort_rows),
                  throttle_deviation_rms=(float(np.sqrt(np.mean(effort[:, 0] ** 2)))
                                          if len(effort) else None),
                  command_tilt_rms_rad=(float(np.sqrt(np.mean(effort[:, 1] ** 2)))
                                       if len(effort) else None),
                  planning_mean_seconds=float(np.mean(latencies)) if latencies else None,
                  planning_p95_seconds=float(np.quantile(latencies, .95)) if latencies else None,
                  plant_parameters=asdict(plant_params), controller_parameters=asdict(design_params),
                  device=str(dev), torch_version=torch.__version__, numpy_version=np.__version__)
    return result


def execute(protocol_path: Path, index: int, checkpoint_root: Path, output_root: Path,
            device="cpu") -> Path:
    """Preflight then claim exactly one immutable episode, preserving software failures."""
    roster = json.loads(protocol_path.read_text())
    protocol.validate_protocol(roster)
    if not 0 <= index < len(roster["rows"]):
        raise ValueError("index outside frozen roster")
    source = os.environ.get("LATENT_WAM_SOURCE_COMMIT", "")
    if not re.fullmatch(r"[0-9a-f]{40}", source):
        raise ValueError("LATENT_WAM_SOURCE_COMMIT must name the immutable execution release")
    row = roster["rows"][index]
    checkpoint = identity = None
    if row["kind"] == "model":
        entry = roster["checkpoint_manifest"]["checkpoints"][row["checkpoint_index"]]
        checkpoint = (checkpoint_root / entry["checkpoint_relpath"]).resolve()
        if not checkpoint.is_relative_to(checkpoint_root.resolve()):
            raise ValueError("checkpoint escaped root through symlink")
        identity = check_checkpoint(checkpoint, entry)
    protocol_digest = protocol.file_hash(protocol_path)
    directory = output_root / roster["phase"] / roster["family_id"] / protocol_digest
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / f"episode_{index:05d}.json"
    metadata = dict(schema=protocol.SCHEMA, phase=roster["phase"], family_id=roster["family_id"],
                    row=row, source_commit=source, protocol_sha256=protocol_digest,
                    source_sha256=roster["source_sha256"],
                    checkpoint_sha256=identity["checkpoint_sha256"] if identity else None,
                    checkpoint_validation=identity,
                    started_utc=datetime.now(timezone.utc).isoformat())
    if output.exists():
        raise FileExistsError(output)
    with output.with_suffix(".claim").open("x") as stream:
        stream.write(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    try:
        result = run(row, checkpoint, device, scored_steps=roster["scored_steps"])
        result.update(metadata)
        with output.open("x") as stream:
            stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    except Exception as exc:
        with output.with_suffix(".error").open("x") as stream:
            stream.write(json.dumps(dict(metadata, software_failure=type(exc).__name__,
                                        detail=str(exc)), indent=2) + "\n")
        raise
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    output = execute(args.protocol, args.index, args.checkpoint_root, args.output_root, args.device)
    print(json.dumps(dict(output=str(output), index=args.index)))


if __name__ == "__main__":
    main()
