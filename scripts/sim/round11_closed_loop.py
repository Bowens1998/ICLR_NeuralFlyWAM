"""Fair Round11D episode executor; immutable roster, shared warmup, retained failures."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO/'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import closed_loop as legacy  # noqa: E402
import round11_closed_loop_protocol as protocol  # noqa: E402

from latent_aero_wam.sim import QuadrotorSim, random_trajectory  # noqa: E402
from latent_aero_wam.training.checkpoint import load_checkpoint  # noqa: E402


class NonfinitePrediction(FloatingPointError):
    pass


def finite_predictor(dyn):
    def predict(actions):
        if not np.isfinite(actions).all():
            raise NonfinitePrediction('nonfinite_candidate_action')
        result = dyn(actions)
        if not np.isfinite(result).all():
            raise NonfinitePrediction('nonfinite_prediction')
        return result
    return predict


def check_checkpoint(path, row):
    state = load_checkpoint(path, map_location='cpu')
    p = state['provenance']
    assert p['git_commit'] == '275afdc26016a892f58b55edc94d688da5745784'
    assert p['model_name'] == row['model'] and p['seed'] == row['model_seed']
    d = row['dataset_replicate']
    a = json.loads((REPO/'artifacts/round11b_v1/acceptance.json').read_text())['replicates'][d]
    assert p['split_checksum'] == a['split_checksum']
    assert p['normalizer_sha256'] == a['normalizer_sha256']
    assert state['config']['model']['readout_norm'] == 'none'
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(row, checkpoint=None, device='cpu', *, scored_steps=1500, dynamics_factory=None):
    """Short scored_steps/factory overrides are for unit tests, never CLI evaluation."""
    dev = torch.device(device)
    sim = QuadrotorSim(seed=row['environment_seed'])
    traj = random_trajectory(np.random.default_rng(row['trajectory_seed']), 33.02, .02)
    velocity = np.gradient(traj, .02, axis=0)
    acceleration = np.gradient(velocity, .02, axis=0)
    state = dict(p=traj[0].copy(), v=np.zeros(3), R=np.eye(3), w=np.zeros(3))
    gust = np.zeros(3)
    kind = row['kind']
    dyn = ctrl = None
    if kind == 'model':
        dyn = (dynamics_factory or legacy.LearnedDynamics)(checkpoint, dev)
    elif kind in ['nominal', 'true_mean']:
        mean = row['wind'] if kind == 'true_mean' else 0.
        dyn = legacy.TruthDynamics(sim, lambda t: np.array([mean, 0., 0.]))
    if dyn is not None:
        ctrl = legacy.MPPI(finite_predictor(dyn), n_samples=64,
                           rng=np.random.default_rng(row['planning_seed']))
    errors, actions, latencies, scored_actions = [], [], [], []
    failure = None
    failed_step = None
    warmup_state = None
    start = time.monotonic()
    warmup_seconds = None
    for i in range(100+scored_steps):
        try:
            if kind == 'model':
                dyn.observe_state(state['v'], state['R'], state['w'])
            if i == 100:
                warmup_state = {k: v.tolist() for k, v in state.items()}
                warmup_seconds = time.monotonic()-start
                if kind == 'model':
                    assert dyn.ready(), 'history contract violated'
            pp = sim.p_
            gust += -.02*gust/pp.gust_tau + sim.rng.normal(
                0, pp.gust_std*max(row['wind'], .3)*np.sqrt(.04/pp.gust_tau), 3)
            if i < 100 or kind == 'pd':
                throttle, rotation = sim.baseline_controller(state, traj[i], velocity[i])
                action = np.r_[throttle, legacy.matrix_to_quat_batch(rotation[None])[0]]
            else:
                if dev.type == 'cuda':
                    torch.cuda.synchronize(dev)
                tic = time.monotonic()
                nominal = legacy.pd_plan(sim, state, traj, velocity, acceleration, i, 50)
                if kind == 'preview_pd':
                    action = nominal[0]
                else:
                    if kind in ['nominal', 'true_mean']:
                        dyn.state = {k: v.copy() for k, v in state.items()}
                        dyn.t = i*.02
                    action = ctrl.plan(nominal, state['p'], traj[i+1:i+51], velocity[i+1:i+51])
                if dev.type == 'cuda':
                    torch.cuda.synchronize(dev)
                latencies.append(time.monotonic()-tic)
            if not np.isfinite(action).all():
                raise NonfinitePrediction('nonfinite_action')
            actions.append(action.tolist())
            if kind == 'model':
                dyn.observe_action(action)
            rotation = legacy.quat_to_matrix_batch(action[None, 1:])[0]
            state = sim.step(state, float(action[0]), rotation, np.array([row['wind'], 0., 0.])+gust)
            failure = protocol.state_failure(state, traj[i+1])
            if i >= 100:
                error = np.linalg.norm(state['p']-traj[i+1])
                errors.append(float(error) if np.isfinite(error) else None)
                scored_actions.append([float(action[0]-.3924),
                                       float(np.arccos(np.clip(rotation[2, 2], -1., 1.)))])
            if failure:
                failed_step = i
                break
        except NonfinitePrediction as exc:
            failure, failed_step = str(exc), i
            break
    elapsed = time.monotonic()-start
    clean_errors = [np.nan if e is None else e for e in errors]
    result = protocol.score(clean_errors, failure is not None, steps=scored_steps)
    effort = np.asarray(scored_actions)
    result.update(row=row, failure_reason=failure, failure_step=failed_step,
                  errors_m=errors, actions=actions, planning_seconds=latencies,
                  warmup_state=warmup_state, warmup_seconds=warmup_seconds,
                  elapsed_seconds=elapsed, observed_effort_steps=len(scored_actions),
                  throttle_deviation_rms=(float(np.sqrt(np.mean(effort[:, 0]**2))) if len(effort) else None),
                  command_tilt_rms_rad=(float(np.sqrt(np.mean(effort[:, 1]**2))) if len(effort) else None),
                  planning_mean_seconds=float(np.mean(latencies)) if latencies else None,
                  planning_p95_seconds=float(np.quantile(latencies, .95)) if latencies else None,
                  device=str(dev), torch_version=torch.__version__)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--index', type=int, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--runs-root', type=Path, required=True)
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()
    roster_path = REPO/'configs/sim/round11_closed_loop.json'
    roster = json.loads(roster_path.read_text())
    assert roster == protocol.design()
    assert 0 <= args.index < len(roster['rows'])
    row = roster['rows'][args.index]
    output = args.output/f'episode_{args.index:04d}.json'
    args.output.mkdir(parents=True, exist_ok=True)
    # Exclusive claim survives failures, prevents accidental duplicate evaluation.
    with output.with_suffix('.claim').open('x') as f:
        f.write(json.dumps(dict(row=row, source=os.environ['LATENT_WAM_SOURCE_COMMIT'])))
    if output.exists():
        raise FileExistsError(output)
    checkpoint = None
    checkpoint_hash = None
    if row['kind'] == 'model':
        checkpoint = args.runs_root/f"round11b_data{row['dataset_replicate']}_v1"/f"{row['model']}_seed{row['model_seed']}"/'checkpoint_best.pt'
        checkpoint_hash = check_checkpoint(checkpoint, row)
    result = run(row, checkpoint, args.device)
    result.update(source_commit=os.environ['LATENT_WAM_SOURCE_COMMIT'],
                  checkpoint_sha256=checkpoint_hash,
                  roster_sha256=hashlib.sha256(roster_path.read_bytes()).hexdigest())
    with output.open('x') as f:
        f.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: result[k] for k in ['row', 'failed', 'capped_tracking_rmse_m', 'elapsed_seconds']}))


if __name__ == '__main__':
    main()
