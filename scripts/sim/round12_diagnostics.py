"""Auditable stochastic planning diagnostics on the unchanged Round11B models.

Each A1 job scores all 60 models at one common first decision. Each A2 job
reconstructs one original donor planner through three anchors and cross-scores
the four matched models. This module never trains or modifies legacy code.
"""
from __future__ import annotations

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
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO/'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import closed_loop as legacy  # noqa: E402
import round11_closed_loop as original_runner  # noqa: E402
import round11_closed_loop_protocol as original_protocol  # noqa: E402
import round12_diagnostic_protocol as protocol  # noqa: E402

from latent_aero_wam.data.normalize import Normalizer  # noqa: E402
from latent_aero_wam.sim import QuadrotorSim, SimParams, random_trajectory  # noqa: E402
from latent_aero_wam.sim.quadrotor import GRAVITY  # noqa: E402
from latent_aero_wam.utils.rotation import so3_exp_np, so3_log_np  # noqa: E402


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def planner_cost(v, p0, p_ref, v_ref, pert, sigma, reg=.05, position=None):
    """Original, unaveraged MPPI objective; leading axes may include MC draws."""
    p = p0 + protocol.DT*np.cumsum(v, axis=-2) if position is None else position
    cost = (np.linalg.norm(p-p_ref, axis=-1)**2).sum(-1)
    cost += .2*(np.linalg.norm(v-v_ref, axis=-1)**2).sum(-1)
    cost += reg*((pert/sigma)**2).sum((-1, -2))/v.shape[-2]
    return cost


def softmin_solution(cost, pert, lam=.05):
    if not np.isfinite(cost).all():
        raise original_runner.NonfinitePrediction('nonfinite_candidate_cost')
    weights = np.exp(-(cost-cost.min())/(lam*(cost.std()+1e-9)))
    weights /= weights.sum()
    return weights, (weights[:, None, None]*pert).sum(0)


class TraceMPPI(legacy.MPPI):
    """Exact legacy perturbations/objective, with a retained planning trace."""
    def sample_bank(self, nominal):
        h = len(nominal)
        prev = (np.concatenate([self.prev[1:], np.zeros((1, 3))], axis=0)
                if self.prev is not None else np.zeros((h, 3)))
        eps = self.rng.normal(0, 1, (self.n, h, 3))
        for t in range(1, h):
            eps[:, t] = .7*eps[:, t-1]+.3*eps[:, t]
        pert = prev[None]+self.sigma*eps
        pert[0] = 0.
        pert[1] = prev
        return dict(nominal=nominal.copy(), shifted_prev=prev, perturbations=pert,
                    actions=self.apply(nominal, pert))

    def plan(self, nominal, p0, p_ref, v_ref):
        bank = self.sample_bank(nominal)
        v = self.predict(bank['actions'])
        cost = planner_cost(v, p0, p_ref, v_ref, bank['perturbations'], self.sigma, self.reg)
        weights, sol = softmin_solution(cost, bank['perturbations'], self.lam)
        self.prev = sol
        self.trace = dict(**bank, predicted_velocity=v, predicted_cost=cost,
                          weights=weights, solution_perturbation=sol)
        return self.apply(nominal, sol[None])[0, 0]


def empty_history():
    # Reuse the exact legacy observe_state/delta alignment without loading a model.
    h = legacy.LearnedDynamics.__new__(legacy.LearnedDynamics)
    h.states, h.v_bodies, h.Rs, h.actions, h.deltas = [], [], [], [], []
    return h


def history_arrays(history):
    k = protocol.HISTORY
    if not history.ready():
        raise ValueError('history is not ready')
    return {name: np.asarray(getattr(history, name)[-n:]).copy()
            for name, n in [('states', k+1), ('v_bodies', k+1), ('Rs', k+1),
                            ('actions', k), ('deltas', k)]}


def model_batch(history, actions, device):
    b = len(actions)
    def rep(a):
        # Match the legacy adapter's materialized layout and host-to-device order.
        x = torch.tensor(np.asarray(a, dtype=np.float32))
        return x.unsqueeze(0).repeat(b, *([1]*x.ndim)).to(device)
    return dict(history_state=rep(history['states'][:-1]),
                history_v_body=rep(history['v_bodies'][:-1]),
                history_action=rep(history['actions']), history_delta_state=rep(history['deltas']),
                current_state=rep(history['states'][-1]), current_v_body=rep(history['v_bodies'][-1]),
                current_R=rep(history['Rs'][-1]),
                future_action=torch.as_tensor(actions.astype(np.float32), device=device))


@torch.no_grad()
def predict(model, history, actions, device):
    out = model(model_batch(history, actions, device))
    states, rotations = out.state.cpu().numpy(), out.R.cpu().numpy()
    if not np.isfinite(states).all() or not np.isfinite(rotations).all():
        raise original_runner.NonfinitePrediction('nonfinite_prediction')
    return states, rotations


def replay(record, anchor_steps, on_anchor, donor_predict=None, action_atol=protocol.ACTION_ATOL):
    """Saved actions recover physical histories; all intervening plans recover prev.

    Snapshots occur before this period's OU innovation. No future realized action
    sequence is used as a candidate. The original actions only reconstruct history.
    """
    row = record['row']
    sim = QuadrotorSim(seed=row['environment_seed'])
    traj = random_trajectory(np.random.default_rng(row['trajectory_seed']), 33.02, .02)
    velocity = np.gradient(traj, .02, axis=0)
    acceleration = np.gradient(velocity, .02, axis=0)
    state = dict(p=traj[0].copy(), v=np.zeros(3), R=np.eye(3), w=np.zeros(3))
    gust = np.zeros(3)
    history = empty_history()
    ctrl = None
    if donor_predict is not None:
        ctrl = TraceMPPI(lambda actions: donor_predict(history_arrays(history), actions),
                         n_samples=64, rng=np.random.default_rng(row['planning_seed']))
    max_action_error = max_error_error = max_warmup_error = 0.
    for i in range(max(anchor_steps)+1):
        if i >= len(record['actions']):
            raise ValueError('source episode does not reach registered anchor')
        history.observe_state(state['v'], state['R'], state['w'])
        if i == 100:
            max_warmup_error = max(float(np.max(np.abs(state[k]-record['warmup_state'][k]))) for k in state)
            if max_warmup_error > 1e-10:
                raise ValueError('saved warmup state does not replay')
        bank = None
        if i >= 100 and (ctrl is not None or i in anchor_steps):
            nominal = legacy.pd_plan(sim, state, traj, velocity, acceleration, i, 50)
            if ctrl is not None:
                action = ctrl.plan(nominal, state['p'], traj[i+1:i+51], velocity[i+1:i+51])
                error = float(np.max(np.abs(action-np.asarray(record['actions'][i]))))
                max_action_error = max(max_action_error, error)
                if error > action_atol:
                    raise ValueError(f'original planner action mismatch at {i}: {error}')
                bank = ctrl.trace
            else:
                if i != 100:
                    raise ValueError('later original banks require full donor planner reconstruction')
                first = TraceMPPI(None, n_samples=64, rng=np.random.default_rng(row['planning_seed']))
                bank = first.sample_bank(nominal)
        if i in anchor_steps:
            snapshot = dict(step=i, state={k: v.copy() for k, v in state.items()},
                            gust=gust.copy(), history=history_arrays(history),
                            p_ref=traj[i+1:i+51].copy(), v_ref=velocity[i+1:i+51].copy())
            on_anchor(snapshot, bank)
        if i == max(anchor_steps):
            break
        pp = sim.p_
        gust += -.02*gust/pp.gust_tau + sim.rng.normal(
            0, pp.gust_std*max(row['wind'], .3)*np.sqrt(.04/pp.gust_tau), 3)
        action = np.asarray(record['actions'][i])
        history.observe_action(action)
        state = sim.step(state, float(action[0]), legacy.quat_to_matrix_batch(action[None, 1:])[0],
                         np.array([row['wind'], 0., 0.])+gust)
        if i >= 100:
            saved = record['errors_m'][i-100]
            error = float(np.linalg.norm(state['p']-traj[i+1]))
            if saved is None or not np.isfinite(error):
                raise ValueError('nonfinite source trajectory before anchor')
            max_error_error = max(max_error_error, abs(error-saved))
            if abs(error-saved) > 1e-9:
                raise ValueError('saved tracking trace does not replay')
    return dict(max_action_abs_error=max_action_error, max_tracking_abs_error=max_error_error,
                max_warmup_abs_error=max_warmup_error,
                plans_reconstructed=max(anchor_steps)-99 if ctrl is not None else 0)


def noise_bundle(seed_words, samples=8, horizon=50, substeps=10):
    """Explicit standard-normal exogenous draws, shared across all actions/models."""
    rng = np.random.default_rng(np.random.SeedSequence(seed_words))
    # Per period order agrees with the environment: OU, throttle, then body-rate.
    raw = rng.normal(size=(samples, horizon, 3+1+3*substeps))
    return dict(gust=raw[..., :3], throttle=raw[..., 3],
                rate=raw[..., 4:].reshape(samples, horizon, substeps, 3))


def simulate(snapshot, actions, noise, wind, params=None):
    """Vectorized exact step equations; outer axis = noise, second = action bank."""
    pp = params or SimParams()
    n, b, horizon = len(noise['gust']), len(actions), actions.shape[1]
    def spread(value):
        return np.broadcast_to(value, (n, b)+value.shape).copy()
    state = snapshot['state']
    p, v, R, w = [spread(state[k]) for k in ('p', 'v', 'R', 'w')]
    gust = np.broadcast_to(snapshot['gust'], (n, 3)).copy()
    positions, velocities, rotations, rates = [], [], [], []
    hh = pp.dt/pp.n_substeps
    for j in range(horizon):
        gust += -pp.dt*gust/pp.gust_tau + noise['gust'][:, j]*pp.gust_std*max(wind, .3)*np.sqrt(2*pp.dt/pp.gust_tau)
        total_wind = np.array([wind, 0., 0.])+gust[:, None, :]
        t_sp = np.clip(actions[None, :, j, 0]+pp.act_noise*noise['throttle'][:, j, None], 0., 1.)
        R_sp = legacy.quat_to_matrix_batch(actions[:, j, 1:])
        for sub in range(pp.n_substeps):
            w_cmd = pp.k_att*so3_log_np(np.swapaxes(R, -2, -1)@R_sp)
            w_cmd += pp.act_noise*noise['rate'][:, j, sub, None, :]
            w = w+hh*(w_cmd-w)/pp.tau_omega
            R = R@so3_exp_np(w*hh)
            thrust = t_sp[..., None]*pp.t_max*R[..., :, 2]
            v_air = np.einsum('...ji,...j->...i', R, v-total_wind)
            f_body = -(np.asarray(pp.drag_lin)+np.asarray(pp.drag_quad)*np.abs(v_air))*v_air
            f = thrust+np.einsum('...ij,...j->...i', R, f_body)+pp.mass*GRAVITY
            v = v+hh*f/pp.mass
            p = p+hh*v
        positions.append(p.copy())
        velocities.append(v.copy())
        rotations.append(R.copy())
        rates.append(w.copy())
    result = dict(p=np.stack(positions, 2), v=np.stack(velocities, 2),
                  R=np.stack(rotations, 2), w=np.stack(rates, 2))
    if any(not np.isfinite(a).all() for a in result.values()):
        raise FloatingPointError('nonfinite_simulator_branch')
    result['p_planner'] = state['p']+pp.dt*np.cumsum(result['v'], axis=2)
    result['threshold_exceeded'] = ((np.linalg.norm(result['p']-snapshot['p_ref'], axis=-1)>5)
        | (np.linalg.norm(result['p'], axis=-1)>20) | (np.linalg.norm(result['v'], axis=-1)>30)
        | (np.linalg.norm(result['w'], axis=-1)>60)).any(-1)
    return result


def costs(truth, snapshot, pert):
    args = (truth['v'], snapshot['state']['p'], snapshot['p_ref'], snapshot['v_ref'],
            pert, np.array([.04, .08, .08]))
    return dict(physical=planner_cost(*args, position=truth['p']), planner=planner_cost(*args))


def ranking_metrics(predicted, truth_mean):
    if np.ptp(predicted) == 0 or np.ptp(truth_mean) == 0:
        rank = None
    else:
        rank = float(spearmanr(predicted, truth_mean).statistic)
    left, right = np.triu_indices(len(predicted), 1)
    actual_diff = truth_mean[left]-truth_mean[right]
    identifiable = np.abs(actual_diff)>1e-12
    wrong = (predicted[left]-predicted[right])*actual_diff < 0
    best = int(np.argmin(predicted))
    return dict(spearman=rank, pairwise_inversion_rate=float(wrong[identifiable].mean()) if identifiable.any() else None,
                predicted_best_candidate=best, empirical_candidate_regret=float(truth_mean[best]-truth_mean.min()),
                cost_mae=float(np.mean(np.abs(predicted-truth_mean))))


def horizon_metrics(states, rotations, truth, snapshot, scales):
    v, w = states[..., :3], states[..., 9:]
    p = snapshot['state']['p']+protocol.DT*np.cumsum(v, axis=1)
    velocity_e = (np.abs(v-truth['v'])/np.maximum(scales['velocity'], 1e-6)).mean(-1)
    rate_e = (np.abs(w-truth['w'])/np.maximum(scales['angular_rate'], 1e-6)).mean(-1)
    rel = np.swapaxes(rotations, -2, -1)@truth['R']
    angle = np.arccos(np.clip((np.trace(rel, axis1=-2, axis2=-1)-1)/2, -1+1e-7, 1-1e-7))
    orientation_e = angle/max(float(scales['orientation'][0]), 1e-6)
    disp_e = (np.abs(p-truth['p_planner'])/np.maximum(scales['displacement'], 1e-6)).mean(-1)
    arrays = dict(original_E=(velocity_e+rate_e+orientation_e)/3, original_velocity_E=velocity_e,
                  original_orientation_E=orientation_e, original_angular_rate_E=rate_e,
                  original_displacement_E=disp_e, velocity_l2_mps=np.linalg.norm(v-truth['v'], axis=-1),
                  physical_position_l2_m=np.linalg.norm(p-truth['p'], axis=-1),
                  planner_position_l2_m=np.linalg.norm(p-truth['p_planner'], axis=-1))
    result = {}
    for h in (1, 5, 10, 25, 50):
        if h > states.shape[1]:
            continue
        result[str(h)] = {k: dict(terminal=float(a[..., h-1].mean()),
                                  prefix_mean=float(a[..., :h].mean())) for k, a in arrays.items()}
        result[str(h)]['velocity_error_vs_ensemble_mean_mps'] = float(np.linalg.norm(v[:, h-1]-truth['v'].mean(0)[:, h-1], axis=-1).mean())
    return result


def flatten_arrays(prefix, value, target):
    for key, item in value.items():
        name = f'{prefix}/{key}'
        if isinstance(item, dict):
            flatten_arrays(name, item, target)
        elif isinstance(item, np.ndarray):
            target[name] = item


def evaluate_anchor(snapshot, bank, model_specs, load_model, row, mc_samples=8):
    """model_specs are matched original roster rows; truth is shared across them."""
    seeds = {phase: protocol.noise_seed_words(row['stage'], row['index'], snapshot['step'], phase)
             for phase in ('construct', 'evaluate')}
    noise = {p: noise_bundle(s, mc_samples) for p, s in seeds.items()}
    construction = simulate(snapshot, bank['actions'], noise['construct'], row['wind'])
    construction_cost = costs(construction, snapshot, bank['perturbations'])
    oracle_weights, oracle_pert = softmin_solution(construction_cost['physical'].mean(0), bank['perturbations'])
    sigma = np.array([.04, .08, .08])
    traces = {}
    flatten_arrays('snapshot', snapshot, traces)
    flatten_arrays('bank', bank, traces)
    flatten_arrays('noise', noise, traces)
    flatten_arrays('construction', construction, traces)
    flatten_arrays('construction_cost', construction_cost, traces)
    traces['oracle/weights'] = oracle_weights
    traces['oracle/perturbation'] = oracle_pert
    records, solutions, valid = [], [oracle_pert], []
    for model_index, spec in enumerate(model_specs):
        model, scales, identity = load_model(spec)
        result = dict(model=spec['model'], dataset_replicate=spec['dataset_replicate'],
                      model_seed=spec['model_seed'], **identity)
        try:
            states, rotations = predict(model, snapshot['history'], bank['actions'], next(model.parameters()).device)
            predicted_cost = planner_cost(states[..., :3], snapshot['state']['p'], snapshot['p_ref'], snapshot['v_ref'], bank['perturbations'], sigma)
            weights, sol = softmin_solution(predicted_cost, bank['perturbations'])
            result.update(status='complete', solution_index=len(solutions),
                          weights_ess=float(1/np.sum(weights**2)),
                          weights_entropy=float(-np.sum(weights*np.log(np.maximum(weights, 1e-300)))),
                          first_action=legacy.MPPI.apply(bank['nominal'], sol[None])[0, 0].tolist())
            if snapshot['step'] == 100:
                source_action = identity.get('source_first_action')
                if source_action is not None:
                    err = float(np.max(np.abs(np.asarray(result['first_action'])-source_action)))
                    result['first_action_reconstruction_abs_error'] = err
                    if err > protocol.ACTION_ATOL:
                        raise ValueError(f'first-decision reconstruction mismatch: {spec["model"]}: {err}')
            flatten_arrays(f'model_{model_index}', dict(state=states, R=rotations, predicted_cost=predicted_cost,
                           weights=weights, solution_perturbation=sol), traces)
            valid.append((len(records), states, rotations, predicted_cost, scales))
            solutions.append(sol)
        except original_runner.NonfinitePrediction as exc:
            result.update(status='nonfinite_prediction', failure_reason=str(exc))
        records.append(result)
    solutions = np.stack(solutions)
    # Independent evaluation draws prevent optimistic scoring of the constructed oracle.
    all_pert = np.concatenate([bank['perturbations'], solutions], axis=0)
    actions = legacy.MPPI.apply(bank['nominal'], all_pert)
    evaluation = simulate(snapshot, actions, noise['evaluate'], row['wind'])
    evaluation_cost = costs(evaluation, snapshot, all_pert)
    b = len(bank['actions'])
    candidates = {k: a[:, :b] for k, a in evaluation.items()}
    for index, states, rotations, predicted_cost, scales in valid:
        result = records[index]
        solution_index = b+result['solution_index']
        for label in ('physical', 'planner'):
            solution_cost = evaluation_cost[label][:, solution_index]
            gap = solution_cost-evaluation_cost[label][:, b]
            result[f'{label}_solution_cost'] = float(solution_cost.mean())
            result[f'{label}_solution_cost_excess_vs_oracle_softmin'] = float(gap.mean())
            result[f'{label}_solution_cost_excess_mc_se'] = float(gap.std(ddof=1)/np.sqrt(mc_samples)) if mc_samples>1 else None
            result[f'{label}_candidate_ranking'] = ranking_metrics(predicted_cost, evaluation_cost[label][:, :b].mean(0))
        result['solution_threshold_exceedance_rate'] = float(evaluation['threshold_exceeded'][:, solution_index].mean())
        result['horizon_metrics'] = horizon_metrics(states, rotations, candidates, snapshot, scales)
    flatten_arrays('evaluation', evaluation, traces)
    flatten_arrays('evaluation_cost', evaluation_cost, traces)
    traces['evaluation/actions'] = actions
    traces['evaluation/perturbations'] = all_pert
    return dict(step=snapshot['step'], model_count=len(records), models=records,
                mc_samples=mc_samples, noise_seed_words=seeds,
                oracle_physical_cost=float(evaluation_cost['physical'][:, b].mean()),
                oracle_planner_cost=float(evaluation_cost['planner'][:, b].mean()),
                oracle_threshold_exceedance_rate=float(evaluation['threshold_exceeded'][:, b].mean()),
                candidate_threshold_exceedance_rate=float(candidates['threshold_exceeded'].mean())), traces


def read_episode(root, index):
    path = Path(root)/f'episode_{index:04d}.json'
    record = json.loads(path.read_text())
    if record['row'] != original_protocol.design()['rows'][index]:
        raise ValueError('original episode roster mismatch')
    if record['roster_sha256'] != sha256(REPO/'configs/sim/round11_closed_loop.json'):
        raise ValueError('original roster hash mismatch')
    if record['failed'] or len(record['actions']) != 1600 or len(record['errors_m']) != 1500:
        raise ValueError('expected the accepted complete Round11D trajectory')
    return record, dict(path=str(path), sha256=sha256(path), source_commit=record['source_commit'])


class ModelCache:
    def __init__(self, checkpoint_root, episode_root, device):
        self.root, self.episodes, self.device = checkpoint_root, episode_root, torch.device(device)
        self.cache = {}

    def __call__(self, row):
        key = (row['model'], row['model_seed'])
        if key not in self.cache:
            checkpoint = protocol.checkpoint_path(self.root, row)
            digest = original_runner.check_checkpoint(checkpoint, row)
            dyn = legacy.LearnedDynamics(checkpoint, self.device)
            scales = Normalizer.load(REPO/f"artifacts/round11b_v1/data{row['dataset_replicate']}_norm.json").metric_scales
            self.cache[key] = (dyn.model, scales, dict(checkpoint_path=str(checkpoint), checkpoint_sha256=digest))
        model, scales, identity = self.cache[key]
        source, source_id = read_episode(self.episodes, row['index'])
        if source['checkpoint_sha256'] != identity['checkpoint_sha256']:
            raise ValueError('checkpoint hash differs from the original control episode')
        return model, scales, dict(identity, source_episode=source_id,
                                    source_first_action=source['actions'][100])


def run(row, checkpoint_root, episode_root, device='cpu', mc_samples=8):
    donor, donor_id = read_episode(episode_root, row['donor_index'])
    cache = ModelCache(checkpoint_root, episode_root, device)
    model_specs = protocol.model_rows(row['wind'], row['episode'],
        row.get('dataset_replicate'), row.get('model_seed'))
    donor_predict = None
    if row['stage'] == 'a2':
        donor_model, _, _ = cache(donor['row'])
        def donor_predict(history, actions):
            return predict(donor_model, history, actions, cache.device)[0][..., :3]
    results, traces = [], {}
    start = time.monotonic()
    def on_anchor(snapshot, bank):
        result, trace = evaluate_anchor(snapshot, bank, model_specs, cache, row, mc_samples)
        results.append(result)
        traces.update({f"anchor_{snapshot['step']}/{k}": a for k, a in trace.items()})
    checks = replay(donor, row['anchor_steps'], on_anchor, donor_predict)
    if cache.device.type == 'cuda':
        torch.cuda.synchronize(cache.device)
    return dict(status='complete', row=row, donor_episode=donor_id, replay_checks=checks,
                anchors=results, device=str(cache.device), torch_version=torch.__version__,
                numpy_version=np.__version__, elapsed_seconds=time.monotonic()-start), traces


def validate_result(result, trace_path):
    design = protocol.design()
    row = result['row']
    if row != design[row['stage']][row['index']] or result['protocol_sha256'] != protocol.canonical_hash(design):
        raise ValueError('diagnostic roster/protocol mismatch')
    if result['status'] != 'complete' or result['trace_sha256'] != sha256(trace_path):
        raise ValueError('incomplete result or corrupt trace')
    checks = result['replay_checks']
    if (checks['max_action_abs_error'] > protocol.ACTION_ATOL
            or checks['max_tracking_abs_error'] > 1e-9 or checks['max_warmup_abs_error'] > 1e-10
            or checks['plans_reconstructed'] != (max(row['anchor_steps'])-99 if row['stage'] == 'a2' else 0)):
        raise ValueError('replay reconstruction checks failed')
    expected_models = 60 if row['stage'] == 'a1' else 4
    if [a['step'] for a in result['anchors']] != row['anchor_steps']:
        raise ValueError('wrong anchor coverage')
    expected_specs = protocol.model_rows(row['wind'], row['episode'], row.get('dataset_replicate'), row.get('model_seed'))
    expected = {(r['model'], r['model_seed']) for r in expected_specs}
    with np.load(trace_path, allow_pickle=False) as trace:
        for anchor in result['anchors']:
            if len(anchor['models']) != expected_models or {(m['model'], m['model_seed']) for m in anchor['models']} != expected:
                raise ValueError('wrong model coverage')
            if not result['pilot'] and anchor['mc_samples'] != protocol.MC_SAMPLES:
                raise ValueError('formal MC budget changed')
            prefix = f"anchor_{anchor['step']}"
            samples = anchor['mc_samples']
            for phase in ('construct', 'evaluate'):
                if anchor['noise_seed_words'][phase] != protocol.noise_seed_words(row['stage'], row['index'], anchor['step'], phase):
                    raise ValueError('changed independent noise stream')
                if trace[f'{prefix}/noise/{phase}/gust'].shape != (samples, 50, 3):
                    raise ValueError('wrong Monte Carlo trace shape')
            if trace[f'{prefix}/bank/actions'].shape != (64, 50, 5):
                raise ValueError('wrong candidate bank shape')
            complete = [m for m in anchor['models'] if m['status'] == 'complete']
            physical_cost = trace[f'{prefix}/evaluation_cost/physical']
            if physical_cost.shape != (samples, 65+len(complete)) or not np.isfinite(physical_cost).all():
                raise ValueError('invalid physical evaluation trace')
            if sorted(m['solution_index'] for m in complete) != list(range(1, len(complete)+1)):
                raise ValueError('invalid weighted-solution indexing')
            for model in anchor['models']:
                if model['status'] not in ('complete', 'nonfinite_prediction'):
                    raise ValueError('unknown model status')
                if model['status'] == 'complete':
                    for field in ('physical_solution_cost', 'physical_solution_cost_excess_vs_oracle_softmin'):
                        if not np.isfinite(model[field]):
                            raise ValueError('missing finite completed-model metric')
                    gap = (physical_cost[:, 64+model['solution_index']]-physical_cost[:, 64]).mean()
                    if not np.isclose(gap, model['physical_solution_cost_excess_vs_oracle_softmin'], rtol=1e-12, atol=1e-12):
                        raise ValueError('reported signed cost excess disagrees with trace')
    return dict(accepted=True, stage=row['stage'], index=row['index'], pilot=result['pilot'],
                anchors=len(result['anchors']), models_per_anchor=expected_models)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage', choices=['a1', 'a2'])
    ap.add_argument('--index', type=int)
    ap.add_argument('--checkpoint-root', type=Path)
    ap.add_argument('--episode-root', type=Path)
    ap.add_argument('--output', type=Path)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--mc-samples', type=int, default=8)
    ap.add_argument('--pilot', action='store_true')
    ap.add_argument('--validate', type=Path, help='Validate an existing result JSON and bound NPZ.')
    args = ap.parse_args()
    if args.validate:
        result = json.loads(args.validate.read_text())
        print(json.dumps(validate_result(result, args.validate.with_suffix('.npz'))))
        return
    if any(getattr(args, key) is None for key in ('stage', 'index', 'checkpoint_root', 'episode_root', 'output')):
        ap.error('stage, index, checkpoint-root, episode-root, and output are required')
    if args.mc_samples < 2 or (not args.pilot and args.mc_samples != 8):
        ap.error('formal MC samples must equal 8; pilots require at least 2')
    design = protocol.design()
    if not 0 <= args.index < len(design[args.stage]):
        ap.error('index outside locked roster')
    row = design[args.stage][args.index]
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output/f"{args.stage}_{args.index:04d}.json"
    identity = dict(protocol_sha256=protocol.canonical_hash(design), pilot=args.pilot,
        source_commit=os.environ.get('LATENT_WAM_SOURCE_COMMIT', 'unrecorded'),
        source_file_sha256={str(p.relative_to(REPO)): sha256(p) for p in [Path(__file__),
            Path(protocol.__file__), REPO/'scripts/sim/closed_loop.py',
            REPO/'scripts/sim/round11_closed_loop.py', REPO/'scripts/sim/round11_closed_loop_protocol.py',
            REPO/'src/latent_aero_wam/sim/quadrotor.py', REPO/'src/latent_aero_wam/utils/rotation.py',
            REPO/'src/latent_aero_wam/models/base.py', REPO/'src/latent_aero_wam/models/context_controls.py',
            REPO/'src/latent_aero_wam/evaluation/evaluator.py']})
    with path.with_suffix('.claim').open('x') as stream:
        json.dump(dict(row=row, **identity), stream, indent=2)
    try:
        result, traces = run(row, args.checkpoint_root, args.episode_root, args.device, args.mc_samples)
        with path.with_suffix('.npz').open('xb') as stream:
            np.savez_compressed(stream, **traces)
        result.update(identity, trace_sha256=sha256(path.with_suffix('.npz')))
        validate_result(result, path.with_suffix('.npz'))
        with path.open('x') as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write('\n')
        print(json.dumps(dict(status='complete', row=row, elapsed_seconds=result['elapsed_seconds'])))
    except Exception as exc:
        with path.with_suffix('.error.json').open('x') as stream:
            json.dump(dict(status='software_or_truth_error', row=row, exception=repr(exc),
                           traceback=traceback.format_exc(), **identity), stream, indent=2)
        raise


if __name__ == '__main__':
    main()
