"""Locked Round11D episode roster and failure-aware scoring, before evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DT = .02
HISTORY = 100
HORIZON = 50
SCORED_STEPS = 1500
CANDIDATES = 64
ERROR_CAP_M = 5.
KINDS = ['model', 'nominal', 'true_mean', 'pd', 'preview_pd']


def design():
    episodes = [dict(episode=e, wind=wind, environment_seed=310000+e,
                     trajectory_seed=320000+e, planning_seed=330000+e)
                for wind in [0., 4.9, 6.1, 8.5] for e in range(10)]
    rows = []
    for d in range(5):
        for seed in [70, 71, 72]:
            for update in ['frozen', 'updated']:
                for norm in [0, 1]:
                    for episode in episodes:
                        rows.append(dict(index=len(rows), kind='model', dataset_replicate=d,
                                         model_seed=seed, model=f'r11b_d{d}_{update}_r{norm}',
                                         **episode))
    for kind in KINDS[1:]:
        for episode in episodes:
            rows.append(dict(index=len(rows), kind=kind, **episode))
    return dict(protocol='docs/ROUND11_PROTOCOL.md', dt=DT, history_steps=HISTORY,
                horizon=HORIZON, scored_steps=SCORED_STEPS, candidates=CANDIDATES,
                warmup_controller='pd', trajectory='random_trajectory',
                error_cap_m=ERROR_CAP_M, failure_thresholds=dict(
                    tracking_error_m=5., position_norm_m=20., velocity_norm_mps=30.,
                    body_rate_norm_radps=60.),
                mppi=dict(sigma_t=.04, sigma_tilt=.08, lam=.05, reg=.05, nominal_decay=.85),
                rows=rows)


def state_failure(state, reference):
    if any(not np.isfinite(np.asarray(v)).all() for v in state.values()):
        return 'nonfinite_state'
    for reason, value, limit in [
        ('tracking_error', np.linalg.norm(state['p']-reference), 5.),
        ('position_norm', np.linalg.norm(state['p']), 20.),
        ('velocity_norm', np.linalg.norm(state['v']), 30.),
        ('body_rate_norm', np.linalg.norm(state['w']), 60.),
    ]:
        if value > limit:
            return reason
    return None


def score(errors, failed, steps=SCORED_STEPS):
    """Retain failures; pad unobserved horizon with fixed five-metre penalty."""
    values = np.asarray(errors, dtype=float)
    if values.ndim != 1 or len(values) > steps or np.any(values < 0):
        raise ValueError('invalid error sequence')
    if not failed and (len(values) != steps or not np.isfinite(values).all()):
        raise ValueError('successful episode must have a complete finite horizon')
    if not failed and np.any(values > ERROR_CAP_M):
        raise ValueError('threshold exceedance must be recorded as a failure')
    capped = np.full(steps, ERROR_CAP_M)
    capped[:len(values)] = np.minimum(np.nan_to_num(values, nan=ERROR_CAP_M,
                                                 posinf=ERROR_CAP_M), ERROR_CAP_M)
    return dict(failed=bool(failed), scored_steps=steps, observed_steps=len(values),
                capped_tracking_rmse_m=float(np.sqrt(np.mean(capped**2))),
                successful_tracking_rmse_m=(float(np.sqrt(np.mean(values**2)))
                                            if not failed else None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        f.write(json.dumps(design(), indent=2)+'\n')


if __name__ == '__main__':
    main()
