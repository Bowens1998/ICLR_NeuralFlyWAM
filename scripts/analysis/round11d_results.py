"""Complete-roster Round11D analysis: retained failures, replicate-level contrasts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import t

ARMS = ['frozen_r0', 'frozen_r1', 'updated_r0', 'updated_r1']
WINDS = [0., 4.9, 6.1, 8.5]


def describe(values):
    a = np.asarray(values, dtype=float)
    if a.shape != (5,) or not np.isfinite(a).all():
        raise ValueError('five finite dataset-level paired differences required')
    mean = float(a.mean())
    se = float(a.std(ddof=1)/np.sqrt(5))
    return dict(differences=a.tolist(), mean=mean,
                descriptive_t95ci=list(t.interval(.95, 4, loc=mean, scale=se)) if se else [mean, mean],
                positive_datasets=int((a > 0).sum()), negative_datasets=int((a < 0).sum()))


def aggregate(rows):
    references = {}
    datasets = {}
    contrasts = {}
    for wind in WINDS:
        subset = [r for r in rows if r['row']['wind'] == wind]
        refs = {}
        for kind in ['nominal', 'true_mean', 'pd', 'preview_pd']:
            episodes = [r for r in subset if r['row']['kind'] == kind]
            if len(episodes) != 10 or {r['row']['episode'] for r in episodes} != set(range(10)):
                raise ValueError('exactly ten unique reference episodes required')
            refs[kind] = dict(capped_rmse_m=float(np.mean([r['capped_tracking_rmse_m'] for r in episodes])),
                              failure_rate=float(np.mean([r['failed'] for r in episodes])))
        references[str(wind)] = refs
        data = {metric: {arm: [] for arm in ARMS} for metric in ['capped_rmse_m', 'failure_rate']}
        for d in range(5):
            for arm in ARMS:
                episodes = [r for r in subset if r['row']['kind'] == 'model'
                            and r['row']['dataset_replicate'] == d and r['row']['model'] == f'r11b_d{d}_{arm}']
                keys = {(r['row']['model_seed'], r['row']['episode']) for r in episodes}
                if len(episodes) != 30 or keys != {(s, e) for s in [70, 71, 72] for e in range(10)}:
                    raise ValueError('full model-seed x episode cross required')
                data['capped_rmse_m'][arm].append(float(np.mean([r['capped_tracking_rmse_m'] for r in episodes])))
                data['failure_rate'][arm].append(float(np.mean([r['failed'] for r in episodes])))
        datasets[str(wind)] = data
        contrasts[str(wind)] = {}
        for metric, arms in data.items():
            a = {k: np.asarray(v) for k, v in arms.items()}
            frozen = a['frozen_r1']-a['frozen_r0']
            updated = a['updated_r1']-a['updated_r0']
            effects = dict(frozen_recurrence_LN_effect=frozen, updated_recurrence_LN_effect=updated,
                           recurrence_effect_updated_minus_frozen=updated-frozen,
                           updated_minus_frozen_no_norm=a['updated_r0']-a['frozen_r0'])
            effects.update({arm+'_minus_nominal': values-refs['nominal'][metric] for arm, values in a.items()})
            contrasts[str(wind)][metric] = {name: describe(value) for name, value in effects.items()}
    return dict(references=references, dataset_scores=datasets, contrasts=contrasts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path('runs/results/round11d_formal_v1'))
    ap.add_argument('--acceptance', type=Path, default=Path('runs/evidence/round11d_formal_bundle/all_acceptance.json'))
    ap.add_argument('--output', type=Path, default=Path('reports/ROUND11D_RESULTS.json'))
    args = ap.parse_args()
    acceptance = json.loads(args.acceptance.read_text())
    assert acceptance['group'] == 'all' and acceptance['accepted_episodes'] == 2560
    assert acceptance['cross_group_warmup_check'] is True
    assert {r['index'] for r in acceptance['files']} == set(range(2560))
    rows = []
    for record in acceptance['files']:
        path = args.root/f"episode_{record['index']:04d}.json"
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == record['sha256']
        rows.append(json.loads(raw))
    report = aggregate(rows)
    report.update(protocol='docs/ROUND11_PROTOCOL.md', accepted_episodes=2560,
                  caveat='Descriptive dataset-replicate uncertainty (n=5), conditional on shared ten test '
                         'episodes per wind. References stored once, reused without independent copies. '
                         'All controller failures retained. No p-values or equivalence claim.',
                  episode_diagnostics=[{k: r[k] for k in ['row', 'failed', 'failure_reason',
                    'capped_tracking_rmse_m', 'successful_tracking_rmse_m', 'observed_effort_steps',
                    'throttle_deviation_rms', 'command_tilt_rms_rad', 'planning_mean_seconds',
                    'planning_p95_seconds', 'elapsed_seconds', 'device']} for r in rows])
    with args.output.open('x') as f:
        f.write(json.dumps(report, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
