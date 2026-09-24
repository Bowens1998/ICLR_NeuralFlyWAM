"""Round11B descriptive replication: dataset is the uncertainty unit, no p-values."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import t

ARMS = ['frozen_r0', 'frozen_r1', 'updated_r0', 'updated_r1']
CONDITIONS = ['static_30wind', 'static_50wind', 'static_70wind', 'slow_35wind', 'fast_35wind']
SOURCE = '275afdc26016a892f58b55edc94d688da5745784'


def summarize_replicates(scores):
    if set(scores) != set(ARMS):
        raise ValueError('all four fixed arms required')
    arrays = {k: np.asarray(v, dtype=float) for k, v in scores.items()}
    if any(a.shape != (5,) or not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError('five independent finite dataset means per arm required')
    frozen = arrays['frozen_r1']-arrays['frozen_r0']
    updated = arrays['updated_r1']-arrays['updated_r0']
    contrasts = {}
    for label, values in [('frozen_recurrence_LN_effect', frozen),
                          ('updated_recurrence_LN_effect', updated),
                          ('recurrence_effect_updated_minus_frozen', updated-frozen)]:
        mean = float(values.mean())
        se = float(values.std(ddof=1)/np.sqrt(5))
        interval = list(t.interval(.95, 4, loc=mean, scale=se)) if se else [mean, mean]
        contrasts[label] = dict(dataset_differences=values.tolist(), mean=mean,
                                descriptive_t95ci=interval, positive_datasets=int((values > 0).sum()),
                                negative_datasets=int((values < 0).sum()))
    return contrasts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs-root', type=Path, default=Path('runs/results'))
    ap.add_argument('--output', type=Path, default=Path('reports/ROUND11B_RESULTS.json'))
    args = ap.parse_args()
    # Check every group's acceptance before opening any scientific outcome.
    for d in range(5):
        accepted = json.loads(Path(f'runs/evidence/round11b_data{d}_bundle/acceptance.json').read_text())
        assert accepted['accepted_runs'] == 12 and accepted['accepted_npz'] == 36
        assert accepted['experiment'] == f'round11b_data{d}_v1'
    roster = json.loads(Path('configs/sim/round11_generation.json').read_text())
    metadata = {row['name']: row for row in roster['rows']}
    scores = {c: {a: [] for a in ARMS} for c in CONDITIONS}
    episode_records = []
    for d in range(5):
        for arm in ARMS:
            by_condition = {c: [] for c in CONDITIONS}
            for seed in [70, 71, 72]:
                model = f'r11b_d{d}_{arm}'
                run = args.runs_root/f'round11b_data{d}_v1'/f'{model}_seed{seed}'
                prov = json.loads((run/'provenance.json').read_text())
                assert prov['git_commit'] == SOURCE and prov['seed'] == seed and prov['model_name'] == model
                for split in ['d2_static_ood', 'd3_changing_ood']:
                    with np.load(run/'eval'/f'{split}_per_window.npz', allow_pickle=False) as data:
                        error = data['err_aggregate']
                        assert np.isfinite(error).all() and (error >= 0).all()
                        assert len(data['flight_names']) == (30 if split == 'd2_static_ood' else 20)
                        groups = ['static'] if split == 'd2_static_ood' else ['slow', 'fast']
                        expected_names = {r['name'] for r in roster['rows']
                                          if r['replicate'] == d and r['group'] in groups}
                        assert {str(name) for name in data['flight_names']} == expected_names
                        component_arrays = {metric: data['err_'+metric] for metric in
                                            ['velocity', 'orientation', 'angular_rate', 'displacement']}
                        for fid, name in enumerate(data['flight_names']):
                            row = metadata[str(name)]
                            assert row['replicate'] == d
                            condition = row['group']+'_'+row['condition']
                            mask = data['flight_ids'] == fid
                            assert int(mask.sum()) == 2850
                            value = float(error[mask].mean())
                            by_condition[condition].append(value)
                            components = {metric: float(values[mask].mean())
                                          for metric, values in component_arrays.items()}
                            episode_records.append(dict(dataset=d, arm=arm, seed=seed, condition=condition,
                                                        trajectory_seed=row['trajectory_seed'], aggregate=value,
                                                        components=components,
                                                        terminal=float(error[mask, -1].mean())))
            for condition, values in by_condition.items():
                assert len(values) == 30  # three seeds x ten equally weighted flights
                scores[condition][arm].append(float(np.mean(values)))
    report = dict(protocol='docs/ROUND11_PROTOCOL.md', source=SOURCE,
                  uncertainty_unit='independent training-data replicate (n=5)',
                  caveat='Simulation only; no p-values or equivalence claim. CIs descriptive and not simultaneous. '
                         'Each replicate uses its own train-only metric scales. Conditions remain separate.',
                  dataset_scores=scores,
                  contrasts={c: summarize_replicates(s) for c, s in scores.items()},
                  episode_records=episode_records)
    with args.output.open('x') as f:
        f.write(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report['contrasts'], indent=2))


if __name__ == '__main__':
    main()
