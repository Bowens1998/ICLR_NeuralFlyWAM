"""Locked four-test Round11C family; run only after complete artifact acceptance."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import t, wilcoxon

ARMS = [f'{u}_r{r}o{o}' for u in ['frozen', 'updated'] for r in [0, 1] for o in [0, 1]]
SEEDS = list(range(60, 70))


def planned_tests(scores):
    if set(scores) != set(ARMS):
        raise ValueError('all eight arms required')
    a = {k: np.asarray(v, dtype=float) for k, v in scores.items()}
    if any(v.shape != (10,) or not np.isfinite(v).all() for v in a.values()):
        raise ValueError('ten finite paired outcomes per arm required')
    rec_u = a['updated_r1o0'] - a['updated_r0o0']
    read_u = a['updated_r0o1'] - a['updated_r0o0']
    rec_f = a['frozen_r1o0'] - a['frozen_r0o0']
    read_f = a['frozen_r0o1'] - a['frozen_r0o0']
    tests = []
    for label, delta in [('updated_recurrence_LN_effect', rec_u),
                         ('updated_readout_LN_effect', read_u),
                         ('recurrence_effect_updated_minus_frozen', rec_u-rec_f),
                         ('readout_effect_updated_minus_frozen', read_u-read_f)]:
        se = delta.std(ddof=1)/np.sqrt(10)
        ci = list(t.interval(.95, 9, loc=delta.mean(), scale=se)) if se else [float(delta.mean())]*2
        p = float(wilcoxon(delta, alternative='two-sided').pvalue) if np.any(delta) else 1.0
        tests.append(dict(label=label, seeds=SEEDS, differences=delta.tolist(),
                          mean=float(delta.mean()), ci95=ci, wilcoxon_p=p,
                          positive_seeds=int((delta > 0).sum()), negative_seeds=int((delta < 0).sum())))
    adjusted = 0.0
    for i, row in enumerate(sorted(tests, key=lambda r: r['wilcoxon_p'])):
        adjusted = max(adjusted, min(1.0, (4-i)*row['wilcoxon_p']))
        row.update(holm_p=adjusted, significant_familywise_005=adjusted < .05)
    return tests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs-root', type=Path, default=Path('runs/results/main_norm_location_v1'))
    ap.add_argument('--output', type=Path, default=Path('reports/ROUND11C_RESULTS.json'))
    args = ap.parse_args()
    acceptance = json.loads(Path('runs/evidence/round11_location_bundle/acceptance.json').read_text())
    assert acceptance['accepted_runs'] == 80 and acceptance['accepted_npz'] == 240
    scores = {arm: [] for arm in ARMS}
    descriptive = []
    for arm in ARMS:
        model = f"r11_{arm}"
        for seed in SEEDS:
            run = args.runs_root/f'{model}_seed{seed}'
            prov = json.loads((run/'provenance.json').read_text())
            assert prov['git_commit'] == 'df266e206871a74fc2bb7e2056bc68bf5e0fa4a9'
            assert prov['seed'] == seed and prov['model_name'] == model
            for split in ['d1_val', 'd2_static_ood', 'd3_changing_ood']:
                with np.load(run/'eval'/f'{split}_per_window.npz') as d:
                    error = d['err_aggregate']
                    assert np.isfinite(error).all()
                    value = float(error.mean())
                    if split == 'd2_static_ood':
                        scores[arm].append(value)
                    flights = []
                    for fid, name in enumerate(d['flight_names']):
                        mask = d['flight_ids'] == fid
                        if mask.any():
                            flights.append(dict(flight=str(name), aggregate=float(error[mask].mean())))
                    descriptive.append(dict(arm=arm, seed=seed, split=split, aggregate=value, flights=flights))
    report = dict(protocol='docs/ROUND11_PROTOCOL.md',
                  source='df266e206871a74fc2bb7e2056bc68bf5e0fa4a9',
                  caveat='Fixed-log training-seed inference; descriptive CIs are not simultaneous. No equivalence claim.',
                  outcome='d2_static_ood aggregate', scores=scores,
                  tests=planned_tests(scores), descriptive=descriptive)
    with args.output.open('x') as f:
        f.write(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report['tests'], indent=2))


if __name__ == '__main__':
    main()
