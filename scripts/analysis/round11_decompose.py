"""Descriptive decomposition of already accepted Round10B; no new tests."""
import json
from pathlib import Path

import numpy as np

ROOT = Path('runs/results/main_factorial_lr_confirm_v1')


def main():
    selection = json.loads(Path('reports/ROUND10B_SELECTION.json').read_text())['selected']
    rows = []
    effects = {}
    for arm, rate in selection.items():
        for seed in range(40, 50):
            for split in ['d1_val', 'd2_static_ood', 'd3_changing_ood']:
                path = ROOT/f'r10b_{arm}_lr{rate}_seed{seed}'/'eval'/f'{split}_per_window.npz'
                with np.load(path) as d:
                    for fid, flight in enumerate(d['flight_names']):
                        mask = d['flight_ids'] == fid
                        if not mask.any():
                            continue
                        for metric in ['aggregate', 'velocity', 'orientation', 'angular_rate', 'displacement']:
                            a = d[f'err_{metric}'][mask]
                            row = dict(arm=arm, seed=seed, split=split, flight=str(flight),
                                       metric=metric, windows=int(mask.sum()), mean=float(a.mean()),
                                       terminal=float(a[:, -1].mean()), horizon=a.mean(0).tolist())
                            rows.append(row)
                            key = (seed, split, str(flight), metric)
                            effects.setdefault(key, {})[arm] = row['mean']
    contrasts = []
    for (seed, split, flight, metric), arms in effects.items():
        frozen = arms['frozen_ln']-arms['frozen_none']
        updated = arms['updated_ln']-arms['updated_none']
        contrasts.append(dict(seed=seed, split=split, flight=flight, metric=metric,
                              frozen_ln_effect=frozen, updated_ln_effect=updated,
                              interaction=updated-frozen))
    report = dict(scope='Post-hoc descriptive decomposition; no new p-values; fixed logs.',
                  source='reports/ROUND10B_RESULTS.json', rows=rows, contrasts=contrasts)
    Path('reports/ROUND11A_DECOMPOSITION.json').write_text(json.dumps(report, indent=2)+'\n')
    print(f'{len(rows)} arm/seed/flight/component rows; {len(contrasts)} paired contrasts')


if __name__ == '__main__':
    main()
