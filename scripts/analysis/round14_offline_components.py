"""Post-review component audit of accepted Round11B arrays; no new inference."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[2]
HORIZONS = [1, 5, 10, 25, 50]
METRICS = ['aggregate', 'velocity', 'orientation', 'angular_rate', 'displacement']


def describe(values):
    a = np.asarray(values, dtype=float)
    assert a.shape == (5,) and np.isfinite(a).all()
    mean = float(a.mean())
    half = float(t.ppf(.975, 4) * a.std(ddof=1) / np.sqrt(5))
    return dict(dataset_differences=a.tolist(), mean=mean,
                descriptive_t95ci=[mean-half, mean+half])


def main():
    original = json.loads((ROOT/'reports/ROUND11B_RESULTS.json').read_text())
    roster = json.loads((ROOT/'configs/sim/round11_generation.json').read_text())
    metadata = {r['name']: r for r in roster['rows']}
    sources, records, means = {}, [], {}
    for d in range(5):
        acceptance = json.loads((ROOT/f'runs/evidence/round11b_data{d}_bundle/acceptance.json').read_text())
        assert acceptance['accepted_runs'] == 12 and acceptance['accepted_npz'] == 36
        for seed in [70, 71, 72]:
            paired_ids = None
            for arm in ['frozen_r0', 'updated_r0']:
                path = ROOT/f'runs/results/round11b_data{d}_v1/r11b_d{d}_{arm}_seed{seed}/eval/d2_static_ood_per_window.npz'
                sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
                with np.load(path, allow_pickle=False) as z:
                    ids = [z[k].copy() for k in ['flight_ids', 'centre_index', 'flight_names']]
                    if paired_ids is None:
                        paired_ids = ids
                    else:
                        assert all(np.array_equal(a, b) for a, b in zip(ids, paired_ids))
                    arrays = {m: z['err_'+m] for m in METRICS}
                    assert all(a.shape == (85500, 50) and np.isfinite(a).all() and (a >= 0).all() for a in arrays.values())
                    assert np.allclose(arrays['aggregate'], sum(arrays[m] for m in METRICS[1:4])/3, rtol=2e-6, atol=1e-7)
                    for fid, name in enumerate(z['flight_names']):
                        meta = metadata[str(name)]
                        condition = meta['group']+'_'+meta['condition']
                        if condition not in ['static_50wind', 'static_70wind']:
                            continue
                        mask = z['flight_ids'] == fid
                        assert mask.sum() == 2850
                        old = [r for r in original['episode_records'] if r['dataset'] == d and r['seed'] == seed and r['arm'] == arm and r['condition'] == condition and r['trajectory_seed'] == meta['trajectory_seed']]
                        assert len(old) == 1
                        assert np.isclose(float(arrays['aggregate'][mask].mean()), old[0]['aggregate'], rtol=1e-12, atol=1e-12)
                        metrics = {m: {'prefix_mean': float(a[mask].mean()), 'terminal': {str(h): float(a[mask, h-1].mean()) for h in HORIZONS}} for m, a in arrays.items()}
                        records.append(dict(dataset=d, seed=seed, arm=arm, condition=condition, flight=str(name), metrics=metrics))
        print('verified dataset', d, flush=True)
    summary = {}
    for condition in ['static_50wind', 'static_70wind']:
        summary[condition] = {}
        for metric in METRICS:
            summary[condition][metric] = {}
            for horizon in ['prefix_mean'] + [str(h) for h in HORIZONS]:
                differences = []
                for d in range(5):
                    arm_means = {}
                    for arm in ['frozen_r0', 'updated_r0']:
                        selected = [r for r in records if r['dataset'] == d and r['arm'] == arm and r['condition'] == condition]
                        assert len(selected) == 30
                        vals = [r['metrics'][metric]['prefix_mean'] if horizon == 'prefix_mean' else r['metrics'][metric]['terminal'][horizon] for r in selected]
                        arm_means[arm] = float(np.mean(vals))
                        if metric == 'aggregate' and horizon == 'prefix_mean':
                            assert np.isclose(arm_means[arm], original['dataset_scores'][condition][arm][d], rtol=1e-12, atol=1e-12)
                    differences.append(arm_means['updated_r0']-arm_means['frozen_r0'])
                summary[condition][metric][horizon] = describe(differences)
    result = dict(post_hoc=True, scope='30 existing no-LN GRU checkpoints; accepted static OOD arrays; no new inference',
                  contrast='Updated minus Frozen; negative favors Updated',
                  aggregation='equal windows within flight, then equal 10 flights and 3 seeds within dataset; n=5 training datasets',
                  caveat='Descriptive non-simultaneous intervals; normalized component units. Stored displacement is coordinate-averaged normalized Euler-integrated velocity error, NOT physical position L2 in meters. Cannot invert this scalar to recover physical position. Physical-position evaluation remains pending.',
                  source_sha256=sources, summary=summary, records=records)
    target = ROOT/'reports/ROUND14_OFFLINE_COMPONENTS.json'
    with target.open('x') as f:
        f.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps({c:{m:{h:round(v['mean'],7) for h,v in s.items()} for m,s in row.items()} for c,row in summary.items()}, indent=2))


if __name__ == '__main__':
    main()
