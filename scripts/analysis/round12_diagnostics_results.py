"""Accept complete Round12A rosters and report paired dataset-level diagnostics.

Pilot acceptance deliberately reports timing/coverage only. Formal summaries
never pool pilots or silently drop failed/undefined observations. Monte Carlo
uncertainty preserves common noise across models, seeds and datasets at an anchor.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import t

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO/'scripts/sim'))
import round12_diagnostic_protocol as protocol  # noqa: E402
import round12_diagnostics as diagnostics  # noqa: E402

SCHEMA = 'round12a-analysis-v1'
VALIDATION_CONTRACT = 'round12a-analysis-preflight-v1'
PRIMARY = ('physical_solution_cost', 'physical_solution_cost_excess_vs_oracle_softmin')
HORIZON_COMPONENTS = ('original_E', 'original_velocity_E', 'original_orientation_E',
                      'original_angular_rate_E', 'original_displacement_E', 'velocity_l2_mps',
                      'physical_position_l2_m', 'planner_position_l2_m')
SECONDARY = ('planner_solution_cost', 'planner_solution_cost_excess_vs_oracle_softmin',
             'weights_ess', 'weights_entropy', 'solution_threshold_exceedance_rate')
METRICS = (list(PRIMARY)+list(SECONDARY)
    + [f'{domain}_candidate_{name}' for domain in ('physical', 'planner')
       for name in ('spearman', 'pairwise_inversion_rate', 'empirical_candidate_regret', 'cost_mae')]
    + [f'h{h}/{name}/{summary}' for h in (1, 5, 10, 25, 50) for name in HORIZON_COMPONENTS
       for summary in ('terminal', 'prefix_mean')]
    + [f'h{h}/velocity_error_vs_ensemble_mean_mps' for h in (1, 5, 10, 25, 50)])
CONTRASTS = {
    'updated_minus_frozen_no_ln': {'updated_r0': 1., 'frozen_r0': -1.},
    'frozen_ln_effect': {'frozen_r1': 1., 'frozen_r0': -1.},
    'updated_ln_effect': {'updated_r1': 1., 'updated_r0': -1.},
    'ln_effect_updated_minus_frozen': {'updated_r1': 1., 'updated_r0': -1.,
                                     'frozen_r1': -1., 'frozen_r0': 1.},
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arm_of(name):
    arm = '_'.join(name.split('_')[-2:])
    if arm not in protocol.ARMS:
        raise ValueError(f'unknown original arm: {name}')
    return arm


def _finite_scalar(value):
    return type(value) in (int, float) and np.isfinite(value)


def strict_preflight(result, trace_path):
    """Harden offline acceptance without changing the immutable execution code."""
    if result['status'] != 'complete' or result['trace_sha256'] != digest(trace_path):
        raise ValueError('incomplete result or corrupt trace')
    row = result['row']
    checks = result['replay_checks']
    for name in ('max_action_abs_error', 'max_tracking_abs_error', 'max_warmup_abs_error'):
        value = checks[name]
        if not _finite_scalar(value) or value < 0:
            raise ValueError(f'replay check must be finite and nonnegative: {name}')
    if type(checks['plans_reconstructed']) is not int or checks['plans_reconstructed'] < 0:
        raise ValueError('replay plan count must be a nonnegative integer')
    specs = protocol.model_rows(row['wind'], row['episode'], row.get('dataset_replicate'), row.get('model_seed'))
    expected = {(r['dataset_replicate'], r['model'], r['model_seed']) for r in specs}
    with np.load(trace_path, allow_pickle=False) as trace:
        def finite_array(name, shape=None):
            if name not in trace:
                raise ValueError(f'missing required trace array: {name}')
            value = trace[name]
            if (value.dtype.kind not in 'fiu' or not np.isfinite(value).all()
                    or (shape is not None and value.shape != shape)):
                raise ValueError(f'invalid finite trace array or shape: {name}')
            return value

        for anchor in result['anchors']:
            models = anchor['models']
            if any(type(m['dataset_replicate']) is not int or type(m['model_seed']) is not int
                   or not isinstance(m['model'], str) for m in models):
                raise ValueError('invalid dataset/model/seed identity types')
            actual = {(m['dataset_replicate'], m['model'], m['model_seed']) for m in models}
            if len(models) != len(expected) or actual != expected:
                raise ValueError('exact dataset/model/seed identity coverage mismatch')
            samples = anchor['mc_samples']
            if type(samples) is not int or samples < 2 or (not result['pilot'] and samples != protocol.MC_SAMPLES):
                raise ValueError('MC samples must be an integer >= 2 for pilot and exactly 8 for formal')
            prefix = f"anchor_{anchor['step']}"
            for name, shape in (('nominal', (50, 5)), ('shifted_prev', (50, 3)),
                                ('perturbations', (64, 50, 3)), ('actions', (64, 50, 5))):
                finite_array(f'{prefix}/bank/{name}', shape)
            # Include optional donor predictions/weights saved in the bank as well.
            for name in trace.files:
                if name.startswith((f'{prefix}/bank/', f'{prefix}/noise/')):
                    finite_array(name)
            for phase in ('construct', 'evaluate'):
                for name, shape in (('gust', (samples, 50, 3)), ('throttle', (samples, 50)),
                                    ('rate', (samples, 50, 10, 3))):
                    finite_array(f'{prefix}/noise/{phase}/{name}', shape)
            complete = [m for m in models if m['status'] == 'complete']
            indices = [m['solution_index'] for m in complete]
            if any(type(i) is not int for i in indices) or sorted(indices) != list(range(1, len(complete)+1)):
                raise ValueError('invalid weighted-solution indexing')
            columns = 65+len(complete)
            finite_array(f'{prefix}/evaluation/actions', (columns, 50, 5))
            finite_array(f'{prefix}/evaluation/perturbations', (columns, 50, 3))
            costs = {domain: finite_array(f'{prefix}/evaluation_cost/{domain}', (samples, columns))
                     for domain in ('physical', 'planner')}
            for model in models:
                for field in ('first_action', 'source_first_action'):
                    if field in model:
                        action = np.asarray(model[field])
                        if action.shape != (5,) or action.dtype.kind not in 'fiu' or not np.isfinite(action).all():
                            raise ValueError(f'invalid finite action: {field}')
                if 'first_action_reconstruction_abs_error' in model:
                    error = model['first_action_reconstruction_abs_error']
                    if not _finite_scalar(error) or not 0 <= error <= protocol.ACTION_ATOL:
                        raise ValueError('invalid first-action replay error')
                if model['status'] != 'complete':
                    continue
                for domain, matrix in costs.items():
                    raw = matrix[:, 64+model['solution_index']]
                    gap = raw-matrix[:, 64]
                    comparisons = (
                        (f'{domain}_solution_cost', raw.mean(), 'raw cost mean'),
                        (f'{domain}_solution_cost_excess_vs_oracle_softmin', gap.mean(), 'signed cost excess'),
                        (f'{domain}_solution_cost_excess_mc_se', gap.std(ddof=1)/np.sqrt(samples), 'paired cost MC SE'),
                    )
                    for field, calculated, label in comparisons:
                        value = model[field]
                        if not _finite_scalar(value) or not np.isclose(value, calculated, rtol=1e-12, atol=1e-12):
                            raise ValueError(f'reported {domain} {label} disagrees with trace')


def accept(root, stage, source, *, pilot=False):
    if not re.fullmatch('[0-9a-f]{40}', source):
        raise ValueError('source must be the immutable forty-character execution SHA')
    root = Path(root)
    paths = sorted(root.glob(f'{stage}_[0-9][0-9][0-9][0-9].json'))
    expected = protocol.design()[stage]
    if pilot:
        if len(paths) not in (1, 2):
            raise ValueError('pilot acceptance requires exactly one or two records')
    elif {p.name for p in paths} != {f'{stage}_{r["index"]:04d}.json' for r in expected}:
        raise ValueError('formal acceptance requires the complete exact roster; no partial results')
    records, files, identities = [], [], {}
    source_files = None
    for path in paths:
        if path.with_suffix('.error.json').exists():
            raise ValueError('record has a retained software/truth error artifact')
        result = json.loads(path.read_text())
        row = result['row']
        if row['stage'] != stage or path.name != f'{stage}_{row["index"]:04d}.json':
            raise ValueError('filename/record stage or index mismatch')
        if result['source_commit'] != source or result['pilot'] is not pilot:
            raise ValueError('execution source or pilot/formal phase mismatch')
        current_sources = result['source_file_sha256']
        if not current_sources or any(not re.fullmatch('[0-9a-f]{64}', value) for value in current_sources.values()):
            raise ValueError('missing execution source-file identities')
        if source_files is None:
            source_files = current_sources
        elif current_sources != source_files:
            raise ValueError('execution source files differ across records')
        trace = path.with_suffix('.npz')
        strict_preflight(result, trace)
        diagnostics.validate_result(result, trace)
        if not np.isfinite(result['elapsed_seconds']) or result['elapsed_seconds'] <= 0:
            raise ValueError('invalid recorded elapsed time')
        for anchor in result['anchors']:
            for model in anchor['models']:
                key = (model['model'], model['model_seed'])
                value = model['checkpoint_sha256']
                if not re.fullmatch('[0-9a-f]{64}', value):
                    raise ValueError('invalid checkpoint identity')
                if key in identities and identities[key] != value:
                    raise ValueError('checkpoint bytes changed within the diagnostic roster')
                identities[key] = value
        file_identity = dict(index=row['index'], json_name=path.name, json_sha256=digest(path),
                             npz_name=trace.name, npz_sha256=digest(trace))
        claim_path = path.with_suffix('.claim')
        if claim_path.exists():
            claim = json.loads(claim_path.read_text())
            for field in ('row', 'source_commit', 'protocol_sha256', 'pilot', 'source_file_sha256'):
                if claim[field] != result[field]:
                    raise ValueError('claim/result identity mismatch')
            file_identity['claim_sha256'] = digest(claim_path)
        records.append((result, trace))
        files.append(file_identity)
    if not pilot and len(identities) != 60:
        raise ValueError('formal roster must retain all sixty checkpoint identities')
    return records, dict(stage=stage, pilot=pilot, source_commit=source,
        validation_contract=VALIDATION_CONTRACT,
        protocol_sha256=protocol.canonical_hash(protocol.design()), source_file_sha256=source_files,
        accepted_records=len(records), accepted_anchors=sum(len(r['anchors']) for r, _ in records),
        checkpoint_count=len(identities),
        checkpoint_identities=[dict(model=k[0], model_seed=k[1], sha256=v) for k, v in sorted(identities.items())],
        files=files)


def metrics_of(model):
    if model['status'] != 'complete':
        return {}
    metrics = {name: model[name] for name in PRIMARY}
    for name in SECONDARY:
        metrics[name] = model.get(name)
    for domain in ('physical', 'planner'):
        for name in ('spearman', 'pairwise_inversion_rate', 'empirical_candidate_regret', 'cost_mae'):
            metrics[f'{domain}_candidate_{name}'] = model.get(f'{domain}_candidate_ranking', {}).get(name)
    for horizon in ('1', '5', '10', '25', '50'):
        for name, value in model.get('horizon_metrics', {}).get(horizon, {}).items():
            if isinstance(value, dict):
                for summary in ('terminal', 'prefix_mean'):
                    metrics[f'h{horizon}/{name}/{summary}'] = value.get(summary)
            else:
                metrics[f'h{horizon}/{name}'] = value
    for name, value in metrics.items():
        if value is not None and not np.isfinite(value):
            raise ValueError(f'nonfinite completed-model metric: {name}')
    return metrics


def collect_units(records):
    units = []
    for result, path in records:
        row = result['row']
        with np.load(path, allow_pickle=False) as trace:
            for anchor in result['anchors']:
                costs = trace[f"anchor_{anchor['step']}/evaluation_cost/physical"]
                for model in anchor['models']:
                    complete = model['status'] == 'complete'
                    actual = costs[:, 64+model['solution_index']].copy() if complete else None
                    signed = actual-costs[:, 64] if complete else None
                    unit = dict(anchor=f"{row['index']}:{anchor['step']}", wind=row['wind'],
                        episode=row['episode'], dataset=model['dataset_replicate'], seed=model['model_seed'],
                        owner=arm_of(row['owner']) if row['stage'] == 'a2' else 'common_pd',
                        step=anchor['step'], arm=arm_of(model['model']), status=model['status'],
                        metrics=metrics_of(model), mc={PRIMARY[0]: actual, PRIMARY[1]: signed})
                    if complete:
                        for metric, vector in unit['mc'].items():
                            if not np.isclose(vector.mean(), unit['metrics'][metric], rtol=1e-11, atol=1e-11):
                                raise ValueError('primary JSON metric disagrees with per-path trace')
                    units.append(unit)
    return units


def describe(values):
    if len(values) != 5:
        raise ValueError('exactly five dataset-level paired differences required')
    if any(value is None for value in values):
        return dict(status='not_estimable', dataset_differences=values,
                    mean=None, descriptive_t95ci=None, positive_datasets=None, negative_datasets=None,
                    reason='At least one required observation failed or has an undefined metric; no observations dropped.')
    a = np.asarray(values, dtype=float)
    if not np.isfinite(a).all():
        raise ValueError('invalid paired differences')
    mean = float(a.mean())
    se = float(a.std(ddof=1)/np.sqrt(5))
    interval = [mean, mean] if se == 0 else list(t.interval(.95, 4, loc=mean, scale=se))
    return dict(status='estimable', dataset_differences=a.tolist(), mean=mean,
                descriptive_t95ci=interval, positive_datasets=int((a>0).sum()),
                negative_datasets=int((a<0).sum()))


def paired_mc(units, coefficients, metric):
    """Pair arms first, then preserve every within-anchor common-noise dependency."""
    pairs = defaultdict(dict)
    for unit in units:
        if unit['arm'] in coefficients:
            key = (unit['anchor'], unit['dataset'], unit['seed'])
            if unit['arm'] in pairs[key]:
                raise ValueError('duplicate model observation in paired MC analysis')
            pairs[key][unit['arm']] = unit
    if not pairs or any(set(pair) != set(coefficients) for pair in pairs.values()):
        raise ValueError('incomplete paired model coverage')
    failed = sum(any(u['status'] != 'complete' for u in pair.values()) for pair in pairs.values())
    if failed:
        return dict(status='not_estimable', mean=None, mc_se=None, paired_observations=len(pairs),
                    failed_pairs=failed, independent_noise_anchors=len({k[0] for k in pairs}))
    weighted = {}
    for key, pair in pairs.items():
        # All model costs at this decision use exactly the same indexed noise paths.
        vector = sum(coefficients[arm]*pair[arm]['mc'][metric] for arm in coefficients)/len(pairs)
        anchor = key[0]
        weighted[anchor] = weighted.get(anchor, np.zeros_like(vector))+vector
    variance = sum(float(vector.var(ddof=1)/len(vector)) for vector in weighted.values())
    return dict(status='estimable', mean=float(sum(v.mean() for v in weighted.values())),
                mc_se=float(np.sqrt(variance)), paired_observations=len(pairs), failed_pairs=0,
                independent_noise_anchors=len(weighted))


def summarize(units):
    metrics = sorted({metric for unit in units for metric in unit['metrics']} | set(METRICS))
    arms = {}
    for arm in protocol.ARMS:
        subset = [u for u in units if u['arm'] == arm]
        if not subset:
            raise ValueError('missing arm in registered stratum')
        values, undefined = {}, {}
        groups = [[u for u in subset if u['dataset'] == d] for d in range(5)]
        if any(not g for g in groups):
            raise ValueError('missing dataset in registered stratum')
        for metric in metrics:
            per_dataset, missing = [], []
            for group in groups:
                data = [u['metrics'].get(metric) for u in group]
                missing.append(sum(u['status'] == 'complete' and u['metrics'].get(metric) is None for u in group))
                per_dataset.append(float(np.mean(data)) if all(v is not None for v in data) else None)
            values[metric], undefined[metric] = per_dataset, missing
        arms[arm] = dict(dataset_means=values,
                        observations_by_dataset=[len(g) for g in groups],
                        failed_by_dataset=[sum(u['status'] != 'complete' for u in g) for g in groups],
                        undefined_metric_by_dataset=undefined,
                        prediction_failure_rate=float(np.mean([u['status'] != 'complete' for u in subset])))
    contrasts, mc = {}, {}
    for name, coefficients in CONTRASTS.items():
        contrasts[name] = {}
        for metric in metrics:
            values = []
            for d in range(5):
                arm_values = {arm: arms[arm]['dataset_means'][metric][d] for arm in coefficients}
                values.append(sum(coefficients[a]*v for a, v in arm_values.items())
                              if all(v is not None for v in arm_values.values()) else None)
            contrasts[name][metric] = describe(values)
        mc[name] = {}
        for metric in PRIMARY:
            overall = paired_mc(units, coefficients, metric)
            per_dataset = [paired_mc([u for u in units if u['dataset'] == d], coefficients, metric)
                           for d in range(5)]
            expected = contrasts[name][metric]
            if overall['status'] != expected['status']:
                raise ValueError('MC and dataset contrast coverage disagree')
            if overall['status'] == 'estimable' and not np.isclose(overall['mean'], expected['mean'], rtol=1e-10, atol=1e-10):
                raise ValueError('MC pairing and dataset aggregation weights disagree')
            mc[name][metric] = dict(overall=overall, datasets=per_dataset)
    return dict(model_observations=len(units), arms=arms, contrasts=contrasts, paired_mc=mc)


def aggregate(units, stage):
    strata, balanced = {}, {}
    for wind in sorted({u['wind'] for u in units}):
        subset = [u for u in units if u['wind'] == wind]
        if stage == 'a1':
            strata[str(wind)] = summarize(subset)
        else:
            strata[str(wind)] = {}
            for owner in ('frozen_r0', 'updated_r0'):
                strata[str(wind)][owner] = {}
                for step in protocol.A2_STEPS:
                    group = [u for u in subset if u['owner'] == owner and u['step'] == step]
                    strata[str(wind)][owner][str(step)] = summarize(group)
            balanced[str(wind)] = summarize(subset)
    return dict(wind_strata=strata, balanced_owner_time_secondary=balanced,
        interpretation='All contrasts are updated minus frozen where named; negative primary cost differences favor updated. '
          'Intervals are descriptive t95 over five training datasets, conditional on fixed episodes, owners and anchor times. '
          'No p-values, simultaneous intervals, equivalence or unique causal-mechanism claim. '
          'Signed oracle-softmin cost excess is not nonnegative regret or an optimal-control bound.',
        mc_method='Form paired arm differences on common evaluation noise paths. Average seeds/datasets sharing an anchor '
          'before variance estimation; sum variances only across independently seeded anchors. '
          'This evaluation-noise MC SE is distinct from the five-dataset uncertainty and conditions on constructed oracle solutions.',
        missing_policy='No failed model or undefined metric is dropped. A required missing observation makes its '
          'dataset mean and full contrast not estimable; failure and undefined counts remain explicit.')


def analyze(root, stage, source, *, pilot=False):
    records, acceptance = accept(root, stage, source, pilot=pilot)
    report = dict(schema=SCHEMA, acceptance=acceptance, scientific_analysis_run=not pilot,
        analysis_source_sha256={str(path.relative_to(REPO)): digest(path) for path in
            (Path(__file__), Path(diagnostics.__file__), Path(protocol.__file__))})
    if pilot:
        report['timing_only'] = [dict(index=r['row']['index'], elapsed_seconds=r['elapsed_seconds'],
            anchors=len(r['anchors']), models_per_anchor=[a['model_count'] for a in r['anchors']],
            mc_samples=[a['mc_samples'] for a in r['anchors']], npz_bytes=path.stat().st_size,
            nonfinite_model_records=sum(m['status'] != 'complete' for a in r['anchors'] for m in a['models']),
            usable_for_complete_model_timing=all(m['status'] == 'complete' for a in r['anchors'] for m in a['models']))
            for r, path in records]
    else:
        report.update(aggregate(collect_units(records), stage))
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--stage', choices=['a1', 'a2'], required=True)
    ap.add_argument('--source', required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--pilotaccept', action='store_true')
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = analyze(args.root, args.stage, args.source, pilot=args.pilotaccept)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(accepted_records=report['acceptance']['accepted_records'],
                         stage=args.stage, pilot=args.pilotaccept, output=str(args.output))))


if __name__ == '__main__':
    main()
