"""Portable simulation reproduction, using the study's unchanged scientific code.

Run from the anonymous repository root. All new data and results stay in a
separate workspace; historical evidence and exported weights remain read-only.
"""
# ruff: noqa: E402 -- standalone entry point.
import argparse
import copy
import hashlib
import importlib
import json
import os
import platform
import shutil
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts/sim'))
sys.path.insert(0, str(ROOT / 'scripts/analysis'))

from latent_aero_wam.data.splits import manifest_checksum
from latent_aero_wam.evaluation.evaluator import evaluate, load_model_from_checkpoint
from latent_aero_wam.training.loop import train
from latent_aero_wam.utils.config import load_config

COHORTS = {
    'original': ('round11', 'round11b_v1', 'round15_generate', 'round15'),
    'confirmation': ('round18', 'round18', 'round18_branches', 'round18'),
}
STUDIES = {'original': 'round11_closed_loop', 'bridge': 'round16_evaluation',
           'confirmation': 'round18_evaluation'}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def execution_environment(device):
    return dict(python=platform.python_version(), numpy=np.__version__, torch=str(torch.__version__),
                device=device,
                gpu=torch.cuda.get_device_name(torch.device(device)) if device.startswith('cuda') else None,
                torch_threads=torch.get_num_threads(),
                cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                cudnn_benchmark=torch.backends.cudnn.benchmark)


def write(path, value, *, existing_ok=False):
    path = Path(path)
    payload = json.dumps(value, indent=2, allow_nan=False) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    if existing_ok and path.exists():
        if path.read_text() != payload:
            raise ValueError(f'existing file differs: {path}')
        return
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
        try:
            os.link(temporary, path)  # Publish only a complete file, without overwriting.
        except FileExistsError:
            if not existing_ok or path.read_text() != payload:
                raise
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@lru_cache(maxsize=1)
def exports():
    return read(ROOT / 'CHECKPOINT_EXPORTS.json')


def entry(model, seed):
    return next(e for e in exports()
                if (e['model'], e['seed']) == (model, seed))


def cohort_of(e):
    return 'confirmation' if e['model'].startswith('r18_') else 'original'


def plan(study, stage=None):
    roster = read(ROOT / f'configs/sim/{STUDIES[study]}.json')
    identities = sorted({(r['model'], r['model_seed']) for r in roster['rows']
                         if r['kind'] == 'model'})
    if study == 'bridge' and stage == 'offline':
        # The bridge's logged-action factorial contains the 240 recipe models.
        # Its 30 original references belong to the original prediction panel.
        roster['models'] = [x for x in roster['models'] if x['coverage'] != 'original']
        identities = sorted({(x['config']['model']['name'], x['seed']) for x in roster['models']})
    return roster, [entry(m, s) for m, s in identities]


def runtime_config(e, workspace):
    cfg = copy.deepcopy(e['config'])
    cohort = cohort_of(e)
    base = workspace / cohort / 'artifacts'
    cfg['data']['manifest_path'] = str(base / f'data{e["dataset"]}_split.json')
    cfg['data']['normalizer_path'] = str(base / f'data{e["dataset"]}_norm.json')
    if 'coverage_manifest' in cfg['data']:
        cfg['data']['coverage_manifest'] = str(base / f'data{e["dataset"]}_coverage.json')
    cfg['output_root'] = str(workspace / 'runs')
    return cfg


def environment(workspace, cohort):
    os.environ['LATENT_WAM_DATA_ROOT'] = str(workspace / cohort / 'csv')
    os.environ['LATENT_WAM_COVERAGE_ROOT'] = str(workspace / cohort / 'branches')
    os.environ['LATENT_WAM_CACHE_ROOT'] = str(workspace / cohort / 'cache')
    os.environ['LATENT_WAM_SOURCE_COMMIT'] = 'portable-reproduction'


def init(args):
    """Rebind anonymous metadata without changing any numerical split definition."""
    _, artifact, _, _ = COHORTS[args.cohort]
    base = args.workspace / args.cohort
    for d in range(5):
        source = ROOT / f'artifacts/{artifact}/data{d}_split.json'
        m = read(source)
        old_checksum = m['checksum']
        m['data_root'] = str(base / 'csv')
        # Anonymization changed provenance strings inside the original checksum.
        # Keep its identity separately, and hash the actual portable manifest.
        m['checksum'] = manifest_checksum(m)
        write(base / f'artifacts/data{d}_split.json', m, existing_ok=True)
        n = ROOT / f'artifacts/{artifact}/data{d}_norm.json'
        dest = base / f'artifacts/data{d}_norm.json'
        if dest.exists():
            assert sha(dest) == sha(n)
        else:
            shutil.copyfile(n, dest)
        write(base / f'artifacts/data{d}_binding.json', dict(
            packaged_split_sha256=sha(source), historical_split_checksum=old_checksum,
            portable_split_checksum=m['checksum'], normalizer_sha256=sha(n),
            scope='Only path/provenance binding changes; split blocks and normalizer are unchanged.'
        ), existing_ok=True)
    print(f'Initialized {args.cohort}: five portable split/normalizer bindings.')


def data(args):
    generation, artifact, _, _ = COHORTS[args.cohort]
    module = importlib.import_module(generation + '_generate')
    roster = read(ROOT / f'configs/sim/{generation}_generation.json')
    assert roster == json.loads(json.dumps(module.design()))
    expected = {}
    for d in range(5):
        split = read(ROOT / f'artifacts/{artifact}/data{d}_split.json')
        for blocks in split['splits'].values():
            expected.update({b['file']: b['sha256'] for b in blocks})
    out = args.workspace / args.cohort / 'csv'
    out.mkdir(parents=True, exist_ok=True)
    for i in args.indices if args.indices is not None else range(370):
        if not 0 <= i < 370:
            raise ValueError('CSV index must be in 0..369')
        row = roster['rows'][i]
        path = out / (row['name'] + '.csv')
        if not path.exists():
            module.generate_one(roster, i, out)
        metadata = read(path.with_suffix('.generation.json'))
        assert metadata['row'] == row and metadata['params'] == roster['params']
        actual = sha(path)
        if actual != expected[row['name']] or actual != metadata['csv_sha256']:
            raise ValueError(f'CSV differs from accepted bytes: {path.name}. '
                             'Use the pinned environment; do not edit expected hashes.')
        print(f'CSV {i}: accepted hash matched', flush=True)


def branches(args):
    module = importlib.import_module(COHORTS[args.cohort][2])
    design = module.design()  # Bind to the distributed files, not historical source hashes.
    base = args.workspace / args.cohort
    write(base / 'branch_design.json', design, existing_ok=True)
    out = base / 'branches'
    out.mkdir(parents=True, exist_ok=True)
    for i in args.indices if args.indices is not None else range(80):
        if not 0 <= i < 80:
            raise ValueError('Branch index must be in 0..79')
        stem = out / f'flight_{i:03d}'
        if not stem.with_suffix('.json').exists():
            if stem.with_suffix('.npz').exists():
                raise FileExistsError(stem.with_suffix('.npz'))
            module.generate(design, i, base / 'csv', stem)
        gate = importlib.import_module('round15_data_accept').accept(
            out, design, 'portable-reproduction', [i])
        assert gate['accepted_files'] == 1
        print(f'Branch {i}: identity, replay, shapes, rotations and deltas passed', flush=True)


def validate_data(args):
    from latent_aero_wam.data.dataset import FlightStore, WindowDataset
    from latent_aero_wam.data.normalize import Normalizer
    from latent_aero_wam.data.splits import load_manifest
    from latent_aero_wam.evaluation.metrics import fit_metric_scales
    base = args.workspace / args.cohort
    for d in args.datasets:
        split_path = base / f'artifacts/data{d}_split.json'
        m = load_manifest(split_path)
        files = {b['file']: b['sha256'] for blocks in m['splits'].values() for b in blocks}
        assert len(files) == 74
        for name, digest in files.items():
            assert sha(base / 'csv' / (name + '.csv')) == digest, name
        store = FlightStore(m)
        ds = WindowDataset(m, 'd1_train', store=store)
        fitted = Normalizer.fit(ds.frames())
        expected = Normalizer.load(base / f'artifacts/data{d}_norm.json')
        for key in fitted.stats:
            np.testing.assert_allclose(fitted.stats[key].mean, expected.stats[key].mean, rtol=1e-7, atol=1e-8)
            np.testing.assert_allclose(fitted.stats[key].std, expected.stats[key].std, rtol=1e-7, atol=1e-8)
        scales = fit_metric_scales(ds)
        for key in scales:
            np.testing.assert_allclose(scales[key], expected.metric_scales[key], rtol=1e-7, atol=1e-8)
        result = dict(dataset=d, csv_files=74, training_windows=len(ds),
                      split_sha256=sha(split_path), fitted_normalizer_matches=True)
        if args.with_branches:
            design = read(base / 'branch_design.json')
            indices = [r['index'] for r in design['rows'] if r['dataset'] == d]
            gate = importlib.import_module('round15_data_accept').accept(
                base / 'branches', design, 'portable-reproduction', indices)
            coverage = dict(schema='round15-accepted-training-v1', dataset=d,
                            data_root=str(base / 'branches'), original_split_checksum=m['checksum'],
                            original_split_sha256=sha(split_path),
                            normalizer_sha256=sha(base / f'artifacts/data{d}_norm.json'),
                            design_sha256=sha(base / 'branch_design.json'),
                            protocol_sha256=design['protocol_sha256'], files=gate['files'])
            write(base / f'artifacts/data{d}_coverage.json', coverage, existing_ok=True)
            result.update(branch_files=gate['accepted_files'],
                          training_windows_per_regime=gate['accepted_windows_per_regime'])
        write(base / f'data{d}_accepted{"_branches" if args.with_branches else ""}.json', result,
              existing_ok=True)
        print(json.dumps(result), flush=True)


def checkpoint(e, args):
    cfg = runtime_config(e, args.workspace)
    if args.weights == 'retrained':
        path = args.workspace / 'runs' / cfg['experiment_name'] / f'{e["model"]}_seed{e["seed"]}' / 'checkpoint_best.pt'
        if not path.exists():
            raise FileNotFoundError(path)
        state = torch.load(path, map_location='cpu', weights_only=False)
        assert state['config'] == cfg, 'retrained configuration differs from the complete protocol'
        return path
    path = args.workspace / 'pretrained' / f'{e["model"]}_seed{e["seed"]}.pt'
    original = ROOT / e['export_path']
    assert sha(original) == e['export_sha256']
    provenance = dict(split_checksum=read(cfg['data']['manifest_path'])['checksum'],
                      normalizer_sha256=sha(cfg['data']['normalizer_path']),
                      original_checkpoint_sha256=e['original_checkpoint_sha256'],
                      tensor_export_sha256=e['export_sha256'], model_name=e['model'], seed=e['seed'])
    if path.exists():
        current = torch.load(path, map_location='cpu', weights_only=False)
        assert current['config'] == cfg and current['provenance'] == provenance
        exported = torch.load(original, map_location='cpu', weights_only=True)['model']
        assert current['model'].keys() == exported.keys()
        assert all(torch.equal(current['model'][k], v) for k, v in exported.items())
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        state = torch.load(original, map_location='cpu', weights_only=True)
        state.update(config=cfg, provenance=provenance)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
                temporary = Path(stream.name)
                torch.save(state, stream)
            try:
                os.link(temporary, path)
            except FileExistsError:
                # Another episode worker published this shared model first.
                return checkpoint(e, args)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return path


def training(args):
    e = entry(args.model, args.seed)
    cohort = cohort_of(e)
    environment(args.workspace, cohort)
    cfg = runtime_config(e, args.workspace)
    suffix = '_branches' if 'coverage_manifest' in cfg['data'] else ''
    gate_path = args.workspace / cohort / f'data{e["dataset"]}_accepted{suffix}.json'
    if not suffix and not gate_path.exists():
        # Full branch acceptance also certifies the same underlying CSV corpus.
        gate_path = args.workspace / cohort / f'data{e["dataset"]}_accepted_branches.json'
    gate = read(gate_path)
    assert gate['dataset'] == e['dataset'] and gate['csv_files'] == 74
    assert gate['split_sha256'] == sha(cfg['data']['manifest_path'])
    assert gate['fitted_normalizer_matches']
    if suffix:
        assert gate['branch_files'] == 16 and gate['training_windows_per_regime'] == 12288
    compute = load_config(ROOT / 'configs/compute/compute_round15.yaml')
    compute.update(device=args.device, num_workers=args.workers)
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; use --device cpu explicitly.')
    branch = 'smoke_runs' if args.smoke else 'runs'
    if args.smoke:
        cfg['optim'].update(max_epochs=1, early_stopping_patience=2)
        cfg['data'].update(training_samples_per_epoch=512, eval_stride=2850)
    run_dir = args.workspace / branch / cfg['experiment_name'] / f'{e["model"]}_seed{e["seed"]}'
    if run_dir.exists():
        raise FileExistsError(f'Use a new workspace; existing run is protected: {run_dir}')
    run_dir.mkdir(parents=True)
    write(run_dir / 'reproduction_config.json', dict(config=cfg, compute=compute, smoke=args.smoke))
    result = train(cfg, compute, run_dir, resume=False)
    write(run_dir / 'train_result.json', dict(result=result, smoke=args.smoke,
                                             environment=execution_environment(args.device)))
    print(json.dumps(dict(model=e['model'], seed=e['seed'], smoke=args.smoke, result=result)))


def offline(args):
    e = entry(args.model, args.seed)
    environment(args.workspace, cohort_of(e))
    path = checkpoint(e, args)
    out = args.workspace / ('smoke_results' if args.smoke else 'results') / args.weights / 'offline'
    target = out / f'{e["model"]}_seed{e["seed"]}_{args.split}.json'
    if target.exists():
        raise FileExistsError(target)
    result = evaluate(path, args.split, torch.device(args.device), batch_size=256,
                      stride=1, amp='off', export_latents=False,
                      max_windows=256 if args.smoke else None)
    winds = result.per_window['wind_mps']
    records = [dict(wind=float(w), windows=int((winds == w).sum()),
                    original_E=float(result.per_window['err_aggregate'][winds == w].mean()))
               for w in sorted(set(winds.tolist()))]
    write(target, dict(model=e['model'], seed=e['seed'], dataset=e['dataset'],
                       checkpoint_sha256=sha(path), smoke=args.smoke, weights=args.weights,
                       summary=result.summary(), wind_records=records,
                       environment=execution_environment(args.device)))
    if args.save_traces:
        np.savez_compressed(target.with_suffix('.npz'), **result.per_window)
    print(json.dumps(dict(output=target.name, wind_records=records)))


def control(args):
    from round11_closed_loop import run
    roster, _ = plan(args.study)
    row = roster['rows'][args.index]
    assert row['index'] == args.index
    out = args.workspace / ('smoke_results' if args.smoke else 'results') / args.weights / args.study / 'control'
    target = out / f'episode_{args.index:05d}.json'
    if target.exists():
        raise FileExistsError(target)
    path = checkpoint(entry(row['model'], row['model_seed']), args) if row['kind'] == 'model' else None
    result = run(row, path, args.device, scored_steps=3 if args.smoke else 1500)
    write(target, dict(result=result, smoke=args.smoke, weights=args.weights,
                       checkpoint_sha256=sha(path) if path else None,
                       roster_sha256=sha(ROOT / f'configs/sim/{STUDIES[args.study]}.json'),
                       environment=execution_environment(args.device)))
    print(json.dumps(dict(output=target.name, rmse=result['capped_tracking_rmse_m'],
                         failed=result['failed'], smoke=args.smoke)))


def query(args):
    module = importlib.import_module('round16_evaluate' if args.study == 'bridge' else 'round18_evaluate')
    roster, _ = plan(args.study)
    target = args.workspace / 'results' / args.weights / args.study / 'query' / f'anchor_{args.index:02d}.json'
    if target.exists():
        raise FileExistsError(target)
    def load(spec):
        x = spec['identity']
        model, _, art = load_model_from_checkpoint(checkpoint(entry(spec['model'], spec['model_seed']), args), torch.device(args.device))
        return model, art['normalizer'].metric_scales, dict(arm=x['arm'], coverage=x['coverage'])
    specs = [dict(model=x['config']['model']['name'], model_seed=x['seed'],
                  dataset_replicate=x['dataset'], identity=x) for x in roster['models']]
    snapshot, bank, row = module.fresh_anchor(args.index)
    result, trace = module.diag.evaluate_anchor(snapshot, bank, specs, load, row, 8)
    write(target, dict(result=result, row=row, weights=args.weights,
                       roster_sha256=sha(ROOT / f'configs/sim/{STUDIES[args.study]}.json'),
                       environment=execution_environment(args.device)))
    if args.save_traces:
        np.savez_compressed(target.with_suffix('.npz'), **trace)
    assert all(x['status'] == 'complete' for x in result['models']), 'failed queries retained; investigate before summarization'
    print(json.dumps(dict(output=target.name, models=len(result['models']))))


def summarize(args):
    """Require the complete declared roster before reporting dataset-level effects."""
    roster, entries = plan(args.study, args.panel)
    base = args.workspace / 'results' / args.weights
    inputs = {}
    def checked(path):
        obj = read(path)
        assert not obj.get('smoke', False) and obj['weights'] == args.weights
        inputs[str(path.relative_to(args.workspace))] = sha(path)
        return obj
    if args.panel == 'control':
        results = []
        expected_roster = sha(ROOT / f'configs/sim/{STUDIES[args.study]}.json')
        for row in roster['rows']:
            obj = checked(base / args.study / 'control' / f'episode_{row["index"]:05d}.json')
            assert obj['roster_sha256'] == expected_roster and obj['result']['row'] == row
            results.append(obj['result'])
        if args.study == 'original':
            summary = importlib.import_module('round11d_results').aggregate(results)
        else:
            module = importlib.import_module('round16_results' if args.study == 'bridge' else 'round18_results')
            records = [module.control_record(r) for r in results]
            summary = {m: module.factorial(records, m) for m in ['rmse_m', 'failed']}
    elif args.panel == 'offline':
        records = {}
        for e in entries:
            obj = checked(base / 'offline' / f'{e["model"]}_seed{e["seed"]}_d2_static_ood.json')
            assert (obj['model'], obj['seed'], obj['dataset']) == (e['model'], e['seed'], e['dataset'])
            assert obj['summary']['n_windows'] == 85500
            assert len(obj['wind_records']) == 3 and all(x['windows'] == 28500 for x in obj['wind_records'])
            records[(e['model'], e['seed'])] = {round(r['wind'], 1): r['original_E'] for r in obj['wind_records']}
        summary = {}
        for wind in [3.7, 6.1, 8.5]:
            if args.study == 'original':
                module = importlib.import_module('round11d_results')
                means = {arm: np.asarray([np.mean([records[(f'r11b_d{d}_{arm}', s)][wind]
                         for s in [70, 71, 72]]) for d in range(5)]) for arm in module.ARMS}
                summary[str(wind)] = dict(arm_dataset_scores={k: v.tolist() for k, v in means.items()},
                    updated_minus_frozen_no_norm=module.describe(means['updated_r0'] - means['frozen_r0']))
            else:
                module = importlib.import_module('round16_results' if args.study == 'bridge' else 'round18_results')
                means = {}
                cells = sorted({x['coverage'] for x in roster['models']})
                for cell in cells:
                    for arm in ['frozen', 'updated']:
                        means[cell + '/' + arm] = np.asarray([np.mean([
                            records[(x['config']['model']['name'], x['seed'])][wind]
                            for x in roster['models'] if (x['dataset'], x['coverage'], x['arm']) == (d, cell, arm)
                        ]) for d in range(5)])
                summary[str(wind)] = module.summarize(means, cells)
    else:
        if args.study == 'original':
            raise ValueError('Use the preserved original-query protocols; this entry point covers fresh bridge/confirmation queries.')
        module = importlib.import_module('round16_results' if args.study == 'bridge' else 'round18_results')
        records = []
        identities = {(x['dataset'], x['coverage'], x['arm'], x['seed']) for x in roster['models']}
        for index in range(40):
            obj = checked(base / args.study / 'query' / f'anchor_{index:02d}.json')
            row = obj['row']
            assert row['index'] == index and row['episode'] == index % 10
            models = obj['result']['models']
            assert len(models) == len(identities)
            assert {(x['dataset_replicate'], x['coverage'], x['arm'], x['model_seed']) for x in models} == identities
            for x in models:
                assert x['status'] == 'complete'
                records.append(dict(wind=row['wind'], episode=row['episode'], dataset=x['dataset_replicate'],
                    coverage=x['coverage'], arm=x['arm'], seed=x['model_seed'],
                    query_E=x['horizon_metrics']['50']['original_E']['prefix_mean'],
                    position_m=x['horizon_metrics']['50']['physical_position_l2_m']['prefix_mean'],
                    physical_solution_cost=x['physical_solution_cost']))
        summary = {m: module.factorial(records, m) for m in ['query_E', 'position_m', 'physical_solution_cost']}
    output = base / f'{args.study}_{args.panel}_summary.json'
    write(output, dict(study=args.study, panel=args.panel, weights=args.weights, inputs_sha256=inputs,
                       summary=summary, uncertainty_unit='five independent training datasets; descriptive nonsimultaneous t95 intervals'))
    print(f'Complete-roster summary written: {output.name}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=Path('reproduction'))
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ['init', 'data', 'branches', 'validate-data']:
        p = sub.add_parser(name)
        p.add_argument('--cohort', choices=COHORTS, default='original')
        if name in ['data', 'branches']:
            p.add_argument('--indices', type=int, nargs='+')
        if name == 'validate-data':
            p.add_argument('--datasets', type=int, nargs='+', choices=range(5), default=list(range(5)))
            p.add_argument('--with-branches', action='store_true')
    p = sub.add_parser('plan')
    p.add_argument('--study', choices=STUDIES, required=True)
    p.add_argument('--stage', choices=['train', 'offline', 'control', 'query'], default='control')
    p = sub.add_parser('summarize')
    p.add_argument('--study', choices=STUDIES, required=True)
    p.add_argument('--panel', choices=['offline', 'control', 'query'], required=True)
    p.add_argument('--weights', choices=['pretrained', 'retrained'], default='pretrained')
    for name in ['train', 'offline', 'control', 'query']:
        p = sub.add_parser(name)
        p.add_argument('--device', default='cpu' if name == 'query' else 'cuda')
        if name != 'query':
            p.add_argument('--smoke', action='store_true')
        if name in ['train', 'offline']:
            p.add_argument('--model', required=True)
            p.add_argument('--seed', type=int, choices=[70, 71, 72], required=True)
        else:
            p.add_argument('--study', choices=STUDIES if name == 'control' else ['bridge', 'confirmation'], required=True)
            p.add_argument('--index', type=int, required=True)
        if name == 'train':
            p.add_argument('--workers', type=int, default=2)
        else:
            p.add_argument('--weights', choices=['pretrained', 'retrained'], default='pretrained')
        if name in ['offline', 'query']:
            p.add_argument('--save-traces', action='store_true')
        if name == 'offline':
            p.add_argument('--split', choices=['d1_val', 'd2_static_ood', 'd3_changing_ood'], default='d2_static_ood')
    args = parser.parse_args()
    args.workspace = args.workspace.resolve()
    if args.workspace == ROOT or ROOT.is_relative_to(args.workspace):
        parser.error('workspace must be a separate directory, not the repository or its parent')
    torch.set_num_threads(1)
    if getattr(args, 'device', '').startswith('cuda') and not torch.cuda.is_available():
        parser.error('CUDA unavailable; use --device cpu explicitly')
    if args.command == 'plan':
        roster, entries = plan(args.study, args.stage)
        print(json.dumps(dict(study=args.study, models=[dict(model=e['model'], seed=e['seed'], dataset=e['dataset']) for e in entries],
                              control_indices=list(range(len(roster['rows'])))), indent=2))
        return
    environment(args.workspace, getattr(args, 'cohort', 'original'))
    handlers = {'init': init, 'data': data, 'branches': branches, 'validate-data': validate_data,
                'train': training, 'offline': offline, 'control': control, 'query': query,
                'summarize': summarize}
    handlers[args.command](args)
    print(json.dumps(dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__)))


if __name__ == '__main__':
    main()
