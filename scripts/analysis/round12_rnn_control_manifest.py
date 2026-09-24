"""Build a control manifest only after complete, SHA-bound formal RNN acceptance.

The checkpoint source is the training source in the acceptance report, never the
current control release or LATENT_WAM_SOURCE_COMMIT environment value. All sixty
runs and all 180 prediction NPZ files must still match their accepted bytes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO/'scripts/analysis'))
sys.path.insert(0, str(REPO/'scripts/sim'))
sys.path.insert(0, str(REPO/'src'))
import round12_control_protocol as controls  # noqa: E402
import round12_rnn_accept as training  # noqa: E402
import round12_rnn_results as prediction  # noqa: E402

from latent_aero_wam.data.normalize import Normalizer  # noqa: E402
from latent_aero_wam.data.splits import load_manifest  # noqa: E402
from latent_aero_wam.models import build_model  # noqa: E402
from latent_aero_wam.utils.config import config_hash, load_config  # noqa: E402

CHECKPOINT_FIELDS = {'model', 'optimizer', 'scheduler', 'scaler', 'epoch', 'best',
                     'bad_epochs', 'config', 'compute', 'provenance', 'val_metrics', 'rng'}
require = training.require


def safe_file(root, relative):
    root, relative = Path(root).resolve(), Path(relative)
    require(not relative.is_absolute() and '..' not in relative.parts,
            'accepted artifact must be a safe relative path')
    target = (root/relative).resolve()
    require(target.is_relative_to(root), 'accepted artifact escapes runs root through a symlink')
    return target


def checkpoint_entry(task, record, runs_root, source, repo=REPO):
    """Recheck model identity while preserving the exact saved configuration."""
    run = safe_file(runs_root, task['run'])
    cfg = yaml.safe_load((run/'resolved_config.yaml').read_text())
    require(training.normalize_config(cfg, runs_root, repo)
            == training.normalize_config(task['config'], runs_root, repo),
            'saved RNN scientific configuration differs from the formal roster')
    require(cfg['model']['family'] == 'context_rnn_control' and cfg['seed'] in (80, 81, 82),
            'control manifest requires formal RNN checkpoints')
    compute = yaml.safe_load((run/'resolved_compute.yaml').read_text())
    require(compute == load_config(repo/task['compute_path']) and compute['amp'] == 'off',
            'saved RNN compute/FP32 profile changed')
    split_path = repo/task['config']['data']['manifest_path']
    norm_path = repo/task['config']['data']['normalizer_path']
    split = load_manifest(split_path)
    require(split['dataset_replicate'] == task['dataset'] and split['history_steps'] == 100
            and split['horizon_steps'] == 50 and split['dt'] == .02,
            'RNN training data timing/dataset identity changed')
    provenance = training.read_json(run/'provenance.json')
    expected = dict(git_commit=source, model_name=cfg['model']['name'], seed=task['seed'],
                    config_hash=config_hash(cfg), split_checksum=split['checksum'],
                    normalizer_sha256=training.sha256(norm_path), n_parameters=28489)
    require(provenance == expected, 'RNN training provenance differs from accepted identity')
    result = training.read_json(run/'train_result.json')
    require({key: result.get(key) for key in expected} == expected,
            'RNN training-result identity differs from provenance')
    path = run/'checkpoint_best.pt'
    state = torch.load(path, map_location='cpu', weights_only=True)
    require(set(state) == CHECKPOINT_FIELDS, 'RNN checkpoint schema changed')
    require(state['config'] == cfg and state['compute'] == compute and state['provenance'] == provenance,
            'RNN checkpoint config/compute/provenance differs from accepted run')
    selected = record['training']
    require(type(state['epoch']) is int and state['epoch'] == selected['best_epoch'],
            'RNN checkpoint differs from the accepted ID-selected epoch')
    require(np.isfinite(state['best'])
            and np.isclose(state['best'], selected['best_val_aggregate'], rtol=1e-10, atol=1e-12)
            and np.isclose(result['best_val_aggregate'], state['best'], rtol=1e-10, atol=1e-12),
            'RNN selected ID score differs from acceptance')
    require(state['bad_epochs'] == 0 and state['scaler'] == {},
            'best FP32 checkpoint must have zero stopping count and no scaler state')
    normalizer = Normalizer.load(norm_path)
    model = build_model(cfg['model'], normalizer, .02)
    reference = model.state_dict()
    require(state['model'].keys() == reference.keys(), 'RNN checkpoint model keys differ')
    require(all(isinstance(value, torch.Tensor) and value.shape == reference[key].shape
                and value.dtype == reference[key].dtype and torch.isfinite(value).all()
                for key, value in state['model'].items()), 'RNN checkpoint tensor shape/dtype/finiteness changed')
    model.load_state_dict(state['model'], strict=True)
    require(model.n_parameters() == 28489, 'RNN architecture parameter count changed')
    for group in Normalizer.GROUPS:
        for statistic in ('mean', 'std'):
            require(np.array_equal(getattr(model, f'{group}_{statistic}').numpy(),
                                   getattr(normalizer.stats[group], statistic)),
                    'RNN checkpoint normalizer buffers differ from accepted data')
    return dict(dataset_replicate=task['dataset'], model_seed=task['seed'],
                model_name=cfg['model']['name'], update_context=cfg['model']['update_context'],
                context_norm=cfg['model']['context_norm'],
                checkpoint_relpath=f"{task['run']}/checkpoint_best.pt",
                checkpoint_sha256=training.sha256(path), source_commit=source,
                config=cfg, config_hash=config_hash(cfg), config_sha256=controls.object_hash(cfg),
                split_checksum=expected['split_checksum'], normalizer_sha256=expected['normalizer_sha256'])


def build_manifest(runs_root, acceptance_path, repo=REPO):
    accepted = training.read_json(acceptance_path)
    source = accepted['source']  # Deliberately independent of the current execution release.
    require(accepted.get('stage') == 'formal', 'pilot checkpoints cannot enter the control manifest')
    for record in accepted.get('records', []):
        for item in record.get('files', []):
            safe_file(runs_root, item['path'])
    accepted, tasks = prediction.verify_acceptance(acceptance_path, runs_root, source, repo)
    records = {record['run']: record for record in accepted['records']}
    entries = [checkpoint_entry(task, records[task['run']], runs_root, source, repo) for task in tasks]
    manifest = dict(schema='round12-checkpoints-v1', family_id='round12_rnn',
                    model_seeds=[80, 81, 82], reuse_round11_learned=False, checkpoints=entries,
                    training_acceptance_sha256=training.sha256(acceptance_path),
                    training_source_commit=source, accepted_training_runs=60, accepted_prediction_npz=180,
                    accepted_files_rehashed=sum(len(record['files']) for record in accepted['records']),
                    accepted_repository_inputs=accepted['repository_inputs'],
                    manifest_builder_sha256=training.sha256(Path(__file__)))
    controls.validate_manifest(manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs-root', type=Path, required=True)
    parser.add_argument('--acceptance', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = build_manifest(args.runs_root, args.acceptance)
    training.write_new(args.output, manifest)
    print(json.dumps(dict(family_id=manifest['family_id'], checkpoints=len(manifest['checkpoints']),
                          training_source_commit=manifest['training_source_commit'],
                          manifest_sha256=training.sha256(args.output), output=str(args.output))))


if __name__ == '__main__':
    main()
