"""Replay a complete existing query anchor from portable weights and simulator code.

The default is the first episode at the first primary wind (anchor20,6.1m/s).
It is fixed before local replay, not selected by score. This checks120 models
and three outcomes each; it does not retrain or reproduce the feedback panel.
"""

# ruff: noqa: E402 -- standalone portable-package entry point.
import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts/sim'))
import round18_evaluate as evaluation

from latent_aero_wam.data.normalize import Normalizer
from latent_aero_wam.models import build_model

# Fixed before replay. Allows ordinary cross-version CPU FP32 roundoff.
RTOL = 1e-4
ATOL = 1e-6


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--anchor', type=int, choices=range(40), default=20)
    parser.add_argument('--output', type=Path, default=Path('REPLAY_CONFIRMATION_QUERY.json'))
    args = parser.parse_args()
    package = json.loads((ROOT / 'PACKAGE_MANIFEST.json').read_text())
    for name, digest in package['files_sha256'].items():
        assert sha(ROOT / name) == digest, name
    assert package['checkpoint_count'] == 450
    exports = json.loads((ROOT / 'CHECKPOINT_EXPORTS.json').read_text())
    by_path = {entry['export_path']: entry for entry in exports}
    roster = json.loads((ROOT / 'configs/sim/round18_evaluation.json').read_text())
    assert len(roster['models']) == 120
    torch.set_num_threads(1)
    specs = []
    for spec in roster['models']:
        entry = by_path['checkpoints/' + spec['path']]
        assert entry['original_checkpoint_sha256'] == spec['checkpoint_sha256']
        assert entry['config'] == spec['config']
        specs.append(dict(model=spec['config']['model']['name'], dataset_replicate=spec['dataset'],
                          model_seed=spec['seed'], identity=spec, export=entry))

    def load(specification):
        entry, spec = specification['export'], specification['identity']
        path = ROOT / entry['export_path']
        assert sha(path) == entry['export_sha256']
        normalizer = Normalizer.load(ROOT / entry['config']['data']['normalizer_path'])
        split = json.loads((ROOT / entry['config']['data']['manifest_path']).read_text())
        model = build_model(entry['config']['model'], normalizer, float(split['dt']))
        state = torch.load(path, map_location='cpu', weights_only=True)['model']
        model.load_state_dict(state, strict=True)
        assert model.n_parameters() == 39497
        assert all(torch.isfinite(value).all() for value in model.state_dict().values())
        model.eval()
        return model, normalizer.metric_scales, dict(checkpoint_sha256=spec['checkpoint_sha256'],
                                                     arm=spec['arm'], coverage=spec['coverage'])

    snapshot, bank, row = evaluation.fresh_anchor(args.anchor)
    result, trace = evaluation.diag.evaluate_anchor(snapshot, bank, specs, load, row, 8)
    del trace
    assert len(result['models']) == 120 and all(r['status'] == 'complete' for r in result['models'])
    accepted = json.loads((ROOT / 'reports/ROUND18_QUERY_RESULTS.json').read_text())
    records = [r for r in accepted['records'] if (r['wind'], r['episode']) == (row['wind'], row['episode'])]
    assert len(records) == 120
    expected = {(r['dataset'], r['coverage'], r['arm'], r['seed']): r for r in records}
    differences = []
    for replayed in result['models']:
        key = (replayed['dataset_replicate'], replayed['coverage'], replayed['arm'], replayed['model_seed'])
        target = expected[key]
        actual = dict(physical_solution_cost=replayed['physical_solution_cost'],
                      query_E=replayed['horizon_metrics']['50']['original_E']['prefix_mean'],
                      position_m=replayed['horizon_metrics']['50']['physical_position_l2_m']['prefix_mean'])
        for metric, value in actual.items():
            np.testing.assert_allclose(value, target[metric], rtol=RTOL, atol=ATOL,
                                       err_msg=f'{key}: {metric}')
            differences.append(dict(identity=list(key), metric=metric, reproduced=value,
                                    accepted=target[metric], absolute_error=abs(value-target[metric])))
    assert len(differences) == 360
    report = dict(anchor=args.anchor, wind=row['wind'], episode=row['episode'], models=120,
                  checked_values=360, relative_tolerance=RTOL, absolute_tolerance=ATOL,
                  maximum_absolute_error=max(d['absolute_error'] for d in differences),
                  comparisons=differences, python=platform.python_version(), torch=torch.__version__,
                  numpy=np.__version__, device='cpu', precision='FP32',
                  package_manifest_sha256=sha(ROOT/'PACKAGE_MANIFEST.json'),
                  scope='Fresh physical/query computation from exact portable tensors, for one complete published anchor. No raw trace or training CSV input. Not a retraining or full closed-loop reproduction.')
    with args.output.open('x') as stream:
        stream.write(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'comparisons'}))


if __name__ == '__main__':
    main()
