"""Validate full and failed Round11D episodes without selecting successful outcomes."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'sim'))
import round11_closed_loop_protocol as protocol  # noqa: E402

SOURCE = '87a0333910cef20b4ff2d5918cfdbfe2dba7a4b0'


def validate_episode(result, claim, row, roster_sha256, checkpoint_sha256):
    r = result
    assert r['row'] == claim['row'] == row
    assert r['source_commit'] == claim['source'] == SOURCE
    assert r['roster_sha256'] == roster_sha256
    assert r['checkpoint_sha256'] == checkpoint_sha256
    assert r['scored_steps'] == 1500
    errors = [np.nan if e is None else e for e in r['errors_m']]
    expected = protocol.score(errors, r['failed'])
    for k, value in expected.items():
        if value is None:
            assert r[k] is None
        else:
            assert np.isclose(r[k], value, rtol=1e-10, atol=1e-12), k
    actions = np.asarray(r['actions'], dtype=float).reshape(-1, 5)
    assert np.isfinite(actions).all()
    assert 0 <= len(actions) <= 1600
    assert r['observed_effort_steps'] == max(0, len(actions)-100)
    assert r['observed_steps'] == len(errors) == r['observed_effort_steps']
    if r['failed']:
        assert isinstance(r['failure_reason'], str) and r['failure_reason']
        assert isinstance(r['failure_step'], int) and 0 <= r['failure_step'] < 1600
        assert len(actions) in [r['failure_step'], r['failure_step']+1]
    else:
        assert len(actions) == 1600
        assert r['failure_reason'] is None and r['failure_step'] is None
    lat = np.asarray(r['planning_seconds'], dtype=float)
    assert lat.ndim == 1 and np.isfinite(lat).all() and (lat > 0).all()
    if row['kind'] == 'pd':
        assert len(lat) == 0
    else:
        assert len(lat) in [len(errors), len(errors)+int(r['failed'])]
    if len(lat):
        assert np.isclose(lat.mean(), r['planning_mean_seconds'])
        assert np.isclose(np.quantile(lat, .95), r['planning_p95_seconds'])
    else:
        assert r['planning_mean_seconds'] is None and r['planning_p95_seconds'] is None
    assert np.isfinite(r['elapsed_seconds']) and r['elapsed_seconds'] > sum(lat)
    if r['warmup_state'] is not None:
        assert all(np.isfinite(v).all() for v in r['warmup_state'].values())
        assert 0 < r['warmup_seconds'] < r['elapsed_seconds']
    if len(errors):
        # Quaternion command q=(x,y,z,w); tilt angle against world z.
        q = actions[100:, 1:]
        np.testing.assert_allclose(np.linalg.norm(q, axis=1), 1., atol=1e-6, rtol=0)
        tilt = np.arccos(np.clip(1-2*(q[:, 0]**2+q[:, 1]**2), -1, 1))
        assert np.isclose(np.sqrt(np.mean(tilt**2)), r['command_tilt_rms_rad'])
        assert np.isclose(np.sqrt(np.mean((actions[100:, 0]-.3924)**2)), r['throttle_deviation_rms'])
    else:
        assert r['command_tilt_rms_rad'] is None and r['throttle_deviation_rms'] is None
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--group', choices=['all', 'reference']+[f'data{d}' for d in range(5)], required=True)
    ap.add_argument('--checkpoint-records', type=Path, nargs='*', default=[])
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    roster_path = Path(__file__).resolve().parents[2]/'configs/sim/round11_closed_loop.json'
    roster = json.loads(roster_path.read_text())
    assert roster == protocol.design()
    roster_hash = hashlib.sha256(roster_path.read_bytes()).hexdigest()
    indices = (range(2560) if args.group == 'all' else range(2400, 2560)
               if args.group == 'reference' else range(int(args.group[-1])*480, (int(args.group[-1])+1)*480))
    checkpoint_hashes = {}
    for path in args.checkpoint_records:
        for line in path.read_text().splitlines():
            if not line.startswith('{'):
                continue
            record = json.loads(line)
            key = record['index']//40
            digest = record['checkpoint_sha256']
            assert key not in checkpoint_hashes or checkpoint_hashes[key] == digest
            checkpoint_hashes[key] = digest
    warmups = {}
    files = []
    failures = 0
    for i in indices:
        path = args.root/f'episode_{i:04d}.json'
        r = json.loads(path.read_text())
        claim = json.loads(path.with_suffix('.claim').read_text())
        expected_checkpoint = checkpoint_hashes[i//40] if i < 2400 else None
        validate_episode(r, claim, roster['rows'][i], roster_hash, expected_checkpoint)
        key = (r['row']['wind'], r['row']['episode'])
        if r['warmup_state'] is not None:
            if key in warmups:
                previous = warmups[key]
                for field, value in r['warmup_state'].items():
                    np.testing.assert_allclose(value, previous['warmup_state'][field], atol=1e-12, rtol=0)
                np.testing.assert_allclose(r['actions'][:100], previous['actions'][:100], atol=1e-12, rtol=0)
            else:
                warmups[key] = r
        failures += int(r['failed'])
        files.append(dict(index=i, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    report = dict(group=args.group, accepted_episodes=len(files), failed_episodes_retained=failures,
                  source=SOURCE, roster_sha256=roster_hash, files=files,
                  primary_analysis_run=False, cross_group_warmup_check=args.group == 'all')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        f.write(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'files'}))


if __name__ == '__main__':
    main()
