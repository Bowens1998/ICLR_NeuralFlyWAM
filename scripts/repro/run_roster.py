"""Run or shard a complete simulation roster through the portable entry point."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from run_study import ROOT, plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=Path('reproduction'))
    parser.add_argument('--study', choices=['original', 'bridge', 'confirmation'], required=True)
    parser.add_argument('--stage', choices=['train', 'offline', 'control', 'query'], required=True)
    parser.add_argument('--indices', type=int, nargs='+', help='Model indices for train/offline; episode/anchor indices otherwise.')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--weights', choices=['pretrained', 'retrained'], default='pretrained')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.stage == 'query' and args.study == 'original':
        parser.error('query stage is available for bridge and confirmation')
    roster, entries = plan(args.study, args.stage)
    count = len(entries) if args.stage in ['train', 'offline'] else (40 if args.stage == 'query' else len(roster['rows']))
    indices = args.indices if args.indices is not None else list(range(count))
    if len(set(indices)) != len(indices) or any(i < 0 or i >= count for i in indices):
        parser.error(f'indices must be unique and in 0..{count-1}')
    for i in indices:
        cmd = [sys.executable, str(ROOT / 'scripts/repro/run_study.py'), '--workspace',
               str(args.workspace.resolve()), args.stage, '--device', args.device]
        if args.stage in ['train', 'offline']:
            e = entries[i]
            cmd += ['--model', e['model'], '--seed', str(e['seed'])]
        else:
            cmd += ['--study', args.study, '--index', str(i)]
        if args.stage != 'train':
            cmd += ['--weights', args.weights]
        print(json.dumps(dict(index=i, command=cmd)), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
