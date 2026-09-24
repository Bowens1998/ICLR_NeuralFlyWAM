"""Fixed crossed-trajectory Round11B generation; never overwrites a flight.

--write-design creates the immutable roster. --index executes one CPU task.
Train/validation/test trajectories are wholly disjoint, not temporal fragments.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO/'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import generate_data as writer  # noqa: E402

from latent_aero_wam.sim import QuadrotorSim, random_trajectory  # noqa: E402
from latent_aero_wam.sim.quadrotor import SimParams  # noqa: E402

CONDITIONS = [('nowind', 0.), ('10wind', 1.3), ('20wind', 2.5), ('40wind', 4.9)]


def design():
    rows = []
    for replicate in range(5):
        for group, ntraj, conditions in [
            ('train', 4, CONDITIONS), ('val', 2, CONDITIONS),
            ('static', 10, [('30wind', 3.7), ('50wind', 6.1), ('70wind', 8.5)]),
            ('slow', 10, [('35wind', 4.2)]), ('fast', 10, [('35wind', 4.2)]),
        ]:
            # Test families share trajectories and noise streams for paired conditions.
            offset = {'train': 0, 'val': 100, 'static': 200, 'slow': 200, 'fast': 200}[group]
            for j in range(ntraj):
                trajectory_seed = 110000+replicate*1000+offset+j
                noise_seed = 210000+replicate*1000+offset+j
                for condition, mean in conditions:
                    rows.append(dict(index=len(rows), replicate=replicate, group=group,
                                     trajectory_seed=trajectory_seed, noise_seed=noise_seed,
                                     name=f'r11d{replicate}_t{trajectory_seed}_{group}_{condition}',
                                     condition=condition, mean_wind=mean,
                                     wind_amplitude=2.4 if group in ('slow', 'fast') else 0.,
                                     wind_angular_frequency={'slow': .25, 'fast': 1.}.get(group, 0.),
                                     duration_seconds=60.))
    return dict(protocol='docs/ROUND11_PROTOCOL.md', params=asdict(SimParams()),
                model_seeds=[70, 71, 72], rows=rows,
                caveat='Dynamic CSV wind label is nominal4.2; exact wind formula in this roster.')


def generate_one(roster, index, output):
    row = roster['rows'][index]
    assert row['index'] == index
    output = Path(output)
    path = output/(row['name']+'.csv')
    evidence = output/(row['name']+'.generation.json')
    if path.exists() or evidence.exists():
        raise FileExistsError(f'preserve existing generation artifacts: {path}')
    pp = SimParams(**roster['params'])
    rng = np.random.default_rng(row['trajectory_seed'])
    trajectory = random_trajectory(rng, row['duration_seconds'], pp.dt)
    sim = QuadrotorSim(pp, seed=row['noise_seed'])
    def wind_fn(t):
        speed = row['mean_wind']+row['wind_amplitude']*np.sin(row['wind_angular_frequency']*t)
        return np.array([speed, 0., 0.])
    start = time.monotonic()
    logs = sim.fly(trajectory, wind_fn)
    for key, value in logs.items():
        assert np.isfinite(value).all(), key
    writer.OUT = output
    writer.write_flight(row['name'], logs, trajectory, np.gradient(trajectory, pp.dt, axis=0))
    elapsed = time.monotonic()-start
    result = dict(row=row, params=roster['params'], seconds=elapsed,
                  source_commit=os.environ.get('LATENT_WAM_SOURCE_COMMIT', 'local-development'),
                  csv_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  trajectory_sha256=hashlib.sha256(trajectory.tobytes()).hexdigest())
    evidence.write_text(json.dumps(result, indent=2)+'\n')
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--design', type=Path, default=REPO/'configs/sim/round11_generation.json')
    ap.add_argument('--write-design', action='store_true')
    ap.add_argument('--index', type=int)
    ap.add_argument('--output', type=Path)
    args = ap.parse_args()
    if args.write_design:
        args.design.parent.mkdir(parents=True, exist_ok=True)
        with args.design.open('x') as f:
            f.write(json.dumps(design(), indent=2)+'\n')
        return
    if args.index is None or args.output is None:
        ap.error('--index and --output required')
    roster = json.loads(args.design.read_text())
    assert roster == json.loads(json.dumps(design())), 'roster differs from locked design'
    print(json.dumps(generate_one(roster, args.index, args.output)))


if __name__ == '__main__':
    main()
