"""Round12A diagnostic roster: no training, immutable Round11B checkpoints."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import round11_closed_loop_protocol as original

ARMS = ('frozen_r0', 'frozen_r1', 'updated_r0', 'updated_r1')
DT, HISTORY, HORIZON, CANDIDATES = .02, 100, 50, 64
MC_SAMPLES = 8
A2_STEPS = (350, 850, 1350)
ACTION_ATOL = 1e-4


def model_rows(wind, episode, dataset=None, seed=None):
    return [r for r in original.design()['rows'] if r['kind'] == 'model'
            and r['wind'] == wind and r['episode'] == episode
            and (dataset is None or r['dataset_replicate'] == dataset)
            and (seed is None or r['model_seed'] == seed)]


def design():
    rows = original.design()['rows']
    a1 = [dict(index=i, stage='a1', wind=r['wind'], episode=r['episode'],
               donor_index=r['index'], anchor_steps=[100])
          for i, r in enumerate(r for r in rows if r['kind'] == 'pd')]
    donors = [r for r in rows if r['kind'] == 'model'
              and r['model'].endswith(('frozen_r0', 'updated_r0'))
              and r['wind'] in (0., 6.1, 8.5) and r['episode'] in (0, 1)]
    a2 = [dict(index=i, stage='a2', donor_index=r['index'], wind=r['wind'],
               episode=r['episode'], dataset_replicate=r['dataset_replicate'],
               model_seed=r['model_seed'], owner=r['model'], anchor_steps=list(A2_STEPS))
          for i, r in enumerate(donors)]
    return dict(version='round12a_v1', dt=DT, history=HISTORY, horizon=HORIZON,
                candidates=CANDIDATES, mc_samples=MC_SAMPLES, action_atol=ACTION_ATOL,
                primary_contrast='updated_r0 minus frozen_r0; negative favors updated',
                primary_metric='physical_solution_cost_excess_vs_oracle_softmin',
                oracle='same-bank expected physical cost; independent construction/evaluation noise',
                noise_seed_namespace=12012026, a1=a1, a2=a2)


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def checkpoint_path(root, row):
    return Path(root)/f"round11b_data{row['dataset_replicate']}_v1"/f"{row['model']}_seed{row['model_seed']}"/'checkpoint_best.pt'


def noise_seed_words(stage, index, step, phase):
    if phase not in ('construct', 'evaluate'):
        raise ValueError('unknown independent noise phase')
    return [12012026, {'a1': 1, 'a2': 2}[stage], int(index), int(step),
            {'construct': 1, 'evaluate': 2}[phase]]
