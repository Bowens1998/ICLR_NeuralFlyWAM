"""Canonical metadata interventions, independent of minibatch boundaries."""
import numpy as np


def wind_vector(dataset):
    return np.asarray([dataset.store.get(name).wind_mps for name, _ in dataset.index],
                      dtype=np.float32)


def intervention(wind, mode, median, seed=None):
    wind = np.asarray(wind, dtype=np.float32)
    if mode == 'identity':
        return wind.copy(), np.arange(len(wind))
    if mode == 'constant':
        return np.full_like(wind, median), np.full(len(wind), -1, dtype=np.int64)
    if mode == 'permuted':
        order = np.random.default_rng(seed).permutation(len(wind))
        return wind[order].copy(), order
    raise ValueError(mode)
