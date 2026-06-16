from __future__ import annotations

import numpy as np

from causaldt_ad.catchment import parcel
from causaldt_ad.drawings.schema import CatchmentConfig


def _assign_split(
    group: parcel.IntArray, val_fraction: float, test_fraction: float, rng: np.random.Generator
) -> parcel.IntArray:
    split = np.zeros(group.shape[0], dtype=np.int64)
    for label in np.unique(group):
        idx = np.where(group == label)[0]
        rng.shuffle(idx)
        count = idx.shape[0]
        n_test = int(round(count * test_fraction))
        n_val = int(round(count * val_fraction))
        split[idx[:n_test]] = 2
        split[idx[n_test : n_test + n_val]] = 1
        split[idx[n_test + n_val :]] = 0
    return split


def _standardize(values: parcel.FloatArray, train_idx: parcel.IntArray) -> parcel.FloatArray:
    pool = values[train_idx].reshape(-1, values.shape[-1])
    mean = pool.mean(axis=0)
    std = pool.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return (values - mean) / std


def prepare(cohort: parcel.Cohort, config: CatchmentConfig) -> parcel.Cohort:
    rng = np.random.default_rng(config.seed + 1)
    split = _assign_split(cohort.group, config.val_fraction, config.test_fraction, rng)
    train_idx = np.where(split == 0)[0].astype(np.int64)

    features = _standardize(cohort.features, train_idx)
    nodes = _standardize(cohort.nodes, train_idx)
    mask = np.ones_like(features)

    for name in config.masked_modalities:
        if name in cohort.modalities:
            lo, hi = cohort.modalities[name]
            mask[:, :, lo:hi] = 0.0

    if config.missing_rate > 0.0:
        drop = rng.random(features.shape) < config.missing_rate
        mask[drop] = 0.0

    features = features * mask
    return parcel.Cohort(
        nodes=nodes,
        features=features,
        cognition=cohort.cognition,
        mask=mask,
        group=cohort.group,
        split=split,
        edges=cohort.edges,
        modalities=cohort.modalities,
    )


def build(config: CatchmentConfig) -> parcel.Cohort:
    from causaldt_ad.catchment.inflow import generate

    return prepare(generate(config), config)
