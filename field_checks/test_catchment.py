from __future__ import annotations

import numpy as np

from causaldt_ad.catchment.confluence_ops import build
from causaldt_ad.catchment.inflow import generate
from causaldt_ad.drawings.schema import CatchmentConfig


def test_cohort_shapes() -> None:
    config = CatchmentConfig(subjects=40, timepoints=5, node_dim=16)
    cohort = generate(config)
    assert cohort.nodes.shape == (40, 5, 16)
    assert cohort.features.shape == (40, 5, config.feature_dim)
    assert cohort.cognition.shape == (40, 5)
    assert cohort.edges.shape == (16, 16)


def test_ground_truth_edges_acyclic() -> None:
    config = CatchmentConfig(subjects=20, node_dim=16, ground_truth_edges=47)
    cohort = generate(config)
    truth = cohort.edges != 0.0
    assert np.all(np.diag(truth) == 0)
    assert int(truth.sum()) == 47
    assert np.allclose(np.tril(cohort.edges), 0.0)


def test_split_and_masking() -> None:
    config = CatchmentConfig(subjects=200, masked_modalities=("imaging",), missing_rate=0.0)
    cohort = build(config)
    fractions = [np.mean(cohort.split == fold) for fold in (0, 1, 2)]
    assert abs(fractions[2] - 0.25) < 0.05
    assert abs(fractions[1] - 0.15) < 0.05
    lo, hi = cohort.modalities["imaging"]
    assert np.allclose(cohort.features[:, :, lo:hi], 0.0)


def test_cohort_is_deterministic() -> None:
    config = CatchmentConfig(subjects=30)
    first = build(config)
    second = build(config)
    assert np.array_equal(first.nodes, second.nodes)
    assert np.array_equal(first.edges, second.edges)
