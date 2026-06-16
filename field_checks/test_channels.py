from __future__ import annotations

import pytest
import torch

from causaldt_ad.catchment.confluence_ops import build
from causaldt_ad.catchment.pathway import forbidden_pairs, supported_pairs
from causaldt_ad.channels.acyclicity import acyclicity
from causaldt_ad.channels.discover import discover
from causaldt_ad.channels.pathway_prior import pathway_penalty, prior_masks
from causaldt_ad.drawings.schema import CatchmentConfig, ChannelConfig
from causaldt_ad.works.levels import set_seed


def test_dag_has_zero_acyclicity() -> None:
    weight = torch.triu(torch.ones(6, 6), diagonal=1)
    assert float(acyclicity(weight)) == pytest.approx(0.0, abs=1e-5)


def test_cycle_has_positive_acyclicity() -> None:
    weight = torch.zeros(3, 3)
    weight[0, 1] = 1.0
    weight[1, 2] = 1.0
    weight[2, 0] = 1.0
    assert float(acyclicity(weight)) > 1e-3


def test_pathway_penalty_rewards_support_and_punishes_forbidden() -> None:
    supported, forbidden = prior_masks(16)
    strong = supported * 0.5 + forbidden * 0.0
    weak = supported * 0.0 + forbidden * 0.9
    penalty_strong = float(pathway_penalty(strong, supported, forbidden, 0.1))
    penalty_weak = float(pathway_penalty(weak, supported, forbidden, 0.1))
    assert penalty_strong < penalty_weak


def test_prior_guided_discovery() -> None:
    set_seed(0)
    cohort = build(CatchmentConfig(subjects=160, timepoints=5, node_dim=16))
    result = discover(cohort, ChannelConfig(epochs=200, outer_steps=10))
    supported = supported_pairs(16)
    forbidden = forbidden_pairs(16)
    recall = sum(bool(result.graph[i, j]) for i, j in supported) / len(supported)
    included_forbidden = sum(bool(result.graph[i, j]) for i, j in forbidden)
    assert recall >= 0.8
    assert included_forbidden == 0
    assert result.h_value < 1.0


def test_discovery_extends_beyond_prior() -> None:
    set_seed(0)
    cohort = build(CatchmentConfig(subjects=160, timepoints=5, node_dim=16))
    result = discover(cohort, ChannelConfig(epochs=200, outer_steps=10))
    truth = cohort.edges != 0.0
    assert int((result.graph & truth).sum()) > len(supported_pairs(16))
