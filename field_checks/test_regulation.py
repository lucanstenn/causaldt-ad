from __future__ import annotations

import pytest
import torch

from causaldt_ad.confluence import reward_targets
from causaldt_ad.drawings.schema import RegulationConfig, RoutingConfig
from causaldt_ad.regulation.abstraction import therapy_reward
from causaldt_ad.regulation.regime import TherapyOptimizer
from causaldt_ad.routing.reservoir import CausalWorldModel
from causaldt_ad.works.levels import set_seed


def test_reward_matches_formula() -> None:
    delta = torch.tensor([1.0])
    biomarkers = torch.tensor([[0.0, 0.0]])
    reference = torch.tensor([0.0, 0.0])
    action = torch.zeros(1, 3)
    reward = therapy_reward(delta, biomarkers, reference, action, (0.5, 0.3, 0.2))
    assert float(reward[0]) == pytest.approx(1.1, abs=1e-6)


def test_cost_term_penalizes_action() -> None:
    delta = torch.tensor([0.0])
    biomarkers = torch.tensor([[0.0]])
    reference = torch.tensor([0.0])
    quiet = therapy_reward(delta, biomarkers, reference, torch.zeros(1, 3), (0.5, 0.3, 0.2))
    loud = therapy_reward(delta, biomarkers, reference, torch.ones(1, 3), (0.5, 0.3, 0.2))
    assert float(loud[0]) < float(quiet[0])


def _optimizer() -> TherapyOptimizer:
    routing = RoutingConfig(latent_dim=16, encoder_hidden=8, transition_hidden=8, horizon=3)
    parents = tuple(tuple(i for i in range(j)) for j in range(8))
    model = CausalWorldModel({"omic": (0, 6)}, 6, 8, parents, 3, routing)
    index, reference = reward_targets(8)
    config = RegulationConfig(buffer_size=256, steps=32, batch_size=8)
    return TherapyOptimizer(model, index, reference, config, routing, seed=0)


def test_sac_update_produces_policy_gradients() -> None:
    set_seed(0)
    optimizer = _optimizer()
    optimizer._rollout(16, deterministic=False)
    optimizer._update()
    assert any(parameter.grad is not None for parameter in optimizer.policy.parameters())


def test_policy_report_is_finite() -> None:
    set_seed(0)
    optimizer = _optimizer()
    report = optimizer.train()
    assert report.cumulative_reward == report.cumulative_reward
    assert report.biomarker_norm == report.biomarker_norm
