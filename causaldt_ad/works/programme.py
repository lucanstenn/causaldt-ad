from __future__ import annotations

from dataclasses import dataclass

import torch

from causaldt_ad.catchment.confluence_ops import build as build_cohort
from causaldt_ad.channels.discover import discover
from causaldt_ad.confluence import CausalDigitalTwin, build_optimizer, build_world_model
from causaldt_ad.drawings.schema import Config
from causaldt_ad.regulation.regime import PolicyReport
from causaldt_ad.routing.reservoir import CausalWorldModel, beta_at
from causaldt_ad.works.levels import set_seed


@dataclass(slots=True)
class TrainingOutcome:
    twin: CausalDigitalTwin
    discovery_h: float
    world_loss: float
    policy: PolicyReport


def _train_world_model(
    world_model: CausalWorldModel,
    features: torch.Tensor,
    cognition: torch.Tensor,
    config: Config,
) -> float:
    actions = torch.zeros(features.shape[0], features.shape[1], config.regulation.action_dim)
    optimizer = torch.optim.Adam(world_model.parameters(), lr=config.routing.learning_rate)
    last = 0.0
    for epoch in range(config.routing.epochs):
        beta = beta_at(epoch, config.routing)
        optimizer.zero_grad()
        report = world_model.elbo(features, actions, beta, cognition)
        report["loss"].backward()
        optimizer.step()
        last = float(report["loss"].detach())
    return last


def run(config: Config) -> TrainingOutcome:
    set_seed(config.works.seed)
    cohort = build_cohort(config.catchment)
    discovery = discover(cohort, config.channels)

    world_model = build_world_model(cohort, discovery, config)
    train_idx = cohort.fold(0)
    features = torch.as_tensor(cohort.features[train_idx], dtype=torch.float32)
    cognition = torch.as_tensor(cohort.cognition[train_idx], dtype=torch.float32)
    world_loss = _train_world_model(world_model, features, cognition, config)

    optimizer = build_optimizer(world_model, config)
    policy = optimizer.train()

    twin = CausalDigitalTwin(
        cohort=cohort, discovery=discovery, world_model=world_model, optimizer=optimizer
    )
    return TrainingOutcome(
        twin=twin, discovery_h=discovery.h_value, world_loss=world_loss, policy=policy
    )
