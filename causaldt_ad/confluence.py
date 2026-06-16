from __future__ import annotations

from dataclasses import dataclass

import torch

from causaldt_ad.catchment import parcel
from causaldt_ad.catchment.parcel import Cohort
from causaldt_ad.channels.discover import DiscoveryResult
from causaldt_ad.drawings.schema import Config
from causaldt_ad.regulation.regime import TherapyOptimizer
from causaldt_ad.routing.reservoir import CausalWorldModel


def reward_targets(node_dim: int) -> tuple[tuple[int, ...], tuple[float, ...]]:
    index = tuple(parcel.node_index(node_dim, name) for name in parcel.REWARD_BIOMARKERS)
    reference = tuple(0.0 for _ in index)
    return index, reference


def _dense_parents(node_dim: int) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(i for i in range(node_dim) if i != j) for j in range(node_dim))


def build_world_model(
    cohort: Cohort, discovery: DiscoveryResult, config: Config
) -> CausalWorldModel:
    parents = (
        discovery.parents if config.routing.graph_structured else _dense_parents(cohort.node_dim)
    )
    return CausalWorldModel(
        modalities=cohort.modalities,
        feature_dim=cohort.feature_dim,
        node_dim=cohort.node_dim,
        parents=parents,
        action_dim=config.regulation.action_dim,
        config=config.routing,
    )


def build_optimizer(world_model: CausalWorldModel, config: Config) -> TherapyOptimizer:
    index, reference = reward_targets(world_model.node_dim)
    return TherapyOptimizer(
        world_model, index, reference, config.regulation, config.routing, config.works.seed
    )


@dataclass(slots=True)
class CausalDigitalTwin:
    cohort: Cohort
    discovery: DiscoveryResult
    world_model: CausalWorldModel
    optimizer: TherapyOptimizer

    def latent_at(self, features: torch.Tensor) -> torch.Tensor:
        mean, _ = self.world_model.encoder(features)
        return mean[:, -1]
