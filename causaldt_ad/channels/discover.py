from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from causaldt_ad.catchment.parcel import Cohort
from causaldt_ad.channels.acyclicity import acyclicity, zero_diagonal
from causaldt_ad.channels.pathway_prior import pathway_penalty, prior_masks
from causaldt_ad.channels.sem import StructuralColumns
from causaldt_ad.drawings.schema import ChannelConfig

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


class CausalDiscoveryEngine(nn.Module):
    def __init__(self, node_dim: int, hidden: int) -> None:
        super().__init__()
        self.node_dim = node_dim
        self.weight = nn.Parameter(0.1 * torch.randn(node_dim, node_dim))
        self.columns = StructuralColumns(node_dim, hidden)

    def adjacency(self) -> torch.Tensor:
        return zero_diagonal(self.weight)

    def forward(self, observed: torch.Tensor) -> torch.Tensor:
        return self.columns(observed, self.adjacency())


@dataclass(frozen=True, slots=True)
class DiscoveryResult:
    weight: FloatArray
    lagged: FloatArray
    graph: BoolArray
    parents: tuple[tuple[int, ...], ...]
    h_value: float


def _samples(nodes: FloatArray, temporal: bool) -> torch.Tensor:
    if temporal:
        return torch.as_tensor(nodes.reshape(-1, nodes.shape[2]), dtype=torch.float32)
    return torch.as_tensor(nodes[:, 0, :], dtype=torch.float32)


def _lagged_structure(nodes: FloatArray) -> FloatArray:
    if nodes.shape[1] < 2:
        return np.zeros((nodes.shape[2], nodes.shape[2]), dtype=np.float64)
    current = nodes[:, 1:, :].reshape(-1, nodes.shape[2])
    previous = nodes[:, :-1, :].reshape(-1, nodes.shape[2])
    gram = previous.T @ previous + 1e-3 * np.eye(nodes.shape[2])
    return np.linalg.solve(gram, previous.T @ current).astype(np.float64)


def discover(cohort: Cohort, config: ChannelConfig) -> DiscoveryResult:
    observed = _samples(cohort.nodes, config.temporal)
    engine = CausalDiscoveryEngine(cohort.node_dim, config.sem_hidden)
    optimizer = torch.optim.Adam(engine.parameters(), lr=config.learning_rate)
    supported, forbidden = prior_masks(cohort.node_dim)
    criterion = nn.MSELoss()

    inner = max(1, config.epochs // config.outer_steps)
    rho = config.dual_step
    alpha = 0.0
    h_prev = float("inf")
    for _ in range(config.outer_steps):
        for _ in range(inner):
            optimizer.zero_grad()
            prediction = engine(observed)
            recon = criterion(prediction, observed)
            weight = engine.adjacency()
            sparsity = weight.abs().sum()
            prior = (
                pathway_penalty(weight, supported, forbidden, config.prior_margin)
                if config.use_prior
                else weight.new_zeros(())
            )
            constraint = acyclicity(weight)
            loss = (
                recon
                + config.lambda_l1 * sparsity
                + config.lambda_prior * prior
                + 0.5 * rho * constraint * constraint
                + alpha * constraint
            )
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            h_now = float(acyclicity(engine.adjacency()))
        alpha += rho * h_now
        if h_now > 0.25 * h_prev:
            rho *= config.penalty_growth
        h_prev = h_now
        if h_now < config.acyclicity_tol:
            break

    weight_np = engine.adjacency().detach().cpu().numpy().astype(np.float64)
    lagged_np = _lagged_structure(cohort.nodes) if config.temporal else np.zeros_like(weight_np)
    graph = np.abs(weight_np) > config.edge_threshold
    parents = tuple(
        tuple(int(i) for i in np.nonzero(graph[:, j])[0]) for j in range(cohort.node_dim)
    )
    return DiscoveryResult(
        weight=weight_np, lagged=lagged_np, graph=graph, parents=parents, h_value=h_prev
    )
