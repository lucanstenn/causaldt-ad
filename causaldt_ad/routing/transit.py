from __future__ import annotations

import torch
from torch import nn


class GraphTransition(nn.Module):
    def __init__(
        self,
        node_dim: int,
        per_node: int,
        parents: tuple[tuple[int, ...], ...],
        action_dim: int,
        hidden: int,
    ) -> None:
        super().__init__()
        self.node_dim = node_dim
        self.per_node = per_node
        latent = node_dim * per_node
        mask = torch.zeros(node_dim, latent)
        for j in range(node_dim):
            own = range(j * per_node, (j + 1) * per_node)
            for index in own:
                mask[j, index] = 1.0
            for parent in parents[j]:
                for index in range(parent * per_node, (parent + 1) * per_node):
                    mask[j, index] = 1.0
        self.register_buffer("parent_mask", mask)
        self.parent_mask: torch.Tensor
        self.cores = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(latent + action_dim, hidden),
                    nn.GELU(),
                    nn.Linear(hidden, 2 * per_node),
                )
                for _ in range(node_dim)
            ]
        )

    def forward(
        self, latent: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        means: list[torch.Tensor] = []
        logvars: list[torch.Tensor] = []
        for j in range(self.node_dim):
            gated = latent * self.parent_mask[j]
            packed = self.cores[j](torch.cat([gated, action], dim=-1))
            mean, logvar = packed[..., : self.per_node], packed[..., self.per_node :]
            means.append(mean)
            logvars.append(logvar.clamp(-8.0, 8.0))
        return torch.cat(means, dim=-1), torch.cat(logvars, dim=-1)
