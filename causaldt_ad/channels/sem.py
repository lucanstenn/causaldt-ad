from __future__ import annotations

import torch
from torch import nn


class StructuralColumns(nn.Module):
    def __init__(self, node_dim: int, hidden: int) -> None:
        super().__init__()
        self.node_dim = node_dim
        self.columns = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(node_dim, hidden), nn.GELU(), nn.Linear(hidden, 1))
                for _ in range(node_dim)
            ]
        )

    def forward(self, observed: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        gated = observed.unsqueeze(1) * weight.t().unsqueeze(0)
        outputs = [self.columns[j](gated[:, j, :]) for j in range(self.node_dim)]
        return torch.cat(outputs, dim=1)
