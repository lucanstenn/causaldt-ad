from __future__ import annotations

import torch


def acyclicity(weight: torch.Tensor) -> torch.Tensor:
    squared = weight * weight
    expm = torch.matrix_exp(squared)
    return torch.trace(expm) - weight.shape[0]


def zero_diagonal(weight: torch.Tensor) -> torch.Tensor:
    return weight - torch.diag(torch.diagonal(weight))
