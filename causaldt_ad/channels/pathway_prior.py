from __future__ import annotations

import torch

from causaldt_ad.catchment.pathway import forbidden_pairs, supported_pairs


def prior_masks(node_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    supported = torch.zeros((node_dim, node_dim))
    forbidden = torch.zeros((node_dim, node_dim))
    for i, j in supported_pairs(node_dim):
        supported[i, j] = 1.0
    for i, j in forbidden_pairs(node_dim):
        forbidden[i, j] = 1.0
    return supported, forbidden


def pathway_penalty(
    weight: torch.Tensor,
    supported: torch.Tensor,
    forbidden: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    magnitude = weight.abs()
    support_term = (supported * torch.clamp(margin - magnitude, min=0.0)).sum()
    forbid_term = (forbidden * magnitude).sum()
    return support_term + forbid_term
