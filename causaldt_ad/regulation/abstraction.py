from __future__ import annotations

import torch


def therapy_reward(
    delta_cognition: torch.Tensor,
    biomarkers: torch.Tensor,
    reference: torch.Tensor,
    action: torch.Tensor,
    weights: tuple[float, float, float],
) -> torch.Tensor:
    cognitive = weights[0] * delta_cognition
    normalization = weights[1] * (1.0 - (biomarkers - reference).abs()).sum(dim=-1)
    cost = weights[2] * action.norm(dim=-1)
    return cognitive + normalization - cost
