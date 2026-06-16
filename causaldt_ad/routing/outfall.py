from __future__ import annotations

import torch
from torch import nn


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, feature_dim: int, hidden: int) -> None:
        super().__init__()
        self.reconstruct = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.GELU(), nn.Linear(hidden, feature_dim)
        )
        self.cognition = nn.Sequential(
            nn.Linear(latent_dim, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1)
        )

    def forward(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        reconstruction = self.reconstruct(latent)
        cognition = self.cognition(latent).squeeze(-1)
        return reconstruction, cognition
