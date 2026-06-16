from __future__ import annotations

import torch
from torch import nn

from causaldt_ad.catchment.gauges import ModalityBank


class Encoder(nn.Module):
    def __init__(
        self, modalities: dict[str, tuple[int, int]], latent_dim: int, hidden: int
    ) -> None:
        super().__init__()
        self.bank = ModalityBank(modalities, hidden)
        self.recurrent = nn.GRU(hidden, hidden, batch_first=True)
        self.to_mean = nn.Linear(hidden, latent_dim)
        self.to_logvar = nn.Linear(hidden, latent_dim)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self.bank(features)
        rolled, _ = self.recurrent(embedded)
        return self.to_mean(rolled), self.to_logvar(rolled).clamp(-8.0, 8.0)
