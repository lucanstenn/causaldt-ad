from __future__ import annotations

import torch
from torch import nn


class ModalityBank(nn.Module):
    def __init__(self, modalities: dict[str, tuple[int, int]], out_dim: int) -> None:
        super().__init__()
        self.bounds = dict(modalities)
        self.projections = nn.ModuleDict(
            {
                name: nn.Sequential(nn.Linear(hi - lo, out_dim), nn.GELU())
                for name, (lo, hi) in modalities.items()
            }
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        fused = features.new_zeros((*features.shape[:-1], self.norm.normalized_shape[0]))
        for name, (lo, hi) in self.bounds.items():
            block = features[..., lo:hi]
            fused = fused + self.projections[name](block)
        return self.norm(fused)
