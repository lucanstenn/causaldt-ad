from __future__ import annotations

import torch
from torch import nn

LOG_STD_BOUNDS: tuple[float, float] = (-5.0, 2.0)


def _mlp(in_dim: int, hidden: tuple[int, int], out_dim: int) -> nn.Sequential:
    first, second = hidden
    return nn.Sequential(
        nn.Linear(in_dim, first),
        nn.GELU(),
        nn.Linear(first, second),
        nn.GELU(),
        nn.Linear(second, out_dim),
    )


class GaussianPolicy(nn.Module):
    def __init__(self, latent_dim: int, action_dim: int, hidden: tuple[int, int]) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.trunk = _mlp(latent_dim, hidden, 2 * action_dim)

    def forward(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.trunk(latent).chunk(2, dim=-1)
        return mean, log_std.clamp(*LOG_STD_BOUNDS)

    def sample(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(latent)
        normal = torch.distributions.Normal(mean, log_std.exp())
        raw = normal.rsample()
        squashed = torch.tanh(raw)
        action = 0.5 * (squashed + 1.0)
        jacobian = torch.log(0.5 * (1.0 - squashed**2) + 1e-6)
        log_prob = (normal.log_prob(raw) - jacobian).sum(dim=-1)
        return action, log_prob

    def act(self, latent: torch.Tensor) -> torch.Tensor:
        mean, _ = self.forward(latent)
        return 0.5 * (torch.tanh(mean) + 1.0)


class TwinCritic(nn.Module):
    def __init__(self, latent_dim: int, action_dim: int, hidden: tuple[int, int]) -> None:
        super().__init__()
        self.first = _mlp(latent_dim + action_dim, hidden, 1)
        self.second = _mlp(latent_dim + action_dim, hidden, 1)

    def forward(
        self, latent: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        joined = torch.cat([latent, action], dim=-1)
        return self.first(joined).squeeze(-1), self.second(joined).squeeze(-1)
