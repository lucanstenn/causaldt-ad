from __future__ import annotations

import torch
from torch import nn

from causaldt_ad.drawings.schema import RoutingConfig
from causaldt_ad.routing.intake import Encoder
from causaldt_ad.routing.outfall import Decoder
from causaldt_ad.routing.transit import GraphTransition


def beta_at(epoch: int, config: RoutingConfig) -> float:
    if config.beta_warmup_epochs <= 0 or epoch >= config.beta_warmup_epochs:
        return config.beta_end
    span = config.beta_end - config.beta_start
    return config.beta_start + span * (epoch / config.beta_warmup_epochs)


def _gaussian_kl(
    mu_q: torch.Tensor, logvar_q: torch.Tensor, mu_p: torch.Tensor, logvar_p: torch.Tensor
) -> torch.Tensor:
    term = logvar_p - logvar_q
    term = term + (logvar_q.exp() + (mu_q - mu_p) ** 2) / logvar_p.exp()
    return 0.5 * (term - 1.0).sum(dim=-1)


class CausalWorldModel(nn.Module):
    def __init__(
        self,
        modalities: dict[str, tuple[int, int]],
        feature_dim: int,
        node_dim: int,
        parents: tuple[tuple[int, ...], ...],
        action_dim: int,
        config: RoutingConfig,
    ) -> None:
        super().__init__()
        per_node = max(1, config.latent_dim // node_dim)
        self.node_dim = node_dim
        self.per_node = per_node
        self.latent_dim = per_node * node_dim
        self.action_dim = action_dim
        self.encoder = Encoder(modalities, self.latent_dim, config.encoder_hidden)
        self.transition = GraphTransition(
            node_dim, per_node, parents, action_dim, config.transition_hidden
        )
        self.decoder = Decoder(self.latent_dim, feature_dim, config.encoder_hidden)

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mean + torch.randn_like(mean) * (0.5 * logvar).exp()

    def node_summary(self, latent: torch.Tensor) -> torch.Tensor:
        reshaped = latent.reshape(*latent.shape[:-1], self.node_dim, self.per_node)
        return reshaped.mean(dim=-1)

    def rollout_step(
        self, latent: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, logvar = self.transition(latent, action)
        nxt = self.reparameterize(mean, logvar)
        _, cognition = self.decoder(nxt)
        return nxt, cognition, self.node_summary(nxt)

    def elbo(
        self,
        features: torch.Tensor,
        actions: torch.Tensor,
        beta: float,
        cognition: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        mean_q, logvar_q = self.encoder(features)
        latent = self.reparameterize(mean_q, logvar_q)
        reconstruction, cognition_hat = self.decoder(latent)
        recon = ((reconstruction - features) ** 2).mean()
        if cognition is not None:
            recon = recon + ((cognition_hat - cognition) ** 2).mean()

        prior_mean = torch.zeros_like(mean_q[:, 0])
        prior_logvar = torch.zeros_like(logvar_q[:, 0])
        kl_terms: list[torch.Tensor] = [
            _gaussian_kl(mean_q[:, 0], logvar_q[:, 0], prior_mean, prior_logvar).mean()
        ]
        for step in range(1, features.shape[1]):
            mean_p, logvar_p = self.transition(latent[:, step - 1], actions[:, step - 1])
            kl_terms.append(
                _gaussian_kl(mean_q[:, step], logvar_q[:, step], mean_p, logvar_p).mean()
            )
        kl = torch.stack(kl_terms).mean()
        loss = recon + beta * kl
        return {"loss": loss, "recon": recon, "kl": kl}
