from __future__ import annotations

from dataclasses import dataclass

import torch

from causaldt_ad.catchment import parcel
from causaldt_ad.confluence import CausalDigitalTwin

ATTRIBUTION_NODES: tuple[str, ...] = (
    "amyloid_beta",
    "lyso_ph",
    "autophagic_flux",
    "ptau",
    "hippocampus",
)


@dataclass(slots=True)
class TwinAssessment:
    early_detection: float
    prognostic_band: dict[str, tuple[float, float]]
    recommended_action: tuple[float, ...]
    causal_attribution: dict[str, float]


@torch.no_grad()
def _rollout_cognition(
    twin: CausalDigitalTwin, latent: torch.Tensor, action: torch.Tensor, horizon: int
) -> torch.Tensor:
    state = latent
    for _ in range(horizon):
        state, _, _ = twin.world_model.rollout_step(state, action)
    return twin.world_model.decoder(state)[1]


@torch.no_grad()
def assess(
    twin: CausalDigitalTwin, features: torch.Tensor, horizon: int, samples: int
) -> TwinAssessment:
    latent = twin.latent_at(features)
    batch = latent.shape[0]
    action_dim = twin.optimizer.config.action_dim
    baseline_cog = twin.world_model.decoder(latent)[1]

    untreated = _rollout_cognition(twin, latent, torch.zeros(batch, action_dim), horizon)
    early = float((baseline_cog - untreated).mean())

    band: dict[str, tuple[float, float]] = {}
    for channel, name in enumerate(parcel.ACTION_TARGETS):
        gains = torch.empty(samples)
        for draw in range(samples):
            action = torch.zeros(batch, action_dim)
            action[:, channel] = 1.0
            treated = _rollout_cognition(twin, latent, action, horizon)
            gains[draw] = (treated - untreated).mean()
        band[name] = (float(gains.mean()), float(gains.std()))

    recommended = tuple(float(value) for value in twin.optimizer.policy.act(latent).mean(dim=0))

    summary = twin.world_model.node_summary(latent).mean(dim=0)
    attribution = {
        name: float(summary[parcel.node_index(twin.cohort.node_dim, name)])
        for name in ATTRIBUTION_NODES
        if name in parcel.node_names(twin.cohort.node_dim)
    }
    return TwinAssessment(
        early_detection=early,
        prognostic_band=band,
        recommended_action=recommended,
        causal_attribution=attribution,
    )
