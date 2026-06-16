from __future__ import annotations

import torch

from causaldt_ad.drawings.schema import RoutingConfig
from causaldt_ad.routing.reservoir import CausalWorldModel, beta_at
from causaldt_ad.routing.sluice import counterfactual, descendants
from causaldt_ad.works.levels import set_seed


def _model(parents: tuple[tuple[int, ...], ...]) -> CausalWorldModel:
    config = RoutingConfig(latent_dim=8, encoder_hidden=8, transition_hidden=8, horizon=3)
    return CausalWorldModel({"omic": (0, 4)}, 4, 4, parents, 1, config)


def test_beta_schedule_anneals() -> None:
    config = RoutingConfig(beta_start=0.01, beta_end=1.0, beta_warmup_epochs=50)
    assert beta_at(0, config) == 0.01
    assert beta_at(25, config) > 0.01
    assert beta_at(50, config) == 1.0
    assert beta_at(80, config) == 1.0


def test_elbo_shapes_and_gradients() -> None:
    set_seed(0)
    model = _model(((), (0,), (1,), ()))
    features = torch.randn(2, 3, 4)
    actions = torch.zeros(2, 3, 1)
    cognition = torch.randn(2, 3)
    report = model.elbo(features, actions, 1.0, cognition)
    assert report["loss"].ndim == 0
    report["loss"].backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_overfit_single_batch() -> None:
    set_seed(0)
    model = _model(((), (0,), (1,), (2,)))
    features = torch.randn(4, 3, 4)
    actions = torch.zeros(4, 3, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    initial = float(model.elbo(features, actions, 0.0)["recon"].detach())
    for _ in range(200):
        optimizer.zero_grad()
        loss = model.elbo(features, actions, 0.0)["recon"]
        loss.backward()
        optimizer.step()
    final = float(model.elbo(features, actions, 0.0)["recon"].detach())
    assert final < 0.4 * initial


def test_descendants_reachability() -> None:
    assert descendants(((), (0,), (1,), ()), 0) == frozenset({1, 2})
    assert descendants(((), (0,), (1,), ()), 3) == frozenset()


def test_do_operator_fixes_non_descendants() -> None:
    set_seed(1)
    model = _model(((), (0,), (1,), ()))
    latent = torch.randn(2, 8)
    actions = torch.zeros(2, 3, 1)
    factual, counter = counterfactual(model, latent, actions, node=0, value=3.0, horizon=3)
    assert torch.allclose(factual[..., 6:8], counter[..., 6:8])
    assert not torch.allclose(factual[:, 1:, 2:4], counter[:, 1:, 2:4])
