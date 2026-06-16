from __future__ import annotations

import math

from causaldt_ad.drawings.loader import load
from causaldt_ad.works.programme import run


def test_smoke_end_to_end_is_finite() -> None:
    outcome = run(load("_smoke"))
    assert math.isfinite(outcome.world_loss)
    assert math.isfinite(outcome.policy.cumulative_reward)
    assert outcome.discovery_h < 5.0


def test_run_is_reproducible() -> None:
    first = run(load("_smoke"))
    second = run(load("_smoke"))
    assert first.world_loss == second.world_loss
    assert first.discovery_h == second.discovery_h


def test_overrides_apply() -> None:
    config = load("_smoke", ("routing.horizon=7", "channels.use_prior=false"))
    assert config.routing.horizon == 7
    assert config.channels.use_prior is False
