from __future__ import annotations

import json
from pathlib import Path

import click
import numpy as np
import torch

from causaldt_ad.catchment.confluence_ops import build as build_cohort
from causaldt_ad.channels.discover import discover
from causaldt_ad.drawings.loader import load
from causaldt_ad.drawings.schema import Config
from causaldt_ad.gauging.discharge import rmse
from causaldt_ad.gauging.rating import edge_scores, shd_accuracy
from causaldt_ad.works.pipeline import assess
from causaldt_ad.works.programme import run


@click.group()
@click.option("--config", "-c", "config_name", default="main")
@click.option("--set", "overrides", multiple=True)
@click.pass_context
def main(ctx: click.Context, config_name: str, overrides: tuple[str, ...]) -> None:
    ctx.obj = load(config_name, overrides)


@main.command()
@click.pass_obj
def carve(config: Config) -> None:
    cohort = build_cohort(config.catchment)
    result = discover(cohort, config.channels)
    truth = cohort.edges != 0.0
    scores = edge_scores(result.graph, truth)
    click.echo(f"discovered_edges={int(result.graph.sum())} h={result.h_value:.3e}")
    click.echo(f"shd_accuracy={shd_accuracy(result.graph, truth):.3f}")
    click.echo(f"f1={scores['f1']:.3f} tpr={scores['tpr']:.3f} fdr={scores['fdr']:.3f}")


@main.command()
@click.pass_obj
def route(config: Config) -> None:
    outcome = run(config)
    click.echo(f"world_model_loss={outcome.world_loss:.4f} discovery_h={outcome.discovery_h:.3e}")


@main.command()
@click.pass_obj
def regulate(config: Config) -> None:
    outcome = run(config)
    report = outcome.policy
    click.echo(
        f"cumulative_reward={report.cumulative_reward:.3f} "
        f"delta_cognition={report.delta_cognition:.3f} "
        f"biomarker_norm={report.biomarker_norm:.3f}"
    )


@main.command()
@click.pass_obj
def gauge(config: Config) -> None:
    outcome = run(config)
    twin = outcome.twin
    test = twin.cohort.fold(2)
    features = torch.as_tensor(twin.cohort.features[test], dtype=torch.float32)
    with torch.no_grad():
        predicted = twin.world_model.decoder(twin.latent_at(features))[1].cpu().numpy()
    actual = twin.cohort.cognition[test, -1].astype(np.float64)
    click.echo(f"cognition_rmse={rmse(predicted.astype(np.float64), actual):.3f}")
    click.echo(f"policy_cumulative_reward={outcome.policy.cumulative_reward:.3f}")


@main.command()
@click.pass_obj
def divert(config: Config) -> None:
    outcome = run(config)
    twin = outcome.twin
    test = twin.cohort.fold(2)[:32]
    features = torch.as_tensor(twin.cohort.features[test], dtype=torch.float32)
    report = assess(twin, features, config.routing.horizon, samples=4)
    click.echo(f"early_detection={report.early_detection:.3f}")
    click.echo(
        f"recommended_action={tuple(round(value, 3) for value in report.recommended_action)}"
    )
    for name, (mean, spread) in report.prognostic_band.items():
        click.echo(f"band[{name}]={mean:.3f}+-{spread:.3f}")


@main.command()
@click.pass_obj
def trace(config: Config) -> None:
    outcome = run(config)
    destination = Path(config.works.out_dir) / "trace.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": config.name,
        "discovery_h": outcome.discovery_h,
        "world_model_loss": outcome.world_loss,
        "cumulative_reward": outcome.policy.cumulative_reward,
        "delta_cognition": outcome.policy.delta_cognition,
    }
    destination.write_text(json.dumps(payload, indent=2))
    click.echo(f"wrote {destination}")


if __name__ == "__main__":
    main()
