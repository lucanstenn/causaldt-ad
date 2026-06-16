# Repository Plan — CausalDT-AD

The package is laid out as a river-basin water network: water (multi-omic signal) is collected in the
catchment, carved into a channel network (the causal DAG), routed through it (the world model), and
regulated by control structures (the RL therapy policy); gauging stations read the discharge.

## Directory tree
```
causaldt_ad/
  confluence.py            assembled CausalDT-AD (the point where the three modules meet)
  catchment/               data intake
    inflow.py              synthetic multi-omic cohort generator
    gauges.py              per-modality feature blocks
    confluence_ops.py      standardize, impute, split, modality masking
    parcel.py              typed subject/occasion records
  channels/                Causal Discovery Engine
    sem.py                 nonlinear SEM columns (Eq. 1)
    acyclicity.py          NOTEARS trace-exponential constraint (Eq. 2)
    pathway_prior.py       Aβ–LD supported/forbidden edge prior (Eq. 4)
    discover.py            augmented-Lagrangian fit + DBN temporal extension (Eq. 3, Alg. 1)
  routing/                 Causal World Model
    intake.py              encoder q_φ
    transit.py             graph-structured transition p_ψ (Eq. 5)
    outfall.py             decoder p_ξ
    reservoir.py           ELBO + β-annealing + imagination rollout (Eq. 6)
    sluice.py              counterfactual do-operator
  regulation/              RL Therapy Optimizer
    weir.py                SAC twin critics + Gaussian actor
    abstraction.py         reward (Eq. 7)
    regime.py              SAC training in imagination
  gauging/                 metrics + statistics
    discharge.py           RMSE/MAE/AUC/CFS
    rating.py              SHD-acc/F1/TPR/FDR
    statistics.py          bootstrap CI, paired t-test, seed aggregation
  works/                   orchestration
    programme.py           three-phase trainer
    pipeline.py            four-stage decision pipeline
    levels.py              seed + atomic checkpoint IO
  drawings/                configuration
    schema.py              msgspec.Struct frozen schema
    loader.py              layered TOML (extends) + key=value overrides
  headworks/
    __main__.py            click subcommand group (carve/route/regulate/gauge/divert/trace)
regimes/                   *.toml experiment configs
field_checks/              pytest suite
fieldwork/                 shell launch helpers
docs/                      project-context, implementation-map, deviations, repo-plan
assets/
```

## Configuration / CLI stack
- Schema: `msgspec.Struct(frozen=True)` typed config models in `drawings/schema.py`.
- Files: TOML with an `extends = "..."` chain resolved by `drawings/loader.py`; CLI `key=value` overrides
  applied last.
- CLI: `click` subcommand group exposed at `python -m causaldt_ad.headworks`.

## Pinned dependencies
```
torch>=2.1,<2.12
numpy>=1.26
scipy>=1.11
scikit-learn>=1.3
pandas>=2.1
msgspec>=0.18
click>=8.1
```
Dev: ruff, black, isort, mypy, pytest, pre-commit.

## Test coverage targets
`catchment`, `channels`, `routing`, `regulation`, `gauging` each have unit tests; one end-to-end smoke
trains two steps on `_smoke.toml`; a style guard scans the package, tests, README and Makefile for
comments/docstrings/forbidden phrases/emoji.
