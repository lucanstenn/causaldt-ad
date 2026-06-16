# CausalDT-AD - Basin Hydrological Yearbook

```
Yearbook of record   : CausalDT-AD multi-omic basin
Basin                 : amyloid-lysosomal regulatory catchment
Datum                 : IEEE J. Biomedical and Health Informatics
Gauging stations      : 8 (catchment, channels, routing, regulation,
                        gauging, works, drawings, headworks)
Issue                 : v0.1.0
```

This yearbook documents the gauging network that turns a patient's multi-omic profile into a
patient-centered causal digital twin. Water (the multi-omic signal) is collected in the catchment,
carved into a channel network (the causal graph), routed through that network by a latent world model,
and managed by control structures (the reinforcement-learning therapy policy). Each section below is a
station record: where it sits, how to read it, and what it gauges.

## Foreword

CausalDT-AD couples three sequentially trained stages. A Causal Discovery Engine learns a sparse,
acyclic graph over the biological variables under an amyloid-lysosomal pathway prior. A Causal World
Model builds a graph-structured latent dynamics engine for 12-month trajectory forecasting and
do-operator counterfactual simulation. A Soft Actor-Critic Therapy Optimizer derives a three-channel
intervention policy (anti-amyloid, lysosomal acidification support, autophagic flux enhancement) inside
the world model's imagined latent space.

Public ADNI and OASIS-3 access is application-gated, so this release ships a deterministic synthetic
multi-omic cohort grounded in the A/T/N staging and the amyloid-lysosomal feedback loop. The synthetic
basin reproduces the pipeline and the qualitative ordering of the ablations; the manuscript's headline
figures are reproducible only on the credentialed cohorts.

## Station register

| Station | Role in the basin | Source |
|---|---|---|
| catchment | multi-omic intake, standardization, imputation, splits, modality masking | `causaldt_ad/catchment` |
| channels | causal discovery engine (NOTEARS + pathway prior, DBN extension) | `causaldt_ad/channels` |
| routing | causal world model (encoder, graph transition, decoder, do-operator) | `causaldt_ad/routing` |
| regulation | SAC therapy optimizer (twin critics, Gaussian policy, reward) | `causaldt_ad/regulation` |
| gauging | trajectory metrics, structural metrics, bootstrap and paired tests | `causaldt_ad/gauging` |
| works | three-phase trainer, decision pipeline, seeding, checkpoints | `causaldt_ad/works` |
| drawings | frozen msgspec config schema and layered TOML loader | `causaldt_ad/drawings` |
| headworks | command-line control structure | `causaldt_ad/headworks` |

## Establishing the gauging network

pip:

```
pip install -e .
```

conda:

```
conda env create -f environment.yml
conda activate causaldt_ad
pip install -e .
```

Docker:

```
docker build -t causaldt_ad .
docker run --rm causaldt_ad --help
```

## Observation programme

Configurations live in `regimes/`. A run selects one with `--config` and may override any field with
`--set section.key=value`.

| Verb | Reading taken |
|---|---|
| `carve` | discover the channel network and report structural accuracy |
| `route` | train the causal world model on the catchment |
| `regulate` | train the therapy policy and report its cumulative reward |
| `gauge` | evaluate held-out 12-month cognition and policy reward |
| `divert` | run the four-stage decision pipeline (instantiate, score, prognose, advise) |
| `trace` | write a run summary to the output directory |

```
python -m causaldt_ad.headworks --config main carve
python -m causaldt_ad.headworks --config main route
python -m causaldt_ad.headworks --config main regulate
python -m causaldt_ad.headworks --config ablation_no_prior carve
python -m causaldt_ad.headworks --config _smoke --set routing.horizon=4 gauge
```

## Records of flow

Manuscript readings on the credentialed cohorts (ADNI unless noted), for orientation. Synthetic runs
do not reproduce these values.

| Reading | Station and verb | Manuscript value |
|---|---|---|
| 12-month trajectory RMSE | `gauge` | 2.40 |
| MCI-to-AD conversion AUC | `gauge` | 0.943 |
| Counterfactual fidelity score | `gauge` | 0.876 |
| Structural Hamming accuracy | `carve` | 0.914 |
| Causal-discovery F1 | `carve` | 0.891 |
| Policy cumulative reward (12 months) | `regulate` | 7.81 |
| Policy delta-MMSE | `regulate` | 5.7 |
| Cross-cohort transfer RMSE (OASIS-3) | `gauge` | 2.73 |

## Rating and quality control

```
make lint
make type
make test
```

The check suite spans shapes, the acyclicity invariant, the do-operator non-descendant invariance,
single-batch overfitting, metric agreement against scikit-learn, gradient flow, determinism, config
layering, and a style guard, alongside the end-to-end smoke run on `regimes/_smoke.toml`.

## Catchment description

| Cohort | Subset | Access | URL |
|---|---|---|---|
| ADNI | 1,247 multi-omic profiles (412 CN, 518 EMCI, 204 LMCI, 113 AD) | application-gated data-use agreement | https://adni.loni.usc.edu |
| OASIS-3 | 1,378 subjects, longitudinal imaging and cognition | application-gated data-use terms | https://www.oasis-brains.org |

Both are public but require an approved application; no anonymous download exists. The shipped synthetic
cohort attaches through the same record schema once a user is credentialed.

## Plant and resources

| Stage | Params (M) | Train (h) | Infer (ms) | Memory (GB) |
|---|---|---|---|---|
| channels (CDE) | 3.1 | 4.8 | - | 4.9 |
| routing (CWM) | 8.7 | 6.2 | 15.6 | 7.4 |
| regulation (RTO) | 2.4 | 3.7 | - | 3.1 |
| total | 14.2 | 14.7 | 24.8 | 7.4 |

Reference hardware: a single NVIDIA A100.

## Custody and abstraction licences

Experiments use only previously published public datasets (ADNI, OASIS-3). No original data were
generated, no human subjects were enrolled, and no biological specimens were used. The pipeline is
read-only with respect to the patient: it returns subject-specific forecasts and a ranked, interpretable
action vector with per-mechanism causal attribution, and never acts autonomously on the patient.
