# Deviations from the manuscript

Each entry links to the manuscript section it touches and states the reason.

## D1 — Synthetic cohort stands in for the credentialed datasets
The manuscript trains on ADNI and OASIS-3 (Sec. IV-A), both application-gated under data-use
agreements with no anonymous download. The repository ships a deterministic, biology-anchored synthetic
multi-omic cohort (`causaldt_ad/catchment/inflow.py`) so the full pipeline runs end-to-end without
credentials. Real cohorts attach through the same record schema once a user is approved. Numerical
results in the manuscript are reproducible only on the real cohorts; synthetic runs reproduce the
pipeline and the relative behaviour of the ablations, not the headline table values.

## D2 — Default training budget is reduced from the reported A100 schedule
`regimes/main.toml` keeps the paper hyperparameters (Sec. III, Table V) as the reference values, but the
imagination-rollout step count and epoch counts are honored only on the full schedule. The pytest smoke
config `regimes/_smoke.toml` uses a 2-step budget and is labelled for unit-test use only; it is never a
source of reported numbers.

## D3 — Imputation backend
MVCE + Predictive Mean Matching (Sec. IV-A) is realized with an iterative conditional-mean imputer over
the synthetic features rather than a specific external MICE implementation, because the synthetic cohort
is generated without missingness beyond the controlled masking study (Table VIII). The public API and
the masking semantics match the manuscript; the numerical imputer differs.
