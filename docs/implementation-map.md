# Implementation Map — CausalDT-AD

Source code carries no inline comments and no docstrings; this table is the sole provenance record
mapping each manuscript item to the module that realizes it.

## Module 1 — Causal Discovery Engine (CDE)

| paper item | location | file | what it implements |
|---|---|---|---|
| Nonlinear SEM, V_j = f_j(Pa_j;θ_j)+ε_j | Eq. 1, Sec. III-C | `causaldt_ad/channels/sem.py` | per-node two-layer MLP (64 hidden, GELU), ε_j ~ N(0,σ_j²) |
| NOTEARS acyclicity h(W)=tr(e^{W∘W})−d | Eq. 2, Sec. II-A/III-C | `causaldt_ad/channels/acyclicity.py` | matrix-exponential constraint + its gradient |
| CDE objective (Gaussian NLL + λ1‖W‖₁ + λ2 L_prior) | Eq. 3, Sec. III-C | `causaldt_ad/channels/discover.py` | augmented-Lagrangian fit of W,θ |
| Aβ–LD pathway prior L_prior | Eq. 4, Sec. III-C | `causaldt_ad/channels/pathway_prior.py` | supported-edge hinge max(0,δ−|W|) + forbidden-edge |W| |
| DBN temporal extension (inter-slice edges) | Sec. III-C, Sec. IV-E | `causaldt_ad/channels/discover.py` | cross-time-slice edges, within-slice acyclicity |
| Algorithm 1 (CDE optimization loop) | Sec. III-C | `causaldt_ad/channels/discover.py` | sparse DAG recovery, avg degree 3.7 |
| 8×8 grid, d=83 features, 47 ground-truth edges | Sec. IV-C, Fig. 1/2 | `regimes/main.toml` | configuration defaults |

## Module 2 — Causal World Model (CWM)

| paper item | location | file | what it implements |
|---|---|---|---|
| Encoder q_φ(z^t | x^{≤t}) | Sec. III-D | `causaldt_ad/routing/intake.py` | recurrent multi-omic encoder to latent d_z=128 |
| Graph-structured transition p_ψ (Eq. 5) | Eq. 5, Sec. III-D-1 | `causaldt_ad/routing/transit.py` | per-latent Gaussian MLP over causal parents Pa(j) only |
| Decoder p_ξ(x^{t+1} | z^{t+1}) | Sec. III-D | `causaldt_ad/routing/outfall.py` | latent-to-observation reconstruction |
| ELBO + β-annealing 0.01→1.0 (50 ep) | Eq. 6, Sec. III-D-2 | `causaldt_ad/routing/reservoir.py` | training objective + linear β schedule |
| Imagination rollout (500k steps, 12 mo, γ=0.99) | Sec. III-D, Fig. 1/2 | `causaldt_ad/routing/reservoir.py` | latent rollouts for RL |
| Counterfactual via do-operator | Sec. III-D-3 | `causaldt_ad/routing/sluice.py` | do(z_Aβ=v), causal-path propagation, non-descendants fixed |

## Module 3 — RL Therapy Optimizer (RTO)

| paper item | location | file | what it implements |
|---|---|---|---|
| MDP (S,A,P,R,γ); action a∈[0,1]³ | Sec. III-E-1 | `causaldt_ad/regulation/regime.py` | latent state, 3-channel intervention action |
| Reward R^t (Eq. 7), α=(0.5,0.3,0.2) | Eq. 7, Sec. III-E | `causaldt_ad/regulation/abstraction.py` | cognitive preservation + biomarker normalization − cost |
| Soft Actor-Critic (twin Q 256-128, Gaussian policy) | Sec. III-E-2 | `causaldt_ad/regulation/weir.py` | twin critics + entropy-regularized actor |
| SAC training in imagination (500k steps, lr 3e-4, buffer 1e5) | Sec. III-E-2 | `causaldt_ad/regulation/regime.py` | off-policy update on imagined transitions |

## Decision pipeline & orchestration

| paper item | location | file |
|---|---|---|
| Four-stage care pipeline (twin instantiation → early-detection scoring → prognostic band → clinician-in-the-loop) | Sec. III-F | `causaldt_ad/works/pipeline.py` |
| Three-phase training timeline (CDE → CWM frozen-G → RTO frozen-CWM) | Fig. 1, Sec. III | `causaldt_ad/works/programme.py` |
| set_seed, atomic checkpoint writes | R3/R4 | `causaldt_ad/works/levels.py` |
| End-to-end assembled CausalDT-AD | Fig. 1/2 | `causaldt_ad/confluence.py` |

## Data

| paper item | location | file |
|---|---|---|
| Multi-omic inputs (genomic/proteomic/transcriptomic/imaging) | Sec. III-A, IV-A | `causaldt_ad/catchment/gauges.py` |
| Synthetic cohort (A/T/N + KL-stage anchored) | surrogate for restricted ADNI/OASIS-3 | `causaldt_ad/catchment/inflow.py` |
| z-score + MVCE/PMM imputation, 60/15/25 split, 30% masking | Sec. IV-A, Table VIII | `causaldt_ad/catchment/confluence_ops.py` |
| Subject/occasion typed records | Sec. III-A | `causaldt_ad/catchment/parcel.py` |

## Metrics & statistics (evaluation tables)

| paper item | location | file |
|---|---|---|
| RMSE / MAE / AUC (MCI→AD) / CFS (Table I, III, VII) | Sec. IV-C | `causaldt_ad/gauging/discharge.py` |
| SHD accuracy / F1 / TPR / FDR (Table II) | Sec. IV-C/E | `causaldt_ad/gauging/rating.py` |
| Bootstrap CI, paired t-test (Table VI), 5-seed aggregation | Sec. IV-C, IV-K | `causaldt_ad/gauging/statistics.py` |

## Experiment configurations

| paper item | file |
|---|---|
| Main result (Table I, II, III) | `regimes/main.toml` |
| Ablations (Table IV): −prior, −temporal, −graph transition, −β-anneal, −RL | `regimes/ablation_*.toml` |
| Cross-dataset transfer (Table VII), missing-modality (Table VIII) | `regimes/supplementary_*.toml` |
| pytest smoke only | `regimes/_smoke.toml` |

## Code availability (verbatim, manuscript Sec. "Code Availability")

> The implementation of CausalDT-AD—including the CDE, CWM, and RTO modules, hyperparameter
> configurations, fixed random seeds, training scripts, and reproducibility instructions—will be made
> available to reviewers and editors upon reasonable request during the review process, and will be
> released to the community upon publication of the manuscript.

## Citation

Y. Dou, Y. Zhou, M. Zeng, and J. Fu, "Patient-Centered Causal Digital Twins of the Amyloid–Lysosomal
Loop for Early Detection and Prognosis in Alzheimer's Disease," IEEE Journal of Biomedical and Health
Informatics.
