# Project Context — CausalDT-AD

```
project_name       : causaldt_ad                                  [HIGH]
domain             : computational neurology — patient-specific   [HIGH]
                     causal digital twin for early Alzheimer's
                     detection and prognosis (multi-omic)
framework          : PyTorch 2.x (plain torch.nn)                 [HIGH]
venue              : IEEE Journal of Biomedical and Health         [HIGH]
                     Informatics (JBHI)
primary_datasets   : 2 datasets (ADNI, OASIS-3); see §6           [HIGH]
compute_target     : single NVIDIA A100, 14.7 h total train       [HIGH]
hparams_reference  : Table V + Methods prose + Fig. 1/2           [HIGH/MED]
supp_path          : none
extra_signals      : verbatim code-availability quote (docs only);
                     public-data-only ethics; Algorithm 1 (CDE);
                     9 baselines; 8 tables; restricted datasets ->
                     in-repo synthetic cohort surrogate; no
                     released checkpoints
```

## 1. project_name
`causaldt_ad` — the manuscript names the system **CausalDT-AD** (Abstract; Sec. II-B). Package and
console namespace use the snake_case form. [HIGH]

## 2. supp_path
No supplementary material accompanies the manuscript. The reference list and all tables are inline in
the 11-page article; a sibling-directory glob for `*supp*`, `*appendix*`, `*_si.*` returned nothing
relevant. [HIGH]

## 3. domain
Patient-centered causal digital twin for Alzheimer's disease: identifiable causal-graph discovery over a
multi-omic profile (genomic / proteomic / transcriptomic / imaging), a graph-structured latent world
model for trajectory forecasting and counterfactual simulation, and reinforcement-learning therapy
optimization. Derived from Abstract and Sec. III (Methodology). [HIGH]

## 4. framework
PyTorch 2.x with plain `torch.nn`. Evidence: the Causal Discovery Engine extends NOTEARS (Sec. II-A,
Sec. III-C, Eq. 2 uses the matrix-exponential acyclicity penalty); the Causal World Model is a
VAE-style encoder/transition/decoder trained by an ELBO with KL annealing (Sec. III-D, Eq. 6); the RL
Therapy Optimizer uses Soft Actor-Critic with twin Q-networks and a Gaussian policy (Sec. III-E). None
of these requires a specialized library beyond `torch.nn` + a numerical stack. [HIGH]

## 5. venue
IEEE Journal of Biomedical and Health Informatics. Evidence: running header "IEEE JOURNAL OF BIOMEDICAL
AND HEALTH INFORMATICS" on every page; two-column IEEE article layout; Index Terms block; references in
IEEE style; explicit self-positioning against the JBHI digital-twin landscape (Sec. V-C). [HIGH]

## 6. primary_datasets
Both datasets are publicly released but access is application-gated (data-use agreement); neither offers
an anonymous direct download. The repository therefore ships a deterministic, biology-anchored synthetic
multi-omic cohort surrogate; real cohorts are wired through a manifest adapter once credentialed.

| name | version / subset | license / access | URL | link check |
|---|---|---|---|---|
| ADNI | ADNI-1/GO/2/3; 1,247 multi-omic subset (412 CN / 518 EMCI / 204 LMCI / 113 AD) | ADNI Data Use Agreement; application-gated via LONI IDA | https://adni.loni.usc.edu | HTTP 200 |
| OASIS-3 | 1,378 subjects; 2,842 MRI, 2,157 PET, UDS cognitive | OASIS Data Use Terms; application-gated | https://www.oasis-brains.org | HTTP 200 |

Source: Sec. IV-A (Datasets), Ethical Statement, Data Availability. Features standardized by z-score with
MVCE + Predictive Mean Matching imputation; modalities with > 8% missingness flagged. 60/15/25
train/validate/test split stratified by diagnostic group and site; mean/SD over five seeds.

## 7. compute_target
Single NVIDIA A100 (Table V, ADNI). Per-module wall-clock and footprint:

| module | params (M) | train (h) | infer (ms) | mem (GB) |
|---|---|---|---|---|
| CDE | 3.1 | 4.8 | — | 4.9 |
| CWM | 8.7 | 6.2 | 15.6 | 7.4 |
| RTO | 2.4 | 3.7 | — | 3.1 |
| total | 14.2 | 14.7 | 24.8 | 7.4 |

COMPUTE_REPORTED. Source: Table V; cross-checked against Sec. IV-J. [HIGH]

## 8. hparams_reference
Primary table: Table V (compute). Numerical hyperparameters are scattered across Methods prose and
Fig. 1/2 captions:

- CDE: 8×8 variable grid; average degree reduced > 8 -> 3.7; sparsity weight λ1 = 0.01; prior margin
  δ = 0.1; nonlinear SEM column = two-layer MLP (64 hidden, GELU); ε_j ~ N(0, σ_j²); d = 83 observed
  features; 47 ground-truth edges (11 in prior set P, 36 newly discovered). (Eq. 1-4; Sec. III-C;
  Sec. IV-C/E; Fig. 1/2.)
- CWM: latent dim d_z = 128; per-latent transition MLP = 32 hidden; β annealed 0.01 -> 1.0 over the
  first 50 epochs; ≈ 200 epochs total; imagination rollout 500k steps, 12-month horizon, γ = 0.99.
  (Eq. 5-6; Sec. III-D; Fig. 1/2.)
- RTO: Soft Actor-Critic; two Q-networks (256-128 hidden each); one Gaussian policy; learning rate
  3×10⁻⁴; replay buffer 10⁵; 500k imagined steps; action a ∈ [0,1]³ (anti-Aβ, lysosomal acidification
  support, autophagic flux enhancer); reward weights α1 = 0.5, α2 = 0.3, α3 = 0.2; γ = 0.99.
  (Eq. 7; Sec. III-E.)
- Protocol: 5 seeds; 60/15/25 split; 30% modality masking robustness study (Table VIII); OASIS-3 uses
  RL cost weight λ1 ≈ 0.05 (Sec. IV-H). [HIGH/MED]

## 9. extra_signals
- Verbatim code-availability paragraph exists in the manuscript; per the release convention it is kept
  in `implementation-map.md` only and omitted from the README.
- Ethics: only previously published public datasets (ADNI, OASIS-3); no original data generation, no
  human subjects, no biological specimens.
- One algorithm box (Algorithm 1, the Causal Discovery Engine optimization).
- Nine baselines across four families (Table I): VAE-DT, LSTM-DT, DT-GPT, CausalTransformer,
  NOTEARS-MLP, CausalFormer, AD-CausalBiomarker, MultiGAN-AD, MF-SAC.
- Restricted-access datasets -> the repository ships a synthetic cohort; no released checkpoints
  ("upon reasonable request").

NEEDS_USER_DECISION: 0
```
