from __future__ import annotations

import msgspec


class CatchmentConfig(msgspec.Struct, frozen=True):
    subjects: int = 1247
    timepoints: int = 6
    node_dim: int = 16
    ground_truth_edges: int = 47
    genomic_dim: int = 8
    proteomic_dim: int = 24
    transcriptomic_dim: int = 33
    imaging_dim: int = 18
    val_fraction: float = 0.15
    test_fraction: float = 0.25
    missing_rate: float = 0.0
    masked_modalities: tuple[str, ...] = ()
    cognition_scale: str = "mmse"
    seed: int = 0

    @property
    def feature_dim(self) -> int:
        return self.genomic_dim + self.proteomic_dim + self.transcriptomic_dim + self.imaging_dim


class ChannelConfig(msgspec.Struct, frozen=True):
    sem_hidden: int = 64
    lambda_l1: float = 0.01
    lambda_prior: float = 5.0
    prior_margin: float = 0.1
    target_degree: float = 3.7
    epochs: int = 200
    learning_rate: float = 1e-3
    dual_step: float = 1.0
    penalty_growth: float = 3.0
    acyclicity_tol: float = 1e-8
    outer_steps: int = 15
    temporal: bool = True
    use_prior: bool = True
    edge_threshold: float = 0.05


class RoutingConfig(msgspec.Struct, frozen=True):
    latent_dim: int = 128
    transition_hidden: int = 32
    encoder_hidden: int = 128
    beta_start: float = 0.01
    beta_end: float = 1.0
    beta_warmup_epochs: int = 50
    epochs: int = 200
    learning_rate: float = 1e-3
    horizon: int = 12
    gamma: float = 0.99
    rollout_steps: int = 500000
    graph_structured: bool = True


class RegulationConfig(msgspec.Struct, frozen=True):
    action_dim: int = 3
    q_hidden: tuple[int, int] = (256, 128)
    policy_hidden: tuple[int, int] = (256, 128)
    learning_rate: float = 3e-4
    buffer_size: int = 100000
    steps: int = 500000
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    target_entropy_scale: float = 1.0
    reward_weights: tuple[float, float, float] = (0.5, 0.3, 0.2)


class WorksConfig(msgspec.Struct, frozen=True):
    device: str = "cpu"
    out_dir: str = "runs/main"
    seed: int = 0
    log_every: int = 50


class Config(msgspec.Struct, frozen=True):
    name: str = "main"
    catchment: CatchmentConfig = msgspec.field(default_factory=CatchmentConfig)
    channels: ChannelConfig = msgspec.field(default_factory=ChannelConfig)
    routing: RoutingConfig = msgspec.field(default_factory=RoutingConfig)
    regulation: RegulationConfig = msgspec.field(default_factory=RegulationConfig)
    works: WorksConfig = msgspec.field(default_factory=WorksConfig)
