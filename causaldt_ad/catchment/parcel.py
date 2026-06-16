from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

ANCHOR_NODES: tuple[str, ...] = (
    "amyloid_beta",
    "vatpase",
    "lyso_ph",
    "autophagic_flux",
    "lyso_genes",
    "ptau",
    "nfl",
    "amyloid_suvr",
    "hippocampus",
    "cortical_thickness",
    "apoe",
    "prs",
    "mmse",
    "glucose_metabolism",
    "neuroinflammation",
    "synaptic_density",
)

GROUP_NAMES: tuple[str, ...] = ("CN", "EMCI", "LMCI", "AD")
GROUP_COUNTS: tuple[int, int, int, int] = (412, 518, 204, 113)

REWARD_BIOMARKERS: tuple[str, ...] = ("amyloid_beta", "ptau", "lyso_ph")
ACTION_TARGETS: tuple[str, ...] = ("amyloid_beta", "lyso_ph", "autophagic_flux")
COGNITION_NODE: str = "mmse"

MODALITY_ORDER: tuple[str, ...] = ("genomic", "proteomic", "transcriptomic", "imaging")


def node_names(node_dim: int) -> tuple[str, ...]:
    if node_dim <= len(ANCHOR_NODES):
        return ANCHOR_NODES[:node_dim]
    extra = tuple(f"latent_{i}" for i in range(node_dim - len(ANCHOR_NODES)))
    return ANCHOR_NODES + extra


def node_index(node_dim: int, name: str) -> int:
    return node_names(node_dim).index(name)


def modality_slices(
    genomic: int, proteomic: int, transcriptomic: int, imaging: int
) -> dict[str, tuple[int, int]]:
    sizes = (genomic, proteomic, transcriptomic, imaging)
    bounds: dict[str, tuple[int, int]] = {}
    cursor = 0
    for name, size in zip(MODALITY_ORDER, sizes, strict=False):
        bounds[name] = (cursor, cursor + size)
        cursor += size
    return bounds


@dataclass(frozen=True, slots=True)
class Cohort:
    nodes: FloatArray
    features: FloatArray
    cognition: FloatArray
    mask: FloatArray
    group: IntArray
    split: IntArray
    edges: FloatArray
    modalities: dict[str, tuple[int, int]]

    @property
    def subjects(self) -> int:
        return int(self.nodes.shape[0])

    @property
    def timepoints(self) -> int:
        return int(self.nodes.shape[1])

    @property
    def node_dim(self) -> int:
        return int(self.nodes.shape[2])

    @property
    def feature_dim(self) -> int:
        return int(self.features.shape[2])

    def fold(self, which: int) -> IntArray:
        return np.where(self.split == which)[0].astype(np.int64)
