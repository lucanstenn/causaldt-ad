from __future__ import annotations

from causaldt_ad.catchment import parcel

SUPPORTED_EDGES: tuple[tuple[str, str], ...] = (
    ("amyloid_beta", "vatpase"),
    ("vatpase", "lyso_ph"),
    ("lyso_ph", "autophagic_flux"),
    ("lyso_genes", "vatpase"),
    ("autophagic_flux", "amyloid_suvr"),
    ("amyloid_beta", "amyloid_suvr"),
    ("amyloid_suvr", "ptau"),
    ("ptau", "nfl"),
    ("nfl", "hippocampus"),
    ("hippocampus", "mmse"),
    ("cortical_thickness", "mmse"),
)

FORBIDDEN_EDGES: tuple[tuple[str, str], ...] = (
    ("mmse", "amyloid_beta"),
    ("mmse", "apoe"),
    ("nfl", "apoe"),
    ("hippocampus", "prs"),
    ("amyloid_suvr", "apoe"),
)


def _to_indices(pairs: tuple[tuple[str, str], ...], node_dim: int) -> tuple[tuple[int, int], ...]:
    names = parcel.node_names(node_dim)
    resolved: list[tuple[int, int]] = []
    for parent, child in pairs:
        if parent in names and child in names:
            resolved.append((names.index(parent), names.index(child)))
    return tuple(resolved)


def supported_pairs(node_dim: int) -> tuple[tuple[int, int], ...]:
    return _to_indices(SUPPORTED_EDGES, node_dim)


def forbidden_pairs(node_dim: int) -> tuple[tuple[int, int], ...]:
    return _to_indices(FORBIDDEN_EDGES, node_dim)
