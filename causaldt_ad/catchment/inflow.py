from __future__ import annotations

import numpy as np

from causaldt_ad.catchment import parcel
from causaldt_ad.catchment.pathway import forbidden_pairs, supported_pairs
from causaldt_ad.drawings.schema import CatchmentConfig


def _truth_graph(node_dim: int, target_edges: int, rng: np.random.Generator) -> np.ndarray:
    weight = np.zeros((node_dim, node_dim), dtype=np.float64)
    blocked = {(i, j) for i, j in forbidden_pairs(node_dim)}
    for i, j in supported_pairs(node_dim):
        lo, hi = (i, j) if i < j else (j, i)
        magnitude = (
            0.73
            if {parcel.ANCHOR_NODES[i], parcel.ANCHOR_NODES[j]}
            <= {
                "amyloid_beta",
                "lyso_ph",
                "vatpase",
                "autophagic_flux",
            }
            else float(rng.uniform(0.4, 0.9))
        )
        weight[lo, hi] = magnitude
    present = int(np.count_nonzero(weight))
    candidates = [
        (i, j)
        for i in range(node_dim)
        for j in range(i + 1, node_dim)
        if weight[i, j] == 0.0 and (i, j) not in blocked
    ]
    rng.shuffle(candidates)
    for i, j in candidates:
        if present >= target_edges:
            break
        sign = -1.0 if rng.random() < 0.4 else 1.0
        weight[i, j] = sign * float(rng.uniform(0.3, 0.8))
        present += 1
    return weight


def _baseline_severity(rng: np.random.Generator, subjects: int) -> np.ndarray:
    order = np.argsort(rng.standard_normal(subjects))
    severity = np.empty(subjects, dtype=np.float64)
    cursor = 0
    centers = (-1.1, -0.2, 0.6, 1.4)
    counts = np.array(parcel.GROUP_COUNTS, dtype=np.float64)
    shares = (counts / counts.sum() * subjects).astype(np.int64)
    shares[-1] = subjects - int(shares[:-1].sum())
    group = np.empty(subjects, dtype=np.int64)
    for label, (count, center) in enumerate(zip(shares.tolist(), centers, strict=False)):
        idx = order[cursor : cursor + count]
        severity[idx] = center + 0.25 * rng.standard_normal(count)
        group[idx] = label
        cursor += count
    return np.stack([severity, group.astype(np.float64)], axis=0)


def generate(config: CatchmentConfig) -> parcel.Cohort:
    rng = np.random.default_rng(config.seed)
    d = config.node_dim
    t = config.timepoints
    n = config.subjects
    names = parcel.node_names(d)
    weight = _truth_graph(d, config.ground_truth_edges, rng)

    sev_group = _baseline_severity(rng, n)
    severity0 = sev_group[0]
    group = sev_group[1].astype(np.int64)

    amyloid = names.index("amyloid_beta")
    lyso = names.index("lyso_ph")
    mmse = names.index("mmse")

    indegree = np.count_nonzero(weight, axis=0)
    is_root = indegree == 0
    root_load = np.where(is_root, rng.uniform(0.5, 0.8, size=d), 0.0)
    noise_scale = 0.5
    nodes = np.zeros((n, t, d), dtype=np.float64)
    slope = 0.18
    reverse_gain = 0.3
    for step in range(t):
        severity = severity0 + slope * step + 0.05 * rng.standard_normal(n)
        layer = np.zeros((n, d), dtype=np.float64)
        for j in range(d):
            parents = np.nonzero(weight[:, j])[0]
            value = root_load[j] * severity + noise_scale * rng.standard_normal(n)
            for i in parents:
                value = value + weight[i, j] * layer[:, i]
            if j == amyloid and step > 0:
                value = value + reverse_gain * nodes[:, step - 1, lyso]
            layer[:, j] = value
        nodes[:, step, :] = layer

    cognition = np.clip(30.0 - 9.5 * (0.5 - 0.5 * nodes[:, :, mmse]), 0.0, 30.0)

    slices = parcel.modality_slices(
        config.genomic_dim, config.proteomic_dim, config.transcriptomic_dim, config.imaging_dim
    )
    feat_dim = config.feature_dim
    features = np.zeros((n, t, feat_dim), dtype=np.float64)
    for _name, (lo, hi) in slices.items():
        loading = rng.standard_normal((d, hi - lo))
        emission = nodes @ loading
        features[:, :, lo:hi] = emission + 0.2 * rng.standard_normal((n, t, hi - lo))

    mask = np.ones_like(features, dtype=np.float64)
    split = np.full(n, -1, dtype=np.int64)
    return parcel.Cohort(
        nodes=nodes,
        features=features,
        cognition=cognition,
        mask=mask,
        group=group,
        split=split,
        edges=weight,
        modalities=slices,
    )
