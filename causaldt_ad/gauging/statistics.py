from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy import stats

FloatArray = npt.NDArray[np.float64]


def aggregate_seeds(values: FloatArray) -> tuple[float, float]:
    return float(np.mean(values)), float(np.std(values))


def bootstrap_ci(
    values: FloatArray, iterations: int, alpha: float, seed: int
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = np.empty(iterations, dtype=np.float64)
    count = values.shape[0]
    for index in range(iterations):
        sample = values[rng.integers(0, count, count)]
        means[index] = float(np.mean(sample))
    lower = float(np.quantile(means, alpha / 2.0))
    upper = float(np.quantile(means, 1.0 - alpha / 2.0))
    return lower, upper


def paired_ttest(reference: FloatArray, contender: FloatArray) -> tuple[float, float]:
    result = stats.ttest_rel(reference, contender)
    return float(result.statistic), float(result.pvalue)


def cohens_d(reference: FloatArray, contender: FloatArray) -> float:
    difference = reference - contender
    spread = float(np.std(difference))
    if spread == 0.0:
        return float("nan")
    return float(np.mean(difference) / spread)
