from __future__ import annotations

import numpy as np
import numpy.typing as npt
from sklearn.metrics import roc_auc_score

FloatArray = npt.NDArray[np.float64]


def rmse(prediction: FloatArray, target: FloatArray) -> float:
    return float(np.sqrt(np.mean((prediction - target) ** 2)))


def mae(prediction: FloatArray, target: FloatArray) -> float:
    return float(np.mean(np.abs(prediction - target)))


def conversion_auc(scores: FloatArray, labels: npt.NDArray[np.int64]) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def counterfactual_fidelity(
    predicted: FloatArray, reference: FloatArray, value_range: float
) -> float:
    rmsd = float(np.sqrt(np.mean((predicted - reference) ** 2)))
    if value_range <= 0.0:
        return float("nan")
    return 1.0 - rmsd / value_range
