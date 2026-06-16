from __future__ import annotations

import numpy as np
import numpy.typing as npt

BoolArray = npt.NDArray[np.bool_]


def structural_hamming(predicted: BoolArray, truth: BoolArray) -> int:
    return int(np.sum(predicted != truth))


def shd_accuracy(predicted: BoolArray, truth: BoolArray) -> float:
    edges = int(np.sum(truth))
    if edges == 0:
        return float("nan")
    return 1.0 - structural_hamming(predicted, truth) / edges


def edge_scores(predicted: BoolArray, truth: BoolArray) -> dict[str, float]:
    true_positive = int(np.sum(predicted & truth))
    false_positive = int(np.sum(predicted & ~truth))
    false_negative = int(np.sum(~predicted & truth))
    precision = (
        true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    )
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    fdr = (
        false_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    )
    return {"f1": f1, "tpr": recall, "fdr": fdr, "precision": precision}
