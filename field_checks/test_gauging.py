from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from causaldt_ad.gauging.discharge import conversion_auc, counterfactual_fidelity, mae, rmse
from causaldt_ad.gauging.rating import edge_scores, shd_accuracy, structural_hamming
from causaldt_ad.gauging.statistics import bootstrap_ci, cohens_d, paired_ttest


def test_rmse_and_mae() -> None:
    prediction = np.array([1.0, 2.0, 3.0])
    target = np.array([1.0, 2.0, 5.0])
    assert rmse(prediction, target) == pytest.approx(np.sqrt(4.0 / 3.0))
    assert mae(prediction, target) == pytest.approx(2.0 / 3.0)


def test_conversion_auc_matches_sklearn() -> None:
    scores = np.array([0.1, 0.4, 0.35, 0.8])
    labels = np.array([0, 0, 1, 1])
    assert conversion_auc(scores, labels) == pytest.approx(roc_auc_score(labels, scores))


def test_counterfactual_fidelity_bounds() -> None:
    perfect = counterfactual_fidelity(np.zeros(5), np.zeros(5), 4.0)
    assert perfect == pytest.approx(1.0)


def test_structural_metrics() -> None:
    truth = np.array([[False, True, False], [False, False, True], [False, False, False]])
    predicted = np.array([[False, True, True], [False, False, True], [False, False, False]])
    assert structural_hamming(predicted, truth) == 1
    assert shd_accuracy(predicted, truth) == pytest.approx(1.0 - 1.0 / 2.0)
    scores = edge_scores(predicted, truth)
    assert scores["tpr"] == pytest.approx(1.0)
    assert scores["fdr"] == pytest.approx(1.0 / 3.0)


def test_bootstrap_and_tests() -> None:
    values = np.array([2.40, 2.41, 2.39, 2.42, 2.38])
    lower, upper = bootstrap_ci(values, iterations=200, alpha=0.05, seed=0)
    assert lower <= float(np.mean(values)) <= upper
    reference = np.array([3.0, 3.3, 2.8, 3.15, 3.05])
    contender = np.array([2.4, 2.5, 2.2, 2.55, 2.35])
    statistic, pvalue = paired_ttest(reference, contender)
    assert statistic > 0
    assert pvalue < 0.05
    assert cohens_d(reference, contender) > 0
