import pytest

from levers.eval import binary_classification_metrics
from levers.thresholds import choose_threshold_for_recall, threshold_sweep


def test_binary_metrics_are_recall_first_for_positive_class():
    metrics = binary_classification_metrics(
        y_true=[1, 1, 0, 0],
        y_score=[0.9, 0.4, 0.8, 0.1],
        threshold=0.5,
    )

    assert metrics.true_positives == 1
    assert metrics.false_positives == 1
    assert metrics.true_negatives == 1
    assert metrics.false_negatives == 1
    assert metrics.recall_defective == 0.5
    assert metrics.precision_defective == 0.5


def test_threshold_sweep_and_recall_choice():
    points = threshold_sweep(
        y_true=[1, 1, 0, 0],
        y_score=[0.9, 0.4, 0.8, 0.1],
        thresholds=[0.3, 0.5, 0.85],
    )

    chosen = choose_threshold_for_recall(points, min_recall=1.0)
    assert chosen.threshold == 0.3
    assert chosen.metrics.recall_defective == 1.0


def test_metrics_reject_empty_inputs():
    with pytest.raises(ValueError, match="at least one example"):
        binary_classification_metrics([], [])
