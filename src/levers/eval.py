"""Binary classification metrics for recall-first evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


@dataclass(frozen=True)
class BinaryMetrics:
    threshold: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    recall_defective: float
    precision_defective: float
    f1_defective: float
    accuracy: float
    auc_pr: float | None = None
    auc_roc: float | None = None

    def as_dict(self) -> dict[str, float | int | None]:
        return {
            "threshold": self.threshold,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "true_negatives": self.true_negatives,
            "false_negatives": self.false_negatives,
            "recall_defective": self.recall_defective,
            "precision_defective": self.precision_defective,
            "f1_defective": self.f1_defective,
            "accuracy": self.accuracy,
            "auc_pr": self.auc_pr,
            "auc_roc": self.auc_roc,
        }


def _safe_divide(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def _ranking_metrics(true: np.ndarray, score: np.ndarray, positive_label: int) -> tuple[float | None, float | None]:
    positive = true == positive_label
    if len(np.unique(positive)) < 2:
        return None, None
    return (
        float(average_precision_score(positive, score)),
        float(roc_auc_score(positive, score)),
    )


def binary_classification_metrics(
    y_true: Iterable[int],
    y_score: Iterable[float],
    *,
    threshold: float = 0.5,
    positive_label: int = 1,
) -> BinaryMetrics:
    """Compute defective-class metrics from labels and positive scores."""

    true = np.asarray(list(y_true))
    score = np.asarray(list(y_score), dtype=float)
    if true.shape[0] != score.shape[0]:
        raise ValueError("y_true and y_score must have the same length")
    if true.shape[0] == 0:
        raise ValueError("at least one example is required")

    true_positive_mask = true == positive_label
    predicted_positive_mask = score >= threshold

    tp = int(np.logical_and(true_positive_mask, predicted_positive_mask).sum())
    fp = int(np.logical_and(~true_positive_mask, predicted_positive_mask).sum())
    tn = int(np.logical_and(~true_positive_mask, ~predicted_positive_mask).sum())
    fn = int(np.logical_and(true_positive_mask, ~predicted_positive_mask).sum())

    recall = _safe_divide(tp, tp + fn)
    precision = _safe_divide(tp, tp + fp)
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    accuracy = _safe_divide(tp + tn, tp + fp + tn + fn)
    auc_pr, auc_roc = _ranking_metrics(true, score, positive_label)

    return BinaryMetrics(
        threshold=float(threshold),
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
        recall_defective=recall,
        precision_defective=precision,
        f1_defective=f1,
        accuracy=accuracy,
        auc_pr=auc_pr,
        auc_roc=auc_roc,
    )
