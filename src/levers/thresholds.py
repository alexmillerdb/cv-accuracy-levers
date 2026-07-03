"""Threshold sweep helpers for recall-first operating point selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from levers.eval import BinaryMetrics, binary_classification_metrics


@dataclass(frozen=True)
class ThresholdPoint:
    threshold: float
    metrics: BinaryMetrics

    def as_dict(self) -> dict[str, float | int]:
        values = self.metrics.as_dict()
        values["threshold"] = self.threshold
        return values


def threshold_sweep(
    y_true: Iterable[int],
    y_score: Iterable[float],
    *,
    thresholds: Sequence[float] | None = None,
    positive_label: int = 1,
) -> list[ThresholdPoint]:
    """Return metrics across candidate thresholds."""

    true_list = list(y_true)
    score_list = list(y_score)
    if thresholds is None:
        thresholds = tuple(float(value) for value in np.linspace(0.0, 1.0, 21))

    points = [
        ThresholdPoint(
            threshold=float(threshold),
            metrics=binary_classification_metrics(
                true_list,
                score_list,
                threshold=float(threshold),
                positive_label=positive_label,
            ),
        )
        for threshold in thresholds
    ]
    return sorted(points, key=lambda point: point.threshold)


def choose_threshold_for_recall(
    points: Sequence[ThresholdPoint],
    *,
    min_recall: float,
) -> ThresholdPoint:
    """Pick the highest-precision point that satisfies a recall floor."""

    eligible = [
        point
        for point in points
        if point.metrics.recall_defective >= min_recall
    ]
    if not eligible:
        raise ValueError(f"no threshold satisfies min_recall={min_recall}")
    return max(
        eligible,
        key=lambda point: (
            point.metrics.precision_defective,
            point.metrics.f1_defective,
            point.threshold,
        ),
    )
