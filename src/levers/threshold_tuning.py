"""Threshold tuning lever over baseline prediction scores."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from levers.baseline import (
    DEFAULT_THRESHOLDS,
    BaselinePrediction,
    run_sample_baseline,
)
from levers.eval import BinaryMetrics, binary_classification_metrics
from levers.thresholds import ThresholdPoint, choose_threshold_for_recall, threshold_sweep


@dataclass(frozen=True)
class ThresholdTuningResult:
    """Measured threshold-tuning comparison against a fixed operating point."""

    selected_point: ThresholdPoint
    tuned_test_metrics: BinaryMetrics
    fixed_test_metrics: BinaryMetrics
    validation_sweep: tuple[ThresholdPoint, ...]
    predictions: tuple[BaselinePrediction, ...]
    sample_size: int
    group_count: int
    train_size: int
    val_size: int
    test_size: int
    baseline_threshold: float

    def tuned_metric_payload(self) -> dict[str, float | int | None]:
        return self.tuned_test_metrics.as_dict()

    def validation_metric_payload(self) -> dict[str, float | int | None]:
        return self.selected_point.metrics.as_dict()

    def fixed_metric_payload(self) -> dict[str, float | int | None]:
        return self.fixed_test_metrics.as_dict()

    def delta_metric_payload(self) -> dict[str, float | int]:
        """Return tuned-minus-fixed comparison metrics."""

        return {
            "threshold": self.selected_point.threshold - self.baseline_threshold,
            "recall_defective": (
                self.tuned_test_metrics.recall_defective
                - self.fixed_test_metrics.recall_defective
            ),
            "precision_defective": (
                self.tuned_test_metrics.precision_defective
                - self.fixed_test_metrics.precision_defective
            ),
            "f1_defective": (
                self.tuned_test_metrics.f1_defective
                - self.fixed_test_metrics.f1_defective
            ),
            "false_negatives": (
                self.tuned_test_metrics.false_negatives
                - self.fixed_test_metrics.false_negatives
            ),
        }

    def validation_sweep_payload(self) -> list[dict[str, float | int | None]]:
        return [point.as_dict() for point in self.validation_sweep]


def _split_predictions(
    predictions: Sequence[BaselinePrediction],
    split: str,
) -> list[BaselinePrediction]:
    return [prediction for prediction in predictions if prediction.split == split]


def _labels_and_scores(
    predictions: Sequence[BaselinePrediction],
) -> tuple[list[int], list[float]]:
    return (
        [prediction.label for prediction in predictions],
        [prediction.score for prediction in predictions],
    )


def tune_threshold_from_predictions(
    predictions: Iterable[BaselinePrediction],
    *,
    min_recall: float = 0.75,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    baseline_threshold: float = 0.5,
    positive_label: int = 1,
) -> ThresholdTuningResult:
    """Tune a validation threshold and compare it with a fixed test threshold."""

    prediction_rows = tuple(predictions)
    if not prediction_rows:
        raise ValueError("threshold tuning requires at least one prediction")

    val_predictions = _split_predictions(prediction_rows, "val")
    test_predictions = _split_predictions(prediction_rows, "test")
    if not val_predictions:
        raise ValueError("threshold tuning requires validation predictions")
    if not test_predictions:
        raise ValueError("threshold tuning requires test predictions")

    val_labels, val_scores = _labels_and_scores(val_predictions)
    validation_sweep = tuple(
        threshold_sweep(
            val_labels,
            val_scores,
            thresholds=thresholds,
            positive_label=positive_label,
        )
    )
    selected_point = choose_threshold_for_recall(
        validation_sweep,
        min_recall=min_recall,
    )

    test_labels, test_scores = _labels_and_scores(test_predictions)
    tuned_test_metrics = binary_classification_metrics(
        test_labels,
        test_scores,
        threshold=selected_point.threshold,
        positive_label=positive_label,
    )
    fixed_test_metrics = binary_classification_metrics(
        test_labels,
        test_scores,
        threshold=baseline_threshold,
        positive_label=positive_label,
    )

    return ThresholdTuningResult(
        selected_point=selected_point,
        tuned_test_metrics=tuned_test_metrics,
        fixed_test_metrics=fixed_test_metrics,
        validation_sweep=validation_sweep,
        predictions=prediction_rows,
        sample_size=len(prediction_rows),
        group_count=len({prediction.group_id for prediction in prediction_rows}),
        train_size=len(_split_predictions(prediction_rows, "train")),
        val_size=len(val_predictions),
        test_size=len(test_predictions),
        baseline_threshold=float(baseline_threshold),
    )


def run_sample_threshold_tuning(
    *,
    sample_mode: bool = True,
    split_seed: int = 1,
    feature_seed: int = 17,
    min_recall: float = 0.75,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    baseline_threshold: float = 0.5,
) -> ThresholdTuningResult:
    """Run the sample baseline scorer, then tune only the operating threshold."""

    baseline_result = run_sample_baseline(
        sample_mode=sample_mode,
        split_seed=split_seed,
        feature_seed=feature_seed,
        min_recall=0.0,
        thresholds=thresholds,
    )
    return tune_threshold_from_predictions(
        baseline_result.predictions,
        min_recall=min_recall,
        thresholds=thresholds,
        baseline_threshold=baseline_threshold,
    )
