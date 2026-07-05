"""False-negative review helpers for recall-first error analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from levers.baseline import DEFAULT_THRESHOLDS, BaselinePrediction
from levers.eval import BinaryMetrics, binary_classification_metrics
from levers.threshold_tuning import run_sample_threshold_tuning


@dataclass(frozen=True)
class FalseNegativeReviewRow:
    """One ranked false-negative candidate for human visual review."""

    rank: int
    image_path: str
    group_id: str
    split: str
    label: int
    score: float
    threshold: float
    margin: float
    review_bucket: str

    def as_dict(self) -> dict[str, str | int | float]:
        return {
            "rank": self.rank,
            "image_path": self.image_path,
            "group_id": self.group_id,
            "split": self.split,
            "label": self.label,
            "score": self.score,
            "threshold": self.threshold,
            "margin": self.margin,
            "review_bucket": self.review_bucket,
        }


@dataclass(frozen=True)
class FalseNegativeReviewResult:
    """False-negative review grid plus metrics at the review threshold."""

    review_rows: tuple[FalseNegativeReviewRow, ...]
    predictions: tuple[BaselinePrediction, ...]
    review_metrics: BinaryMetrics
    review_threshold: float
    selected_threshold: float
    review_split: str
    total_false_negatives: int
    sample_size: int
    group_count: int
    train_size: int
    val_size: int
    test_size: int

    def review_rows_payload(self) -> list[dict[str, str | int | float]]:
        return [row.as_dict() for row in self.review_rows]

    def bucket_counts_payload(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in self.review_rows:
            counts[row.review_bucket] = counts.get(row.review_bucket, 0) + 1
        return counts

    def metric_payload(self) -> dict[str, float | int | None]:
        payload = self.review_metrics.as_dict()
        payload.update(
            {
                "review_threshold": self.review_threshold,
                "selected_threshold": self.selected_threshold,
                "threshold_delta": self.review_threshold - self.selected_threshold,
                "total_false_negatives": self.total_false_negatives,
                "reviewed_false_negatives": len(self.review_rows),
            }
        )
        return payload

    def summary_payload(self) -> dict[str, str | int | float | dict[str, int]]:
        return {
            "lever_name": "false_negative_review",
            "review_split": self.review_split,
            "review_threshold": self.review_threshold,
            "selected_threshold": self.selected_threshold,
            "total_false_negatives": self.total_false_negatives,
            "reviewed_false_negatives": len(self.review_rows),
            "sample_size": self.sample_size,
            "group_count": self.group_count,
            "bucket_counts": self.bucket_counts_payload(),
        }

    def leaderboard_row_payload(self) -> dict[str, str | int | float | None]:
        return {
            "lever_name": "false_negative_review",
            "lever_type": "analysis_artifact",
            "review_split": self.review_split,
            "threshold": self.review_threshold,
            "recall_defective": self.review_metrics.recall_defective,
            "precision_defective": self.review_metrics.precision_defective,
            "f1_defective": self.review_metrics.f1_defective,
            "false_negatives": self.review_metrics.false_negatives,
            "reviewed_false_negatives": len(self.review_rows),
            "recommended_next_action": "inspect_false_negative_review_rows",
        }


def _bucket_false_negative(
    margin: float,
    *,
    borderline_margin: float,
    high_confidence_margin: float,
) -> str:
    if margin <= borderline_margin:
        return "borderline_threshold_miss"
    if margin >= high_confidence_margin:
        return "high_confidence_good_prediction"
    return "medium_confidence_good_prediction"


def _validate_review_settings(
    *,
    top_k: int | None,
    borderline_margin: float,
    high_confidence_margin: float,
) -> None:
    if top_k is not None and top_k < 0:
        raise ValueError("top_k must be non-negative")
    if borderline_margin < 0:
        raise ValueError("borderline_margin must be non-negative")
    if high_confidence_margin < 0:
        raise ValueError("high_confidence_margin must be non-negative")
    if high_confidence_margin < borderline_margin:
        raise ValueError("high_confidence_margin must be >= borderline_margin")


def build_false_negative_review_rows(
    predictions: Iterable[BaselinePrediction],
    *,
    threshold: float,
    split: str = "test",
    positive_label: int = 1,
    top_k: int | None = None,
    borderline_margin: float = 0.05,
    high_confidence_margin: float = 0.25,
) -> tuple[FalseNegativeReviewRow, ...]:
    """Return ranked false negatives for a split at a given threshold."""

    _validate_review_settings(
        top_k=top_k,
        borderline_margin=borderline_margin,
        high_confidence_margin=high_confidence_margin,
    )
    false_negatives: list[tuple[BaselinePrediction, float]] = []
    for prediction in predictions:
        if prediction.split != split:
            continue
        if prediction.label != positive_label:
            continue
        if prediction.score >= threshold:
            continue
        false_negatives.append((prediction, float(threshold - prediction.score)))

    ranked = sorted(
        false_negatives,
        key=lambda item: (-item[1], item[0].score, item[0].image_path),
    )
    if top_k is not None:
        ranked = ranked[:top_k]

    return tuple(
        FalseNegativeReviewRow(
            rank=index,
            image_path=prediction.image_path,
            group_id=prediction.group_id,
            split=prediction.split,
            label=prediction.label,
            score=float(prediction.score),
            threshold=float(threshold),
            margin=margin,
            review_bucket=_bucket_false_negative(
                margin,
                borderline_margin=borderline_margin,
                high_confidence_margin=high_confidence_margin,
            ),
        )
        for index, (prediction, margin) in enumerate(ranked, start=1)
    )


def _split_predictions(
    predictions: Sequence[BaselinePrediction],
    split: str,
) -> list[BaselinePrediction]:
    return [prediction for prediction in predictions if prediction.split == split]


def review_false_negatives_from_predictions(
    predictions: Iterable[BaselinePrediction],
    *,
    review_threshold: float,
    selected_threshold: float,
    review_split: str = "test",
    positive_label: int = 1,
    top_k: int | None = 20,
    borderline_margin: float = 0.05,
    high_confidence_margin: float = 0.25,
) -> FalseNegativeReviewResult:
    """Build false-negative review rows and split metrics from predictions."""

    prediction_rows = tuple(predictions)
    if not prediction_rows:
        raise ValueError("false-negative review requires at least one prediction")

    split_rows = _split_predictions(prediction_rows, review_split)
    if not split_rows:
        raise ValueError(f"false-negative review requires {review_split!r} predictions")

    all_review_rows = build_false_negative_review_rows(
        prediction_rows,
        threshold=review_threshold,
        split=review_split,
        positive_label=positive_label,
        top_k=None,
        borderline_margin=borderline_margin,
        high_confidence_margin=high_confidence_margin,
    )
    review_rows = all_review_rows if top_k is None else all_review_rows[:top_k]
    labels = [prediction.label for prediction in split_rows]
    scores = [prediction.score for prediction in split_rows]
    review_metrics = binary_classification_metrics(
        labels,
        scores,
        threshold=review_threshold,
        positive_label=positive_label,
    )

    return FalseNegativeReviewResult(
        review_rows=tuple(review_rows),
        predictions=prediction_rows,
        review_metrics=review_metrics,
        review_threshold=float(review_threshold),
        selected_threshold=float(selected_threshold),
        review_split=review_split,
        total_false_negatives=len(all_review_rows),
        sample_size=len(prediction_rows),
        group_count=len({prediction.group_id for prediction in prediction_rows}),
        train_size=len(_split_predictions(prediction_rows, "train")),
        val_size=len(_split_predictions(prediction_rows, "val")),
        test_size=len(_split_predictions(prediction_rows, "test")),
    )


def run_sample_false_negative_review(
    *,
    sample_mode: bool = True,
    split_seed: int = 1,
    feature_seed: int = 17,
    min_recall: float = 0.75,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    baseline_threshold: float = 0.5,
    review_threshold: float | None = None,
    review_split: str = "test",
    top_k: int | None = 20,
    borderline_margin: float = 0.05,
    high_confidence_margin: float = 0.25,
) -> FalseNegativeReviewResult:
    """Run the sample baseline and create a false-negative review grid."""

    threshold_result = run_sample_threshold_tuning(
        sample_mode=sample_mode,
        split_seed=split_seed,
        feature_seed=feature_seed,
        min_recall=min_recall,
        thresholds=thresholds,
        baseline_threshold=baseline_threshold,
    )
    selected_threshold = threshold_result.selected_point.threshold
    return review_false_negatives_from_predictions(
        threshold_result.predictions,
        review_threshold=selected_threshold if review_threshold is None else review_threshold,
        selected_threshold=selected_threshold,
        review_split=review_split,
        top_k=top_k,
        borderline_margin=borderline_margin,
        high_confidence_margin=high_confidence_margin,
    )
