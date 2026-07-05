"""Crop-first sample A/B comparison helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from levers.baseline import (
    DEFAULT_THRESHOLDS,
    BaselinePrediction,
    build_sample_baseline_dataset,
    train_centroid_baseline,
)
from levers.data import DatasetRecord, make_grouped_split, validate_group_splits
from levers.eval import BinaryMetrics, binary_classification_metrics
from levers.thresholds import ThresholdPoint, choose_threshold_for_recall, threshold_sweep


BASELINE_SCORE_SOURCE = "baseline_whole_image_sample"
CROP_FIRST_SCORE_SOURCE = "crop_first_sample"
CROP_FIRST_FEATURE_SOURCE = "sample_crop_region_emphasis_features_v1"
LEVER_NAME = "crop_first_ab"


@dataclass(frozen=True)
class CropFirstArmResult:
    """One scored arm in the crop-first A/B comparison."""

    arm_name: str
    score_source: str
    selected_point: ThresholdPoint
    test_metrics: BinaryMetrics
    validation_sweep: tuple[ThresholdPoint, ...]
    predictions: tuple[BaselinePrediction, ...]

    def metric_payload(self) -> dict[str, float | int | None]:
        return self.test_metrics.as_dict()

    def validation_metric_payload(self) -> dict[str, float | int | None]:
        return self.selected_point.metrics.as_dict()

    def validation_sweep_payload(self) -> list[dict[str, float | int | None]]:
        return [point.as_dict() for point in self.validation_sweep]

    def predictions_payload(self) -> list[dict[str, str | int | float]]:
        return [prediction.__dict__ for prediction in self.predictions]


@dataclass(frozen=True)
class CropFirstReviewRow:
    """One test-row comparison explaining how crop-first changed a prediction."""

    rank: int
    image_path: str
    group_id: str
    split: str
    label: int
    baseline_score: float
    crop_first_score: float
    score_delta: float
    baseline_threshold: float
    crop_first_threshold: float
    baseline_predicted_label: int
    crop_first_predicted_label: int
    baseline_false_negative: bool
    crop_first_false_negative: bool
    change_type: str
    recall_impact: str

    def as_dict(self) -> dict[str, str | int | float | bool]:
        return {
            "rank": self.rank,
            "image_path": self.image_path,
            "group_id": self.group_id,
            "split": self.split,
            "label": self.label,
            "baseline_score": self.baseline_score,
            "crop_first_score": self.crop_first_score,
            "score_delta": self.score_delta,
            "baseline_threshold": self.baseline_threshold,
            "crop_first_threshold": self.crop_first_threshold,
            "baseline_predicted_label": self.baseline_predicted_label,
            "crop_first_predicted_label": self.crop_first_predicted_label,
            "baseline_false_negative": self.baseline_false_negative,
            "crop_first_false_negative": self.crop_first_false_negative,
            "change_type": self.change_type,
            "recall_impact": self.recall_impact,
        }


@dataclass(frozen=True)
class CropFirstABResult:
    """Baseline-vs-crop-first comparison on the same sample split."""

    baseline: CropFirstArmResult
    crop_first: CropFirstArmResult
    comparison_rows: tuple[CropFirstReviewRow, ...]
    sample_size: int
    group_count: int
    train_size: int
    val_size: int
    test_size: int
    min_recall: float
    crop_emphasis: float
    sample_mode: bool
    split_seed: int | None
    feature_seed: int | None
    review_split: str
    feature_source: str = CROP_FIRST_FEATURE_SOURCE

    @property
    def changed_rows(self) -> tuple[CropFirstReviewRow, ...]:
        return tuple(
            row for row in self.comparison_rows if row.change_type != "unchanged"
        )

    def delta_metric_payload(self) -> dict[str, float | int]:
        baseline = self.baseline.test_metrics
        crop = self.crop_first.test_metrics
        payload: dict[str, float | int] = {
            "threshold": (
                self.crop_first.selected_point.threshold
                - self.baseline.selected_point.threshold
            ),
            "true_positives": crop.true_positives - baseline.true_positives,
            "false_positives": crop.false_positives - baseline.false_positives,
            "true_negatives": crop.true_negatives - baseline.true_negatives,
            "false_negatives": crop.false_negatives - baseline.false_negatives,
            "recall_defective": (
                crop.recall_defective - baseline.recall_defective
            ),
            "precision_defective": (
                crop.precision_defective - baseline.precision_defective
            ),
            "f1_defective": crop.f1_defective - baseline.f1_defective,
            "accuracy": crop.accuracy - baseline.accuracy,
        }
        if baseline.auc_pr is not None and crop.auc_pr is not None:
            payload["auc_pr"] = crop.auc_pr - baseline.auc_pr
        if baseline.auc_roc is not None and crop.auc_roc is not None:
            payload["auc_roc"] = crop.auc_roc - baseline.auc_roc
        return payload

    def metric_payload(self) -> dict[str, float | int | None]:
        payload: dict[str, float | int | None] = {}
        for key, value in self.baseline.metric_payload().items():
            payload[f"baseline.{key}"] = value
        for key, value in self.crop_first.metric_payload().items():
            payload[f"crop_first.{key}"] = value
        for key, value in self.delta_metric_payload().items():
            payload[f"delta.{key}"] = value
        payload["baseline.selected_threshold"] = (
            self.baseline.selected_point.threshold
        )
        payload["crop_first.selected_threshold"] = (
            self.crop_first.selected_point.threshold
        )
        payload["changed_prediction_count"] = len(self.changed_rows)
        payload["false_negative_recovered_count"] = len(
            [
                row
                for row in self.changed_rows
                if row.change_type == "false_negative_recovered"
            ]
        )
        payload["new_false_negative_count"] = len(
            [row for row in self.changed_rows if row.change_type == "new_false_negative"]
        )
        return payload

    def param_payload(self) -> dict[str, str | int | float]:
        payload: dict[str, str | int | float] = {
            "lever_name": LEVER_NAME,
            "lever_type": "ab_comparison",
            "sample_mode": str(self.sample_mode).lower(),
            "sample_size": self.sample_size,
            "group_count": self.group_count,
            "train_size": self.train_size,
            "val_size": self.val_size,
            "test_size": self.test_size,
            "min_recall": self.min_recall,
            "crop_emphasis": self.crop_emphasis,
            "threshold_source": "validation",
            "review_split": self.review_split,
            "baseline_score_source": self.baseline.score_source,
            "crop_first_score_source": self.crop_first.score_source,
            "crop_feature_source": self.feature_source,
            "model_family": "centroid_sample_baseline",
        }
        if self.split_seed is not None:
            payload["split_seed"] = self.split_seed
        if self.feature_seed is not None:
            payload["feature_seed"] = self.feature_seed
        return payload

    def comparison_rows_payload(self) -> list[dict[str, str | int | float | bool]]:
        return [row.as_dict() for row in self.comparison_rows]

    def review_rows_payload(self) -> list[dict[str, str | int | float | bool]]:
        return [row.as_dict() for row in self.changed_rows]

    def validation_sweeps_payload(
        self,
    ) -> dict[str, list[dict[str, float | int | None]]]:
        return {
            "baseline": self.baseline.validation_sweep_payload(),
            "crop_first": self.crop_first.validation_sweep_payload(),
        }

    def predictions_payload(self) -> list[dict[str, str | int | float]]:
        crop_by_path = {
            prediction.image_path: prediction
            for prediction in self.crop_first.predictions
        }
        return [
            {
                "image_path": baseline_prediction.image_path,
                "group_id": baseline_prediction.group_id,
                "split": baseline_prediction.split,
                "label": baseline_prediction.label,
                "baseline_score": baseline_prediction.score,
                "crop_first_score": crop_by_path[
                    baseline_prediction.image_path
                ].score,
            }
            for baseline_prediction in self.baseline.predictions
        ]

    def summary_payload(self) -> dict[str, str | int | float | dict[str, float | int]]:
        return {
            "lever_name": LEVER_NAME,
            "lever_type": "ab_comparison",
            "sample_size": self.sample_size,
            "group_count": self.group_count,
            "train_size": self.train_size,
            "val_size": self.val_size,
            "test_size": self.test_size,
            "min_recall": self.min_recall,
            "crop_emphasis": self.crop_emphasis,
            "baseline_selected_threshold": self.baseline.selected_point.threshold,
            "crop_first_selected_threshold": self.crop_first.selected_point.threshold,
            "baseline_false_negatives": self.baseline.test_metrics.false_negatives,
            "crop_first_false_negatives": self.crop_first.test_metrics.false_negatives,
            "changed_prediction_count": len(self.changed_rows),
            "delta_metrics": self.delta_metric_payload(),
        }

    def leaderboard_row_payload(self) -> dict[str, str | int | float | None]:
        delta = self.delta_metric_payload()
        return {
            "lever_name": LEVER_NAME,
            "lever_type": "ab_comparison",
            "baseline_score_source": self.baseline.score_source,
            "crop_first_score_source": self.crop_first.score_source,
            "crop_feature_source": self.feature_source,
            "baseline_threshold": self.baseline.selected_point.threshold,
            "crop_first_threshold": self.crop_first.selected_point.threshold,
            "baseline_recall_defective": (
                self.baseline.test_metrics.recall_defective
            ),
            "crop_first_recall_defective": (
                self.crop_first.test_metrics.recall_defective
            ),
            "delta_recall_defective": delta["recall_defective"],
            "baseline_precision_defective": (
                self.baseline.test_metrics.precision_defective
            ),
            "crop_first_precision_defective": (
                self.crop_first.test_metrics.precision_defective
            ),
            "delta_precision_defective": delta["precision_defective"],
            "baseline_f1_defective": self.baseline.test_metrics.f1_defective,
            "crop_first_f1_defective": self.crop_first.test_metrics.f1_defective,
            "delta_f1_defective": delta["f1_defective"],
            "baseline_false_negatives": self.baseline.test_metrics.false_negatives,
            "crop_first_false_negatives": self.crop_first.test_metrics.false_negatives,
            "delta_false_negatives": delta["false_negatives"],
            "changed_prediction_count": len(self.changed_rows),
            "recommended_next_action": (
                "inspect_crop_first_review_rows"
                if self.changed_rows
                else "no_sample_prediction_changes"
            ),
        }


def _as_2d_features(features: Sequence[Sequence[float]]) -> np.ndarray:
    feature_array = np.asarray(features, dtype=float)
    if feature_array.ndim != 2:
        raise ValueError("features must be a 2D array-like value")
    if feature_array.shape[0] == 0:
        raise ValueError("at least one feature row is required")
    if feature_array.shape[1] < 4:
        raise ValueError("crop-first sample features require at least four columns")
    return feature_array


def build_crop_first_sample_features(
    whole_image_features: Sequence[Sequence[float]],
    *,
    crop_emphasis: float = 1.35,
) -> np.ndarray:
    """Create deterministic region-emphasis features from whole-image features."""

    if crop_emphasis < 0.0:
        raise ValueError("crop_emphasis must be non-negative")

    features = _as_2d_features(whole_image_features)
    defect_channel = features[:, 0]
    background_channel = features[:, 1]
    texture_channel = features[:, 2]
    context_channel = features[:, 3]
    region_signal = defect_channel + texture_channel - (0.5 * background_channel)

    return np.column_stack(
        [
            defect_channel + (crop_emphasis * region_signal),
            texture_channel + (0.5 * crop_emphasis * (defect_channel - background_channel)),
            background_channel / (1.0 + crop_emphasis),
            context_channel * 0.5,
        ]
    )


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


def _validate_same_prediction_frame(
    baseline_predictions: Sequence[BaselinePrediction],
    crop_first_predictions: Sequence[BaselinePrediction],
) -> None:
    if len(baseline_predictions) != len(crop_first_predictions):
        raise ValueError("baseline and crop-first predictions must have equal length")

    split_by_group: dict[str, str] = {}
    for baseline, crop_first in zip(baseline_predictions, crop_first_predictions):
        if (
            baseline.image_path != crop_first.image_path
            or baseline.group_id != crop_first.group_id
            or baseline.split != crop_first.split
            or baseline.label != crop_first.label
        ):
            raise ValueError(
                "baseline and crop-first predictions must use the same "
                "image, group, split, and label order"
            )
        existing = split_by_group.setdefault(baseline.group_id, baseline.split)
        if existing != baseline.split:
            raise ValueError(
                f"group {baseline.group_id!r} appears in both {existing!r} "
                f"and {baseline.split!r}"
            )


def _evaluate_arm(
    predictions: Sequence[BaselinePrediction],
    *,
    arm_name: str,
    score_source: str,
    min_recall: float,
    thresholds: Sequence[float],
    positive_label: int,
) -> CropFirstArmResult:
    val_predictions = _split_predictions(predictions, "val")
    test_predictions = _split_predictions(predictions, "test")
    if not val_predictions:
        raise ValueError(f"{arm_name} requires validation predictions")
    if not test_predictions:
        raise ValueError(f"{arm_name} requires test predictions")

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
    test_metrics = binary_classification_metrics(
        test_labels,
        test_scores,
        threshold=selected_point.threshold,
        positive_label=positive_label,
    )

    return CropFirstArmResult(
        arm_name=arm_name,
        score_source=score_source,
        selected_point=selected_point,
        test_metrics=test_metrics,
        validation_sweep=validation_sweep,
        predictions=tuple(predictions),
    )


def _predicted_label(
    score: float,
    *,
    threshold: float,
    positive_label: int,
) -> int:
    return positive_label if score >= threshold else 0


def _change_type(
    *,
    label: int,
    baseline_predicted_label: int,
    crop_first_predicted_label: int,
    positive_label: int,
) -> str:
    if baseline_predicted_label == crop_first_predicted_label:
        return "unchanged"
    if label == positive_label and baseline_predicted_label != positive_label:
        return "false_negative_recovered"
    if label == positive_label and crop_first_predicted_label != positive_label:
        return "new_false_negative"
    if label != positive_label and baseline_predicted_label == positive_label:
        return "false_positive_removed"
    if label != positive_label and crop_first_predicted_label == positive_label:
        return "new_false_positive"
    return "changed_prediction"


def _recall_impact(change_type: str) -> str:
    if change_type == "false_negative_recovered":
        return "improved"
    if change_type == "new_false_negative":
        return "regressed"
    return "none"


def _build_comparison_rows(
    *,
    baseline_predictions: Sequence[BaselinePrediction],
    crop_first_predictions: Sequence[BaselinePrediction],
    baseline_threshold: float,
    crop_first_threshold: float,
    review_split: str,
    positive_label: int,
) -> tuple[CropFirstReviewRow, ...]:
    unranked_rows: list[CropFirstReviewRow] = []
    for baseline, crop_first in zip(baseline_predictions, crop_first_predictions):
        if baseline.split != review_split:
            continue

        baseline_predicted_label = _predicted_label(
            baseline.score,
            threshold=baseline_threshold,
            positive_label=positive_label,
        )
        crop_first_predicted_label = _predicted_label(
            crop_first.score,
            threshold=crop_first_threshold,
            positive_label=positive_label,
        )
        change_type = _change_type(
            label=baseline.label,
            baseline_predicted_label=baseline_predicted_label,
            crop_first_predicted_label=crop_first_predicted_label,
            positive_label=positive_label,
        )
        baseline_false_negative = (
            baseline.label == positive_label
            and baseline_predicted_label != positive_label
        )
        crop_first_false_negative = (
            baseline.label == positive_label
            and crop_first_predicted_label != positive_label
        )
        unranked_rows.append(
            CropFirstReviewRow(
                rank=0,
                image_path=baseline.image_path,
                group_id=baseline.group_id,
                split=baseline.split,
                label=baseline.label,
                baseline_score=float(baseline.score),
                crop_first_score=float(crop_first.score),
                score_delta=float(crop_first.score - baseline.score),
                baseline_threshold=float(baseline_threshold),
                crop_first_threshold=float(crop_first_threshold),
                baseline_predicted_label=baseline_predicted_label,
                crop_first_predicted_label=crop_first_predicted_label,
                baseline_false_negative=baseline_false_negative,
                crop_first_false_negative=crop_first_false_negative,
                change_type=change_type,
                recall_impact=_recall_impact(change_type),
            )
        )

    ranked = sorted(
        unranked_rows,
        key=lambda row: (
            row.change_type == "unchanged",
            -abs(row.score_delta),
            row.group_id,
            row.image_path,
        ),
    )
    return tuple(
        CropFirstReviewRow(
            rank=index,
            image_path=row.image_path,
            group_id=row.group_id,
            split=row.split,
            label=row.label,
            baseline_score=row.baseline_score,
            crop_first_score=row.crop_first_score,
            score_delta=row.score_delta,
            baseline_threshold=row.baseline_threshold,
            crop_first_threshold=row.crop_first_threshold,
            baseline_predicted_label=row.baseline_predicted_label,
            crop_first_predicted_label=row.crop_first_predicted_label,
            baseline_false_negative=row.baseline_false_negative,
            crop_first_false_negative=row.crop_first_false_negative,
            change_type=row.change_type,
            recall_impact=row.recall_impact,
        )
        for index, row in enumerate(ranked, start=1)
    )


def compare_crop_first_to_baseline(
    baseline_predictions: Iterable[BaselinePrediction],
    crop_first_predictions: Iterable[BaselinePrediction],
    *,
    min_recall: float = 0.75,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    positive_label: int = 1,
    crop_emphasis: float = 1.35,
    sample_mode: bool = True,
    split_seed: int | None = None,
    feature_seed: int | None = None,
    review_split: str = "test",
) -> CropFirstABResult:
    """Select validation thresholds and compare test metrics for both arms."""

    baseline_rows = tuple(baseline_predictions)
    crop_first_rows = tuple(crop_first_predictions)
    if not baseline_rows:
        raise ValueError("crop-first A/B requires at least one prediction")
    _validate_same_prediction_frame(baseline_rows, crop_first_rows)

    baseline = _evaluate_arm(
        baseline_rows,
        arm_name=BASELINE_SCORE_SOURCE,
        score_source=BASELINE_SCORE_SOURCE,
        min_recall=min_recall,
        thresholds=thresholds,
        positive_label=positive_label,
    )
    crop_first = _evaluate_arm(
        crop_first_rows,
        arm_name=CROP_FIRST_SCORE_SOURCE,
        score_source=CROP_FIRST_SCORE_SOURCE,
        min_recall=min_recall,
        thresholds=thresholds,
        positive_label=positive_label,
    )
    comparison_rows = _build_comparison_rows(
        baseline_predictions=baseline_rows,
        crop_first_predictions=crop_first_rows,
        baseline_threshold=baseline.selected_point.threshold,
        crop_first_threshold=crop_first.selected_point.threshold,
        review_split=review_split,
        positive_label=positive_label,
    )
    if not comparison_rows:
        raise ValueError(f"crop-first review requires {review_split!r} predictions")

    return CropFirstABResult(
        baseline=baseline,
        crop_first=crop_first,
        comparison_rows=comparison_rows,
        sample_size=len(baseline_rows),
        group_count=len({prediction.group_id for prediction in baseline_rows}),
        train_size=len(_split_predictions(baseline_rows, "train")),
        val_size=len(_split_predictions(baseline_rows, "val")),
        test_size=len(_split_predictions(baseline_rows, "test")),
        min_recall=float(min_recall),
        crop_emphasis=float(crop_emphasis),
        sample_mode=sample_mode,
        split_seed=split_seed,
        feature_seed=feature_seed,
        review_split=review_split,
    )


def _split_indices(records: Sequence[DatasetRecord], split: str) -> list[int]:
    return [index for index, record in enumerate(records) if record.split == split]


def _predictions_from_scores(
    records: Sequence[DatasetRecord],
    scores: Sequence[float],
) -> tuple[BaselinePrediction, ...]:
    if len(records) != len(scores):
        raise ValueError("records and scores must have the same row count")
    return tuple(
        BaselinePrediction(
            image_path=record.image_path,
            group_id=record.group_id,
            split=str(record.split),
            label=record.label,
            score=float(score),
        )
        for record, score in zip(records, scores)
    )


def run_sample_crop_first_ab(
    *,
    sample_mode: bool = True,
    split_seed: int = 1,
    feature_seed: int = 17,
    min_recall: float = 0.75,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    crop_emphasis: float = 1.35,
) -> CropFirstABResult:
    """Run whole-image and crop-first sample scorers on the same split."""

    if not sample_mode:
        raise NotImplementedError(
            "Full-dataset crop-first A/B is deferred. Use sample_mode=True for "
            "the CPU smoke path."
        )

    records, whole_image_features = build_sample_baseline_dataset(seed=feature_seed)
    split_records = make_grouped_split(records, seed=split_seed)
    validate_group_splits(split_records)

    labels = np.asarray([record.label for record in split_records], dtype=int)
    train_indices = _split_indices(split_records, "train")
    if not train_indices:
        raise ValueError("crop-first A/B requires train predictions")

    baseline_model = train_centroid_baseline(
        whole_image_features[train_indices],
        labels[train_indices],
    )
    baseline_scores = baseline_model.score_samples(whole_image_features)

    crop_first_features = build_crop_first_sample_features(
        whole_image_features,
        crop_emphasis=crop_emphasis,
    )
    crop_first_model = train_centroid_baseline(
        crop_first_features[train_indices],
        labels[train_indices],
    )
    crop_first_scores = crop_first_model.score_samples(crop_first_features)

    baseline_predictions = _predictions_from_scores(split_records, baseline_scores)
    crop_first_predictions = _predictions_from_scores(split_records, crop_first_scores)
    return compare_crop_first_to_baseline(
        baseline_predictions,
        crop_first_predictions,
        min_recall=min_recall,
        thresholds=thresholds,
        crop_emphasis=crop_emphasis,
        sample_mode=sample_mode,
        split_seed=split_seed,
        feature_seed=feature_seed,
    )
